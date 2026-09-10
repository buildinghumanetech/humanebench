//! Local SQLite store: turns, scores, and the content-hash cache.
//!
//! The cache is the reason a given turn crosses the trust boundary at most once, ever.
//! `content_hash` covers the exact bytes sent to the judge, so re-running `score` after a
//! rubric revision or a model swap correctly re-judges while re-running it unchanged
//! spends nothing.
//!
//! `content_hash` is deliberately *not* what identifies a row. Two different turns can
//! produce byte-identical prompts — "thanks" answered "You're welcome!" in two sessions —
//! and both deserve a score. A row is identified by `(identity, tier, judge_model)`
//! instead: what was judged, at which tier, by which model. That is also what makes a
//! rollup supersede rather than accumulate when its arc grows.

use crate::judge::{Judgement, PrincipleScore, ScoreRecord, Tier};
use crate::transcript::{Action, Record, Role};
use anyhow::{bail, Context, Result};
use chrono::{DateTime, Utc};
use rusqlite::{params, Connection, OptionalExtension};
use std::path::{Path, PathBuf};

/// The schema this build understands. Bump it and add a step to [`Store::migrate`]
/// whenever the schema changes — an existing user database is otherwise left on the old
/// shape, and the change silently does nothing for everyone who already has one.
pub const SCHEMA_VERSION: i64 = 2;

pub struct Store {
    conn: Connection,
}

/// Filters shared by report rendering and the MCP query tools.
#[derive(Debug, Clone, Default)]
pub struct Filter {
    pub since: Option<DateTime<Utc>>,
    pub until: Option<DateTime<Utc>>,
    pub source: Option<String>,
    pub model: Option<String>,
    pub tier: Option<Tier>,
    pub session_id: Option<String>,
}

const LEGACY_CONSENT_KEY: &str = "consent.judge";

fn consent_key(destination: &str) -> String {
    format!("{LEGACY_CONSENT_KEY}:{destination}")
}

/// Where the store lives by default.
pub fn default_db_path() -> PathBuf {
    let base = std::env::var("HUMANEBENCH_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
            PathBuf::from(home).join(".humanebench")
        });
    base.join("humanebench.db")
}

impl Store {
    pub fn open(path: &Path) -> Result<Store> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
        let conn = Connection::open(path)
            .with_context(|| format!("opening store at {}", path.display()))?;
        let mut store = Store { conn };
        store
            .migrate()
            .with_context(|| format!("migrating the store at {}", path.display()))?;
        Ok(store)
    }

    #[cfg(test)]
    pub fn open_in_memory() -> Result<Store> {
        let mut store = Store {
            conn: Connection::open_in_memory()?,
        };
        store.migrate()?;
        Ok(store)
    }

    /// The schema version this database is currently on.
    pub fn schema_version(&self) -> Result<i64> {
        Ok(self
            .conn
            .query_row("PRAGMA user_version", [], |r| r.get(0))?)
    }

    /// `PRAGMA user_version` lives in the database header, so setting it inside a
    /// transaction commits and rolls back with everything else the step did. A step that
    /// is interrupted therefore leaves the version it started at and is simply retried.
    fn stamp_version(tx: &rusqlite::Transaction<'_>, v: i64) -> Result<()> {
        // PRAGMA takes no bound parameters; `v` is an i64 this module chose, not input.
        tx.execute_batch(&format!("PRAGMA user_version = {v}"))?;
        Ok(())
    }

    /// Bring the database up to [`SCHEMA_VERSION`], one numbered step at a time.
    ///
    /// A database written before versioning existed reports `user_version = 0`, which is
    /// also what a brand-new file reports — so step 1 is written to be correct for both:
    /// it only creates what is missing. Every later step must assume it is running
    /// against real user data that has already been paid for.
    fn migrate(&mut self) -> Result<()> {
        self.conn
            .execute_batch("PRAGMA journal_mode = WAL; PRAGMA foreign_keys = ON;")?;

        let mut version = self.schema_version()?;
        if version > SCHEMA_VERSION {
            bail!(
                "this store was written by a newer humanebench (schema v{version}; this \
                 build understands v{SCHEMA_VERSION}). Upgrade humanebench, or point --db \
                 at a different file."
            );
        }

        if version < 1 {
            self.migrate_to_v1()?;
            version = 1;
        }
        if version < 2 {
            self.migrate_to_v2()?;
            version = 2;
        }

        debug_assert_eq!(version, SCHEMA_VERSION, "migrate must reach SCHEMA_VERSION");
        Ok(())
    }

    /// v1: the original shape. `IF NOT EXISTS` throughout, because this also runs against
    /// pre-versioning databases that already have every one of these objects.
    fn migrate_to_v1(&mut self) -> Result<()> {
        let tx = self.conn.transaction()?;
        tx.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS turns (
                turn_id     TEXT PRIMARY KEY,
                session_id  TEXT NOT NULL,
                source      TEXT NOT NULL,
                role        TEXT NOT NULL,
                text        TEXT NOT NULL,
                timestamp   TEXT NOT NULL,
                parent_id   TEXT,
                model       TEXT,
                actions     TEXT NOT NULL DEFAULT '[]',
                sidechain   INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);
            CREATE INDEX IF NOT EXISTS idx_turns_ts      ON turns(timestamp);

            CREATE TABLE IF NOT EXISTS scores (
                content_hash      TEXT PRIMARY KEY,
                turn_id           TEXT NOT NULL,
                session_id        TEXT NOT NULL,
                tier              TEXT NOT NULL,
                judge_model       TEXT NOT NULL,
                regime            TEXT NOT NULL,
                scored_at         TEXT NOT NULL,
                principles        TEXT NOT NULL,
                global_violations TEXT NOT NULL,
                confidence        REAL NOT NULL,
                source            TEXT NOT NULL DEFAULT '',
                model             TEXT,
                timestamp         TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            "#,
        )?;
        Self::stamp_version(&tx, 1)?;
        tx.commit()?;
        Ok(())
    }

    /// v2: a score row is identified by what it judged, not by the bytes it judged.
    ///
    /// SQLite cannot drop a primary key in place, so `scores` is rebuilt. The copy keys
    /// each row exactly the way the runtime will, and keeps the newest when the old table
    /// already holds several for one subject — which is exactly what a database that has
    /// been ingested and re-scored a few times contains.
    fn migrate_to_v2(&mut self) -> Result<()> {
        let existing = self.v1_score_rows()?;
        let tx = self.conn.transaction()?;
        tx.execute_batch(
            r#"
            CREATE TABLE scores_v2 (
                identity          TEXT NOT NULL,
                tier              TEXT NOT NULL,
                judge_model       TEXT NOT NULL,
                content_hash      TEXT NOT NULL,
                turn_id           TEXT NOT NULL,
                session_id        TEXT NOT NULL,
                regime            TEXT NOT NULL,
                scored_at         TEXT NOT NULL,
                principles        TEXT NOT NULL,
                global_violations TEXT NOT NULL,
                confidence        REAL NOT NULL,
                source            TEXT NOT NULL DEFAULT '',
                model             TEXT,
                timestamp         TEXT NOT NULL,
                PRIMARY KEY (identity, tier, judge_model)
            );
            "#,
        )?;
        {
            let mut stmt = tx.prepare(&score_upsert_sql("scores_v2"))?;
            for row in &existing {
                stmt.execute(params![
                    row.identity,
                    row.tier,
                    row.judge_model,
                    row.content_hash,
                    row.turn_id,
                    row.session_id,
                    row.regime,
                    row.scored_at,
                    row.principles,
                    row.global_violations,
                    row.confidence,
                    row.source,
                    row.model,
                    row.timestamp,
                ])?;
            }
        }
        tx.execute_batch(
            r#"
            DROP TABLE scores;
            ALTER TABLE scores_v2 RENAME TO scores;
            CREATE INDEX IF NOT EXISTS idx_scores_turn    ON scores(turn_id);
            CREATE INDEX IF NOT EXISTS idx_scores_session ON scores(session_id);
            CREATE INDEX IF NOT EXISTS idx_scores_ts      ON scores(timestamp);
            CREATE INDEX IF NOT EXISTS idx_scores_tier    ON scores(tier);
            CREATE INDEX IF NOT EXISTS idx_scores_hash    ON scores(content_hash);
            "#,
        )?;
        Self::stamp_version(&tx, 2)?;
        tx.commit()?;
        Ok(())
    }

    /// Read the v1 `scores` table, oldest judgement first, tagging each row with the
    /// identity it will be keyed on. Oldest-first means a later duplicate overwrites an
    /// earlier one during the copy, so a session that accumulated several rollups keeps
    /// the one judged on the most complete arc.
    fn v1_score_rows(&self) -> Result<Vec<V1ScoreRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT content_hash, turn_id, session_id, tier, judge_model, regime, scored_at,
                    principles, global_violations, confidence, source, model, timestamp
             FROM scores ORDER BY scored_at, content_hash",
        )?;
        let rows = stmt.query_map([], |r| {
            let tier: String = r.get(3)?;
            let turn_id: String = r.get(1)?;
            let session_id: String = r.get(2)?;
            let timestamp: String = r.get(12)?;
            // v1 stamped a rollup with the session's start instant, which is precisely
            // what its identity is built from.
            let identity = if tier == "rollup" {
                rollup_identity_raw(&session_id, &timestamp)
            } else {
                turn_id.clone()
            };
            Ok(V1ScoreRow {
                identity,
                tier,
                judge_model: r.get(4)?,
                content_hash: r.get(0)?,
                turn_id,
                session_id,
                regime: r.get(5)?,
                scored_at: r.get(6)?,
                principles: r.get(7)?,
                global_violations: r.get(8)?,
                confidence: r.get(9)?,
                source: r.get(10)?,
                model: r.get(11)?,
                timestamp,
            })
        })?;
        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    }

    // ---- consent -------------------------------------------------------------

    /// Consent is per destination. Agreeing to send transcripts to one place is not
    /// agreement to send them to another, so changing provider, project, or region asks
    /// again.
    pub fn consent_granted(&self, destination: &str) -> Result<bool> {
        if self.meta_get(&consent_key(destination))?.as_deref() == Some("granted") {
            return Ok(true);
        }
        // Consent recorded before it was keyed to a destination could only have been for
        // OpenRouter, which was the only backend at the time.
        Ok(destination == crate::judge::openrouter::DESTINATION
            && self.meta_get(LEGACY_CONSENT_KEY)?.as_deref() == Some("granted"))
    }

    pub fn grant_consent(&self, destination: &str) -> Result<()> {
        let key = consent_key(destination);
        self.meta_set(&key, "granted")?;
        self.meta_set(&format!("{key}.at"), &Utc::now().to_rfc3339())
    }

    pub fn meta_get(&self, key: &str) -> Result<Option<String>> {
        Ok(self
            .conn
            .query_row("SELECT value FROM meta WHERE key = ?1", params![key], |r| {
                r.get::<_, String>(0)
            })
            .optional()?)
    }

    pub fn meta_set(&self, key: &str, value: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES(?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![key, value],
        )?;
        Ok(())
    }

    // ---- turns ---------------------------------------------------------------

    /// Insert or refresh normalized records. Transcripts on disk are read-only and never
    /// modified; this is the only place their content is copied.
    pub fn upsert_records(&mut self, records: &[Record]) -> Result<usize> {
        let tx = self.conn.transaction()?;
        let mut n = 0;
        {
            let mut stmt = tx.prepare(
                "INSERT INTO turns(turn_id, session_id, source, role, text, timestamp,
                                   parent_id, model, actions, sidechain)
                 VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)
                 ON CONFLICT(turn_id) DO UPDATE SET
                   session_id=excluded.session_id, source=excluded.source, role=excluded.role,
                   text=excluded.text, timestamp=excluded.timestamp, parent_id=excluded.parent_id,
                   model=excluded.model, actions=excluded.actions, sidechain=excluded.sidechain",
            )?;
            for r in records {
                stmt.execute(params![
                    r.turn_id,
                    r.session_id,
                    r.source,
                    match r.role {
                        Role::User => "user",
                        Role::Assistant => "assistant",
                    },
                    r.text,
                    r.timestamp.to_rfc3339(),
                    r.parent_id,
                    r.model,
                    serde_json::to_string(&r.actions)?,
                    r.sidechain as i32,
                ])?;
                n += 1;
            }
        }
        tx.commit()?;
        Ok(n)
    }

    pub fn all_records(&self) -> Result<Vec<Record>> {
        let mut stmt = self.conn.prepare(
            "SELECT turn_id, session_id, source, role, text, timestamp, parent_id,
                    model, actions, sidechain
             FROM turns ORDER BY timestamp, turn_id",
        )?;
        let rows = stmt.query_map([], |r| {
            let ts: String = r.get(5)?;
            let actions: String = r.get(8)?;
            let role: String = r.get(3)?;
            Ok(Record {
                schema: crate::transcript::SCHEMA.to_string(),
                turn_id: r.get(0)?,
                session_id: r.get(1)?,
                source: r.get(2)?,
                role: if role == "user" {
                    Role::User
                } else {
                    Role::Assistant
                },
                text: r.get(4)?,
                timestamp: DateTime::parse_from_rfc3339(&ts)
                    .map(|t| t.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now()),
                parent_id: r.get(6)?,
                model: r.get(7)?,
                actions: serde_json::from_str::<Vec<Action>>(&actions).unwrap_or_default(),
                sidechain: r.get::<_, i32>(9)? != 0,
            })
        })?;
        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    }

    pub fn turn_text(&self, turn_id: &str) -> Result<Option<String>> {
        Ok(self
            .conn
            .query_row(
                "SELECT text FROM turns WHERE turn_id = ?1",
                params![turn_id],
                |r| r.get::<_, String>(0),
            )
            .optional()?)
    }

    pub fn count_turns(&self) -> Result<i64> {
        Ok(self
            .conn
            .query_row("SELECT COUNT(*) FROM turns", [], |r| r.get(0))?)
    }

    // ---- scores / cache ------------------------------------------------------

    /// The cache check: does this subject already carry a score produced from these exact
    /// bytes?
    ///
    /// Keyed on the subject as well as the bytes. Asking only "were these bytes judged?"
    /// is what made a second, byte-identical turn read as already-scored and never get a
    /// row of its own — it was billed and then thrown away.
    pub fn is_scored(
        &self,
        identity: &str,
        tier: Tier,
        judge_model: &str,
        content_hash: &str,
    ) -> Result<bool> {
        let n: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM scores
             WHERE identity = ?1 AND tier = ?2 AND judge_model = ?3 AND content_hash = ?4",
            params![identity, tier.as_str(), judge_model, content_hash],
            |r| r.get(0),
        )?;
        Ok(n > 0)
    }

    /// A judgement already recorded for these exact bytes, whatever subject it was
    /// recorded against.
    ///
    /// This is what keeps the promise that a given prompt crosses the trust boundary at
    /// most once, ever: a byte-identical turn in another session is filled from the
    /// judgement already paid for rather than sent again.
    pub fn judgement_for_hash(&self, content_hash: &str) -> Result<Option<Judgement>> {
        let row = self
            .conn
            .query_row(
                "SELECT principles, global_violations, confidence FROM scores
                 WHERE content_hash = ?1 LIMIT 1",
                params![content_hash],
                |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, f64>(2)?,
                    ))
                },
            )
            .optional()?;
        Ok(row.map(|(principles, violations, confidence)| Judgement {
            principles: serde_json::from_str::<Vec<PrincipleScore>>(&principles)
                .unwrap_or_default(),
            global_violations: serde_json::from_str::<Vec<String>>(&violations).unwrap_or_default(),
            confidence,
        }))
    }

    /// Convenience for tests: stamp the score with its own `scored_at`.
    #[cfg(test)]
    pub fn insert_score(&self, rec: &ScoreRecord, source: &str, model: Option<&str>) -> Result<()> {
        self.insert_score_at(rec, source, model, rec.scored_at)
    }

    /// Insert a score under the identity implied by the record itself.
    ///
    /// `row_ts` is when the conversation happened, not when it was judged, so trends plot
    /// against the transcript. Rollups whose session id may be renumbered by a later
    /// idle-gap split must go through [`Store::insert_score_as`] with a stable identity
    /// instead — see [`rollup_identity`].
    pub fn insert_score_at(
        &self,
        rec: &ScoreRecord,
        source: &str,
        model: Option<&str>,
        row_ts: DateTime<Utc>,
    ) -> Result<()> {
        self.insert_score_as(rec, &rec.turn_id, source, model, row_ts)
    }

    /// Insert a score, superseding whatever this subject last scored at this tier from
    /// this judge model.
    ///
    /// Superseding rather than accumulating is the point: a rollup's prompt covers the
    /// rendered session arc, which grows on every ingest, so an append-only table ends up
    /// averaging the early, truncated arcs in with the complete one.
    pub fn insert_score_as(
        &self,
        rec: &ScoreRecord,
        identity: &str,
        source: &str,
        model: Option<&str>,
        row_ts: DateTime<Utc>,
    ) -> Result<()> {
        self.conn.execute(
            &score_upsert_sql("scores"),
            params![
                identity,
                rec.tier.as_str(),
                rec.judge_model,
                rec.content_hash,
                rec.turn_id,
                rec.session_id,
                rec.regime,
                rec.scored_at.to_rfc3339(),
                serde_json::to_string(&rec.principles)?,
                serde_json::to_string(&rec.global_violations)?,
                rec.confidence,
                source,
                model,
                row_ts.to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    /// A scored row joined with what it needs for the report.
    pub fn scores(&self, filter: &Filter) -> Result<Vec<ScoredTurn>> {
        let mut sql = String::from(
            "SELECT s.content_hash, s.turn_id, s.session_id, s.tier, s.judge_model, s.regime,
                    s.scored_at, s.principles, s.global_violations, s.confidence,
                    s.source, s.model, s.timestamp
             FROM scores s WHERE 1=1",
        );
        let mut args: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();

        if let Some(since) = filter.since {
            sql.push_str(" AND s.timestamp >= ?");
            args.push(Box::new(since.to_rfc3339()));
        }
        if let Some(until) = filter.until {
            sql.push_str(" AND s.timestamp <= ?");
            args.push(Box::new(until.to_rfc3339()));
        }
        if let Some(src) = &filter.source {
            sql.push_str(" AND s.source = ?");
            args.push(Box::new(src.clone()));
        }
        if let Some(model) = &filter.model {
            sql.push_str(" AND s.model = ?");
            args.push(Box::new(model.clone()));
        }
        if let Some(tier) = filter.tier {
            sql.push_str(" AND s.tier = ?");
            args.push(Box::new(tier.as_str().to_string()));
        }
        if let Some(sid) = &filter.session_id {
            sql.push_str(" AND s.session_id = ?");
            args.push(Box::new(sid.clone()));
        }
        sql.push_str(" ORDER BY s.timestamp, s.turn_id");

        let mut stmt = self.conn.prepare(&sql)?;
        let refs: Vec<&dyn rusqlite::ToSql> = args.iter().map(|b| b.as_ref()).collect();

        let rows = stmt.query_map(refs.as_slice(), |r| {
            let principles: String = r.get(7)?;
            let violations: String = r.get(8)?;
            let scored_at: String = r.get(6)?;
            let ts: String = r.get(12)?;
            let tier: String = r.get(3)?;
            Ok(ScoredTurn {
                record: ScoreRecord {
                    content_hash: r.get(0)?,
                    turn_id: r.get(1)?,
                    session_id: r.get(2)?,
                    tier: if tier == "rollup" {
                        Tier::Rollup
                    } else {
                        Tier::Turn
                    },
                    judge_model: r.get(4)?,
                    regime: r.get(5)?,
                    scored_at: DateTime::parse_from_rfc3339(&scored_at)
                        .map(|t| t.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now()),
                    principles: serde_json::from_str::<Vec<PrincipleScore>>(&principles)
                        .unwrap_or_default(),
                    global_violations: serde_json::from_str::<Vec<String>>(&violations)
                        .unwrap_or_default(),
                    confidence: r.get(9)?,
                },
                source: r.get(10)?,
                model: r.get(11)?,
                timestamp: DateTime::parse_from_rfc3339(&ts)
                    .map(|t| t.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now()),
            })
        })?;

        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    }

    /// Distinct values, for report filters and MCP facets.
    pub fn distinct(&self, column: &str) -> Result<Vec<String>> {
        let sql = match column {
            "source" => "SELECT DISTINCT source FROM scores WHERE source <> '' ORDER BY source",
            "model" => "SELECT DISTINCT model FROM scores WHERE model IS NOT NULL ORDER BY model",
            "judge_model" => "SELECT DISTINCT judge_model FROM scores ORDER BY judge_model",
            "regime" => "SELECT DISTINCT regime FROM scores ORDER BY regime",
            _ => return Ok(Vec::new()),
        };
        let mut stmt = self.conn.prepare(sql)?;
        let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    }
}

/// One row of the pre-v2 `scores` table, tagged with the identity it will be keyed on.
struct V1ScoreRow {
    identity: String,
    tier: String,
    judge_model: String,
    content_hash: String,
    turn_id: String,
    session_id: String,
    regime: String,
    scored_at: String,
    principles: String,
    global_violations: String,
    confidence: f64,
    source: String,
    model: Option<String>,
    timestamp: String,
}

/// The one upsert both the runtime and the v2 migration write through, so the two can
/// never disagree about what supersedes what.
fn score_upsert_sql(table: &str) -> String {
    format!(
        "INSERT INTO {table}(identity, tier, judge_model, content_hash, turn_id, session_id,
                             regime, scored_at, principles, global_violations, confidence,
                             source, model, timestamp)
         VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14)
         ON CONFLICT(identity, tier, judge_model) DO UPDATE SET
           content_hash=excluded.content_hash,
           turn_id=excluded.turn_id,
           session_id=excluded.session_id,
           regime=excluded.regime,
           scored_at=excluded.scored_at,
           principles=excluded.principles,
           global_violations=excluded.global_violations,
           confidence=excluded.confidence,
           source=excluded.source,
           model=excluded.model,
           timestamp=excluded.timestamp"
    )
}

/// The thread a session id belongs to, with any `#N` sub-session suffix removed.
///
/// `transcript::sessionize` numbers sub-sessions positionally, so the suffix is a
/// property of how the thread happens to be split *right now*: a thread that has only
/// ever been one chunk is `sA`, and becomes `sA#1` the first time an idle gap splits it.
/// The root is the part that does not move.
pub fn thread_root(session_id: &str) -> &str {
    session_id.split('#').next().unwrap_or(session_id)
}

/// The stable identity of a session's rollup row.
///
/// Keying the rollup on the rendered session id would strand the pre-split row as an
/// orphan that no later run can supersede, so the key is the thread root plus the instant
/// the chunk begins — neither of which the renumbering touches.
pub fn rollup_identity(session_id: &str, started_at: DateTime<Utc>) -> String {
    rollup_identity_raw(session_id, &started_at.to_rfc3339())
}

fn rollup_identity_raw(session_id: &str, started_at_rfc3339: &str) -> String {
    format!("{}@{started_at_rfc3339}:rollup", thread_root(session_id))
}

/// A score plus the denormalized fields the report and MCP need.
#[derive(Debug, Clone)]
pub struct ScoredTurn {
    pub record: ScoreRecord,
    pub source: String,
    pub model: Option<String>,
    pub timestamp: DateTime<Utc>,
}

/// Build a `ScoreRecord` from a validated judgement.
pub fn score_record(
    turn_id: &str,
    session_id: &str,
    tier: Tier,
    content_hash: &str,
    judge_model: &str,
    regime: &str,
    judgement: Judgement,
) -> ScoreRecord {
    ScoreRecord {
        turn_id: turn_id.to_string(),
        session_id: session_id.to_string(),
        tier,
        content_hash: content_hash.to_string(),
        judge_model: judge_model.to_string(),
        regime: regime.to_string(),
        scored_at: Utc::now(),
        principles: judgement.principles,
        global_violations: judgement.global_violations,
        confidence: judgement.confidence,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::judge::{Judgement, PrincipleScore, PRINCIPLES};
    use chrono::TimeZone;

    fn judgement() -> Judgement {
        Judgement {
            principles: PRINCIPLES
                .iter()
                .map(|n| PrincipleScore {
                    name: n.to_string(),
                    score: 0.5,
                    rationale: None,
                })
                .collect(),
            global_violations: vec![],
            confidence: 0.8,
        }
    }

    fn rec(id: &str, ts: &str) -> Record {
        Record::new(
            "claude-code",
            "s1",
            id,
            Role::Assistant,
            format!("text {id}"),
            DateTime::parse_from_rfc3339(ts)
                .unwrap()
                .with_timezone(&Utc),
        )
    }

    #[test]
    fn round_trips_records() {
        let mut s = Store::open_in_memory().unwrap();
        let recs = vec![
            rec("t1", "2026-01-01T00:00:00Z"),
            rec("t2", "2026-01-02T00:00:00Z"),
        ];
        assert_eq!(s.upsert_records(&recs).unwrap(), 2);
        assert_eq!(s.count_turns().unwrap(), 2);
        assert_eq!(s.turn_text("t1").unwrap().unwrap(), "text t1");
        assert_eq!(s.all_records().unwrap().len(), 2);
    }

    #[test]
    fn upsert_is_idempotent() {
        let mut s = Store::open_in_memory().unwrap();
        let recs = vec![rec("t1", "2026-01-01T00:00:00Z")];
        s.upsert_records(&recs).unwrap();
        s.upsert_records(&recs).unwrap();
        assert_eq!(s.count_turns().unwrap(), 1);
    }

    #[test]
    fn cache_stops_a_turn_being_judged_twice() {
        let s = Store::open_in_memory().unwrap();
        let r = score_record(
            "t1",
            "s1",
            Tier::Turn,
            "blake3:abc",
            "m",
            "single",
            judgement(),
        );
        assert!(!s.is_scored("t1", Tier::Turn, "m", "blake3:abc").unwrap());
        s.insert_score(&r, "claude-code", Some("claude-opus-5"))
            .unwrap();
        assert!(s.is_scored("t1", Tier::Turn, "m", "blake3:abc").unwrap());
        // Re-inserting the same subject is a supersede, not a duplicate row.
        s.insert_score(&r, "claude-code", Some("claude-opus-5"))
            .unwrap();
        assert_eq!(s.scores(&Filter::default()).unwrap().len(), 1);
    }

    #[test]
    fn a_second_turn_with_the_same_bytes_reuses_the_judgement() {
        // The whole promise of the cache is that a given prompt crosses the trust
        // boundary at most once, ever. A turn elsewhere in the corpus that assembles to
        // the same bytes must be fillable from what was already paid for.
        let s = Store::open_in_memory().unwrap();
        assert!(s.judgement_for_hash("blake3:abc").unwrap().is_none());
        let r = score_record("t1", "sA", Tier::Turn, "blake3:abc", "m", "s", judgement());
        s.insert_score(&r, "claude-code", None).unwrap();
        let reused = s.judgement_for_hash("blake3:abc").unwrap().unwrap();
        assert_eq!(reused.principles.len(), PRINCIPLES.len());
        assert!((reused.confidence - 0.8).abs() < 1e-9);
    }

    #[test]
    fn a_different_hash_is_a_miss() {
        let s = Store::open_in_memory().unwrap();
        let r = score_record(
            "t1",
            "s1",
            Tier::Turn,
            "blake3:abc",
            "m",
            "single",
            judgement(),
        );
        s.insert_score(&r, "claude-code", None).unwrap();
        assert!(
            !s.is_scored("t1", Tier::Turn, "m", "blake3:def").unwrap(),
            "a rubric revision changes the bytes, so it must miss"
        );
        assert!(
            !s.is_scored("t1", Tier::Turn, "other", "blake3:abc")
                .unwrap(),
            "a model swap must miss"
        );
    }

    #[test]
    fn filters_by_time_source_and_tier() {
        let s = Store::open_in_memory().unwrap();
        let t1 = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
        let t2 = Utc.with_ymd_and_hms(2026, 6, 1, 0, 0, 0).unwrap();

        let a = score_record("t1", "s1", Tier::Turn, "h1", "m", "single", judgement());
        let b = score_record("t2", "s1", Tier::Turn, "h2", "m", "single", judgement());
        let c = score_record("s1", "s1", Tier::Rollup, "h3", "m", "single", judgement());
        s.insert_score_at(&a, "claude-code", Some("opus"), t1)
            .unwrap();
        s.insert_score_at(&b, "codex", Some("gpt"), t2).unwrap();
        s.insert_score_at(&c, "claude-code", Some("opus"), t2)
            .unwrap();

        assert_eq!(s.scores(&Filter::default()).unwrap().len(), 3);

        let recent = Filter {
            since: Some(t2),
            ..Default::default()
        };
        assert_eq!(s.scores(&recent).unwrap().len(), 2);

        let cc = Filter {
            source: Some("claude-code".into()),
            ..Default::default()
        };
        assert_eq!(s.scores(&cc).unwrap().len(), 2);

        let turns = Filter {
            tier: Some(Tier::Turn),
            ..Default::default()
        };
        assert_eq!(s.scores(&turns).unwrap().len(), 2);

        let rollups = Filter {
            tier: Some(Tier::Rollup),
            ..Default::default()
        };
        assert_eq!(s.scores(&rollups).unwrap().len(), 1);
    }

    #[test]
    fn tiers_are_kept_separable() {
        // Averaging a rollup together with turn scores would be meaningless, so the
        // store must be able to hand them back apart.
        let s = Store::open_in_memory().unwrap();
        let a = score_record("t1", "s1", Tier::Turn, "h1", "m", "single", judgement());
        let c = score_record("s1", "s1", Tier::Rollup, "h3", "m", "single", judgement());
        s.insert_score(&a, "claude-code", None).unwrap();
        s.insert_score(&c, "claude-code", None).unwrap();
        let turns = s
            .scores(&Filter {
                tier: Some(Tier::Turn),
                ..Default::default()
            })
            .unwrap();
        assert!(turns.iter().all(|t| t.record.tier == Tier::Turn));
    }

    #[test]
    fn consent_is_off_until_granted_and_then_persists() {
        let s = Store::open_in_memory().unwrap();
        let dest = "Google Vertex AI (project acme-prod, location us-central1)";
        assert!(!s.consent_granted(dest).unwrap());
        s.grant_consent(dest).unwrap();
        assert!(s.consent_granted(dest).unwrap());
        assert!(s
            .meta_get(&format!("{}.at", consent_key(dest)))
            .unwrap()
            .is_some());
    }

    #[test]
    fn consent_does_not_carry_across_destinations() {
        let s = Store::open_in_memory().unwrap();
        s.grant_consent("Google Vertex AI (project acme-prod, location us-central1)")
            .unwrap();
        assert!(
            !s.consent_granted("Google Vertex AI (project personal, location us-central1)")
                .unwrap(),
            "a different project is a different destination"
        );
        assert!(!s
            .consent_granted(crate::judge::openrouter::DESTINATION)
            .unwrap());
    }

    #[test]
    fn consent_predating_destination_keying_still_covers_openrouter() {
        let s = Store::open_in_memory().unwrap();
        s.meta_set(LEGACY_CONSENT_KEY, "granted").unwrap();
        assert!(s
            .consent_granted(crate::judge::openrouter::DESTINATION)
            .unwrap());
        assert!(!s
            .consent_granted("Google Vertex AI (project acme-prod, location us-central1)")
            .unwrap());
    }

    #[test]
    fn overall_is_the_mean_of_eight() {
        let r = score_record("t1", "s1", Tier::Turn, "h", "m", "single", judgement());
        assert!((r.overall() - 0.5).abs() < 1e-9);
    }

    // ---- F2: identical bytes, different turns -------------------------------

    #[test]
    fn two_identical_turns_in_different_sessions_are_two_rows() {
        // "thanks" / "You're welcome!" in two sessions assembles to the same bytes, so
        // both carry the same content hash. Storing one and dropping the other makes the
        // second turn permanently invisible to every report.
        let s = Store::open_in_memory().unwrap();
        let a = score_record("t1", "sA", Tier::Turn, "h", "m", "single", judgement());
        let b = score_record("t2", "sB", Tier::Turn, "h", "m", "single", judgement());
        s.insert_score(&a, "claude-code", None).unwrap();
        s.insert_score(&b, "claude-code", None).unwrap();
        assert_eq!(
            s.scores(&Filter::default()).unwrap().len(),
            2,
            "a shared content hash must not collapse two turns into one row"
        );
    }

    // ---- F3: a rollup supersedes, it does not accumulate ---------------------

    #[test]
    fn a_regrown_arc_supersedes_the_rollup_it_replaces() {
        // The rollup hash covers the rendered arc, which grows on every ingest. Keeping
        // both rows means `report` averages the stub arc with the complete one.
        let s = Store::open_in_memory().unwrap();
        let ts = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
        let ident = rollup_identity("sA", ts);

        let first = score_record("sA:rollup", "sA", Tier::Rollup, "h1", "m", "s", judgement());
        s.insert_score_as(&first, &ident, "claude-code", None, ts)
            .unwrap();
        let second = score_record("sA:rollup", "sA", Tier::Rollup, "h2", "m", "s", judgement());
        s.insert_score_as(&second, &ident, "claude-code", None, ts)
            .unwrap();

        let rows = s.scores(&Filter::default()).unwrap();
        assert_eq!(
            rows.len(),
            1,
            "one rollup row per session, not one per ingest"
        );
        assert_eq!(
            rows[0].record.content_hash, "h2",
            "the surviving row must be the one judged on the complete arc"
        );
    }

    #[test]
    fn an_idle_gap_split_does_not_orphan_the_rollup_it_renames() {
        // `sessionize` numbers sub-sessions positionally, so a thread that has only ever
        // been one chunk is `sA` and becomes `sA#1` the first time a gap splits it. The
        // row's identity must not move with the label.
        let started = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
        assert_eq!(
            rollup_identity("sA", started),
            rollup_identity("sA#1", started),
            "renumbering a sub-session must not change which row it supersedes"
        );
        assert_ne!(
            rollup_identity("sA#1", started),
            rollup_identity("sA#2", started + chrono::Duration::days(7)),
            "genuinely different chunks stay different rows"
        );
    }

    // ---- F11: versioned migrations -------------------------------------------

    /// Write the pre-versioning schema by hand: `user_version` unset, `content_hash` the
    /// primary key on `scores`, no `identity` column.
    fn write_v0_database(path: &Path) {
        let conn = Connection::open(path).unwrap();
        conn.execute_batch(
            r#"
            CREATE TABLE turns (
                turn_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, source TEXT NOT NULL,
                role TEXT NOT NULL, text TEXT NOT NULL, timestamp TEXT NOT NULL,
                parent_id TEXT, model TEXT, actions TEXT NOT NULL DEFAULT '[]',
                sidechain INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE scores (
                content_hash TEXT PRIMARY KEY, turn_id TEXT NOT NULL,
                session_id TEXT NOT NULL, tier TEXT NOT NULL, judge_model TEXT NOT NULL,
                regime TEXT NOT NULL, scored_at TEXT NOT NULL, principles TEXT NOT NULL,
                global_violations TEXT NOT NULL, confidence REAL NOT NULL,
                source TEXT NOT NULL DEFAULT '', model TEXT, timestamp TEXT NOT NULL
            );
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

            INSERT INTO turns(turn_id, session_id, source, role, text, timestamp)
              VALUES('t1','sA','claude-code','assistant','hello','2026-01-01T00:00:00+00:00');
            INSERT INTO scores VALUES
              ('h1','t1','sA','turn','m','single','2026-01-01T00:00:00+00:00',
               '[]','[]',0.8,'claude-code','opus','2026-01-01T00:00:00+00:00'),
              ('h2','sA:rollup','sA','rollup','m','single','2026-01-01T00:00:00+00:00',
               '[]','[]',0.8,'claude-code',NULL,'2026-01-01T00:00:00+00:00');
            INSERT INTO meta VALUES('consent.judge','granted');
            "#,
        )
        .unwrap();
    }

    #[test]
    fn migrates_a_v0_database_forward_and_keeps_its_rows() {
        let dir = std::env::temp_dir().join(format!("hb-migrate-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("v0.db");
        let _ = std::fs::remove_file(&path);
        write_v0_database(&path);

        let s = Store::open(&path).unwrap();

        assert_eq!(s.schema_version().unwrap(), SCHEMA_VERSION);
        assert_eq!(s.count_turns().unwrap(), 1, "ingested turns must survive");
        let rows = s.scores(&Filter::default()).unwrap();
        assert_eq!(rows.len(), 2, "paid-for scores must survive");
        assert!(
            s.consent_granted(crate::judge::openrouter::DESTINATION)
                .unwrap(),
            "recorded consent must survive — re-asking would be a privacy regression"
        );

        // Idempotent: opening an already-migrated store changes nothing.
        drop(s);
        let s = Store::open(&path).unwrap();
        assert_eq!(s.schema_version().unwrap(), SCHEMA_VERSION);
        assert_eq!(s.scores(&Filter::default()).unwrap().len(), 2);

        drop(s);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn distinct_lists_facets() {
        let s = Store::open_in_memory().unwrap();
        let a = score_record("t1", "s1", Tier::Turn, "h1", "m", "single", judgement());
        s.insert_score(&a, "claude-code", Some("opus")).unwrap();
        assert_eq!(s.distinct("source").unwrap(), vec!["claude-code"]);
        assert_eq!(s.distinct("model").unwrap(), vec!["opus"]);
    }
}
