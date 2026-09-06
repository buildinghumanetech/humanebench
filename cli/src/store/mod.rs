//! Local SQLite store: turns, scores, and the content-hash cache.
//!
//! The cache is the reason a given turn crosses the trust boundary at most once, ever.
//! `content_hash` is the primary key on `scores`, so re-running `score` after a rubric
//! revision or a model swap correctly re-judges, while re-running it unchanged spends
//! nothing.

use crate::judge::{Judgement, PrincipleScore, ScoreRecord, Tier};
use crate::transcript::{Action, Record, Role};
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use rusqlite::{params, Connection, OptionalExtension};
use std::path::{Path, PathBuf};

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
        let store = Store { conn };
        store.migrate()?;
        Ok(store)
    }

    #[cfg(test)]
    pub fn open_in_memory() -> Result<Store> {
        let store = Store {
            conn: Connection::open_in_memory()?,
        };
        store.migrate()?;
        Ok(store)
    }

    fn migrate(&self) -> Result<()> {
        self.conn.execute_batch(
            r#"
            PRAGMA journal_mode = WAL;
            PRAGMA foreign_keys = ON;

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

            -- content_hash is the primary key: that IS the cache.
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
            CREATE INDEX IF NOT EXISTS idx_scores_turn    ON scores(turn_id);
            CREATE INDEX IF NOT EXISTS idx_scores_session ON scores(session_id);
            CREATE INDEX IF NOT EXISTS idx_scores_ts      ON scores(timestamp);
            CREATE INDEX IF NOT EXISTS idx_scores_tier    ON scores(tier);

            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            "#,
        )?;
        Ok(())
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

    /// The cache check. A hit means these exact bytes were already judged.
    pub fn has_score(&self, content_hash: &str) -> Result<bool> {
        let n: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM scores WHERE content_hash = ?1",
            params![content_hash],
            |r| r.get(0),
        )?;
        Ok(n > 0)
    }

    /// Convenience for tests: stamp the score with its own `scored_at`.
    #[cfg(test)]
    pub fn insert_score(&self, rec: &ScoreRecord, source: &str, model: Option<&str>) -> Result<()> {
        self.insert_score_at(rec, source, model, rec.scored_at)
    }

    /// Insert a score, stamping it with the turn's own timestamp so trends plot against
    /// when the conversation happened, not when it was judged.
    pub fn insert_score_at(
        &self,
        rec: &ScoreRecord,
        source: &str,
        model: Option<&str>,
        turn_ts: DateTime<Utc>,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT INTO scores(content_hash, turn_id, session_id, tier, judge_model, regime,
                                scored_at, principles, global_violations, confidence,
                                source, model, timestamp)
             VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13)
             ON CONFLICT(content_hash) DO NOTHING",
            params![
                rec.content_hash,
                rec.turn_id,
                rec.session_id,
                rec.tier.as_str(),
                rec.judge_model,
                rec.regime,
                rec.scored_at.to_rfc3339(),
                serde_json::to_string(&rec.principles)?,
                serde_json::to_string(&rec.global_violations)?,
                rec.confidence,
                source,
                model,
                turn_ts.to_rfc3339(),
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
        assert!(!s.has_score("blake3:abc").unwrap());
        s.insert_score(&r, "claude-code", Some("claude-opus-5"))
            .unwrap();
        assert!(s.has_score("blake3:abc").unwrap());
        // Re-inserting the same hash is a no-op, not a duplicate row.
        s.insert_score(&r, "claude-code", Some("claude-opus-5"))
            .unwrap();
        assert_eq!(s.scores(&Filter::default()).unwrap().len(), 1);
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
            !s.has_score("blake3:def").unwrap(),
            "model/rubric swap must miss"
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

    #[test]
    fn distinct_lists_facets() {
        let s = Store::open_in_memory().unwrap();
        let a = score_record("t1", "s1", Tier::Turn, "h1", "m", "single", judgement());
        s.insert_score(&a, "claude-code", Some("opus")).unwrap();
        assert_eq!(s.distinct("source").unwrap(), vec!["claude-code"]);
        assert_eq!(s.distinct("model").unwrap(), vec!["opus"]);
    }
}
