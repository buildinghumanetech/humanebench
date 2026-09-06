//! The normalized transcript schema — the entire contract between adapters and the engine.
//!
//! An adapter that emits this schema is a first-class citizen with no code in the binary.
//! Two rules live here rather than in any one adapter because they belong to the contract:
//! the flattening rule (trees collapse to the newest-leaf path) and idle-gap session
//! splitting.

use anyhow::{bail, Context, Result};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Literal schema tag. Lets the binary reject a format it can't read instead of guessing.
pub const SCHEMA: &str = "humanebench.transcript/v1";

/// Default idle gap that splits one thread into separate sessions.
pub const DEFAULT_IDLE_GAP_HOURS: i64 = 6;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

/// A tool, MCP call, or skill invoked while producing a turn. Context for the judge,
/// never scored on its own.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Action {
    pub name: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub summary: String,
}

/// One JSONL record per message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    pub schema: String,
    pub source: String,
    pub session_id: String,
    pub turn_id: String,
    pub role: Role,
    pub text: String,
    pub timestamp: DateTime<Utc>,

    /// Parent turn_id. Present for tree-shaped sources, absent for flat ones.
    /// Absence means "already linear".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,

    /// Model that produced an assistant turn. Without it the report can't break
    /// scores down by model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub actions: Vec<Action>,

    /// True marks a turn no human ever saw (subagent chatter). Excluded from scoring.
    #[serde(default)]
    pub sidechain: bool,
}

impl Record {
    /// Adapters build `Record` literally so every field is a deliberate choice; this
    /// shorthand exists for tests.
    #[cfg(test)]
    pub fn new(
        source: impl Into<String>,
        session_id: impl Into<String>,
        turn_id: impl Into<String>,
        role: Role,
        text: impl Into<String>,
        timestamp: DateTime<Utc>,
    ) -> Self {
        Record {
            schema: SCHEMA.to_string(),
            source: source.into(),
            session_id: session_id.into(),
            turn_id: turn_id.into(),
            role,
            text: text.into(),
            timestamp,
            parent_id: None,
            model: None,
            actions: Vec::new(),
            sidechain: false,
        }
    }

    /// Validate a record against the contract. Empty or whitespace-only text is a
    /// contract violation, not something to silently keep.
    pub fn validate(&self) -> Result<()> {
        if self.schema != SCHEMA {
            bail!(
                "unknown schema {:?} (this binary reads {:?})",
                self.schema,
                SCHEMA
            );
        }
        if self.session_id.trim().is_empty() {
            bail!("record {:?} has empty session_id", self.turn_id);
        }
        if self.turn_id.trim().is_empty() {
            bail!("record in session {:?} has empty turn_id", self.session_id);
        }
        if self.text.trim().is_empty() {
            bail!(
                "record {:?} has empty or whitespace-only text; adapters must not emit these",
                self.turn_id
            );
        }
        Ok(())
    }
}

/// Parse pre-normalized JSONL. Blank lines are skipped; a malformed line is an error
/// that names its line number rather than being silently dropped.
pub fn parse_jsonl(input: &str) -> Result<Vec<Record>> {
    let mut out = Vec::new();
    for (i, line) in input.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let rec: Record = serde_json::from_str(line)
            .with_context(|| format!("line {}: not a valid transcript record", i + 1))?;
        rec.validate().with_context(|| format!("line {}", i + 1))?;
        out.push(rec);
    }
    Ok(out)
}

/// The flattening rule.
///
/// Walk from the newest leaf to the root via `parent_id`; that path IS the conversation,
/// and every unreferenced branch is discarded. Score what the person actually saw — a
/// regeneration they scrolled past and abandoned never treated them any way at all.
///
/// Records with no `parent_id` anywhere are already linear and are returned in timestamp
/// order. Returns `(kept, discarded_count)`.
pub fn flatten(records: Vec<Record>) -> (Vec<Record>, usize) {
    if records.is_empty() {
        return (records, 0);
    }

    // "Already linear": no record in the set claims a parent.
    if records.iter().all(|r| r.parent_id.is_none()) {
        let mut linear = records;
        linear.sort_by_key(|r| (r.timestamp, r.turn_id.clone()));
        return (linear, 0);
    }

    let total = records.len();
    let by_id: HashMap<&str, &Record> = records.iter().map(|r| (r.turn_id.as_str(), r)).collect();

    // A leaf is a record nobody names as parent.
    let referenced: HashSet<&str> = records
        .iter()
        .filter_map(|r| r.parent_id.as_deref())
        .collect();

    // Newest leaf wins; turn_id breaks ties so the choice is deterministic across runs.
    let newest_leaf = records
        .iter()
        .filter(|r| !referenced.contains(r.turn_id.as_str()))
        .max_by_key(|r| (r.timestamp, r.turn_id.clone()));

    let Some(leaf) = newest_leaf else {
        // Every record is referenced => the parent links form a cycle. Refuse to guess;
        // fall back to timestamp order rather than looping forever.
        let mut linear = records;
        linear.sort_by_key(|r| (r.timestamp, r.turn_id.clone()));
        return (linear, 0);
    };

    let mut path: Vec<Record> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let mut cursor: Option<&Record> = Some(leaf);

    while let Some(rec) = cursor {
        if !seen.insert(rec.turn_id.clone()) {
            break; // cycle guard
        }
        path.push(rec.clone());
        cursor = rec
            .parent_id
            .as_deref()
            .and_then(|pid| by_id.get(pid).copied());
    }

    path.reverse();
    let discarded = total - path.len();
    (path, discarded)
}

/// A bounded unit of conversation: one thread, split at idle gaps.
#[derive(Debug, Clone)]
pub struct Session {
    pub session_id: String,
    pub source: String,
    pub records: Vec<Record>,
}

impl Session {
    pub fn started_at(&self) -> Option<DateTime<Utc>> {
        self.records.first().map(|r| r.timestamp)
    }
}

/// Group records into threads, flatten each, then split at idle gaps over `gap_hours`.
///
/// Thread boundaries alone don't work: a chat thread can run for months, overflowing the
/// rollup judge's context and blurring the trend line. Sub-session ids are suffixed
/// `#1`, `#2`, ... deterministically so the cache still hits across runs.
pub fn sessionize(records: Vec<Record>, gap_hours: i64) -> Vec<Session> {
    let mut threads: HashMap<String, Vec<Record>> = HashMap::new();
    for rec in records {
        threads.entry(rec.session_id.clone()).or_default().push(rec);
    }

    let gap = Duration::hours(gap_hours);
    let mut sessions = Vec::new();

    // Deterministic order so report output doesn't churn between runs.
    let mut threads: Vec<(String, Vec<Record>)> = threads.into_iter().collect();
    threads.sort_by(|a, b| a.0.cmp(&b.0));

    for (id, thread) in threads {
        let (linear, _discarded) = flatten(thread);
        if linear.is_empty() {
            continue;
        }

        let source = linear[0].source.clone();
        let mut chunks: Vec<Vec<Record>> = vec![Vec::new()];
        let mut prev: Option<DateTime<Utc>> = None;

        for rec in linear {
            if let Some(p) = prev {
                if rec.timestamp - p > gap {
                    chunks.push(Vec::new());
                }
            }
            prev = Some(rec.timestamp);
            chunks.last_mut().unwrap().push(rec);
        }

        let multi = chunks.len() > 1;
        for (i, chunk) in chunks.into_iter().enumerate() {
            if chunk.is_empty() {
                continue;
            }
            let sid = if multi {
                format!("{}#{}", id, i + 1)
            } else {
                id.clone()
            };
            sessions.push(Session {
                session_id: sid,
                source: source.clone(),
                records: chunk,
            });
        }
    }

    sessions.sort_by_key(|s| (s.started_at(), s.session_id.clone()));
    sessions
}

/// A unit of work for the turn-tier judge: one assistant turn plus the user prompt it
/// was answering.
#[derive(Debug, Clone)]
pub struct ScorableTurn {
    pub session_id: String,
    pub turn_id: String,
    pub source: String,
    pub model: Option<String>,
    pub timestamp: DateTime<Utc>,
    pub user_prompt: String,
    pub assistant_text: String,
    pub actions: Vec<Action>,
}

/// Select the assistant turns worth judging.
///
/// The `user_prompt` slot takes the most recent kept user turn walking *backward*, not
/// the immediately preceding record. Most assistant turns follow tool results, not a
/// person typing; requiring an immediately-preceding user record would silently discard
/// the majority of turns and every number in the report would be wrong without anything
/// appearing to fail. The same user turn legitimately serves several assistant turns.
///
/// Sidechain turns are excluded: a subagent talking to itself had no human on the other
/// end. An assistant turn with no kept user turn anywhere before it — a transcript that
/// begins mid-stream — is unscorable and skipped.
pub fn scorable_turns(session: &Session) -> Vec<ScorableTurn> {
    let mut out = Vec::new();
    let mut last_user: Option<&str> = None;

    for rec in &session.records {
        if rec.sidechain {
            continue;
        }
        match rec.role {
            Role::User => last_user = Some(rec.text.as_str()),
            Role::Assistant => {
                let Some(prompt) = last_user else {
                    continue; // begins mid-stream: unscorable
                };
                out.push(ScorableTurn {
                    session_id: session.session_id.clone(),
                    turn_id: rec.turn_id.clone(),
                    source: rec.source.clone(),
                    model: rec.model.clone(),
                    timestamp: rec.timestamp,
                    user_prompt: prompt.to_string(),
                    assistant_text: rec.text.clone(),
                    actions: rec.actions.clone(),
                });
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(s: &str) -> DateTime<Utc> {
        DateTime::parse_from_rfc3339(s).unwrap().with_timezone(&Utc)
    }

    fn rec(id: &str, parent: Option<&str>, role: Role, t: &str) -> Record {
        let mut r = Record::new("test", "s1", id, role, format!("text {id}"), ts(t));
        r.parent_id = parent.map(String::from);
        r
    }

    #[test]
    fn rejects_empty_text() {
        let mut r = Record::new(
            "test",
            "s1",
            "t1",
            Role::User,
            "   ",
            ts("2026-01-01T00:00:00Z"),
        );
        assert!(r.validate().is_err());
        r.text = "hi".into();
        assert!(r.validate().is_ok());
    }

    #[test]
    fn rejects_foreign_schema() {
        let mut r = Record::new(
            "test",
            "s1",
            "t1",
            Role::User,
            "hi",
            ts("2026-01-01T00:00:00Z"),
        );
        r.schema = "something/v9".into();
        assert!(r.validate().is_err());
    }

    #[test]
    fn flatten_keeps_newest_leaf_path_and_discards_branches() {
        // a -> b -> c   (kept, newest leaf c)
        //  \-> b2       (abandoned regeneration, discarded)
        let records = vec![
            rec("a", None, Role::User, "2026-01-01T00:00:00Z"),
            rec("b", Some("a"), Role::Assistant, "2026-01-01T00:01:00Z"),
            rec("b2", Some("a"), Role::Assistant, "2026-01-01T00:02:00Z"),
            rec("c", Some("b"), Role::User, "2026-01-01T00:03:00Z"),
        ];
        let (kept, discarded) = flatten(records);
        let ids: Vec<&str> = kept.iter().map(|r| r.turn_id.as_str()).collect();
        assert_eq!(ids, vec!["a", "b", "c"]);
        assert_eq!(discarded, 1);
    }

    #[test]
    fn flatten_treats_parentless_set_as_linear() {
        let records = vec![
            rec("b", None, Role::Assistant, "2026-01-01T00:01:00Z"),
            rec("a", None, Role::User, "2026-01-01T00:00:00Z"),
        ];
        let (kept, discarded) = flatten(records);
        let ids: Vec<&str> = kept.iter().map(|r| r.turn_id.as_str()).collect();
        assert_eq!(ids, vec!["a", "b"]);
        assert_eq!(discarded, 0);
    }

    #[test]
    fn flatten_survives_a_parent_cycle() {
        let mut a = rec("a", Some("b"), Role::User, "2026-01-01T00:00:00Z");
        a.parent_id = Some("b".into());
        let b = rec("b", Some("a"), Role::Assistant, "2026-01-01T00:01:00Z");
        let (kept, _) = flatten(vec![a, b]);
        assert_eq!(kept.len(), 2); // fell back to timestamp order, did not hang
    }

    #[test]
    fn sessionize_splits_on_idle_gap() {
        let records = vec![
            rec("a", None, Role::User, "2026-01-01T00:00:00Z"),
            rec("b", None, Role::Assistant, "2026-01-01T00:05:00Z"),
            // 7h later -> new session
            rec("c", None, Role::User, "2026-01-01T07:05:00Z"),
            rec("d", None, Role::Assistant, "2026-01-01T07:06:00Z"),
        ];
        let sessions = sessionize(records, DEFAULT_IDLE_GAP_HOURS);
        assert_eq!(sessions.len(), 2);
        assert_eq!(sessions[0].session_id, "s1#1");
        assert_eq!(sessions[1].session_id, "s1#2");
        assert_eq!(sessions[0].records.len(), 2);
    }

    #[test]
    fn sessionize_keeps_single_session_id_unsuffixed() {
        let records = vec![
            rec("a", None, Role::User, "2026-01-01T00:00:00Z"),
            rec("b", None, Role::Assistant, "2026-01-01T00:05:00Z"),
        ];
        let sessions = sessionize(records, DEFAULT_IDLE_GAP_HOURS);
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].session_id, "s1");
    }

    /// The load-bearing test: one user turn feeds several assistant turns.
    #[test]
    fn user_prompt_walks_backward_not_just_previous_record() {
        let records = vec![
            rec("u1", None, Role::User, "2026-01-01T00:00:00Z"),
            rec("a1", None, Role::Assistant, "2026-01-01T00:01:00Z"),
            rec("a2", None, Role::Assistant, "2026-01-01T00:02:00Z"),
            rec("a3", None, Role::Assistant, "2026-01-01T00:03:00Z"),
        ];
        let sessions = sessionize(records, DEFAULT_IDLE_GAP_HOURS);
        let turns = scorable_turns(&sessions[0]);
        assert_eq!(turns.len(), 3, "all three assistant turns must be scorable");
        assert!(turns.iter().all(|t| t.user_prompt == "text u1"));
    }

    #[test]
    fn assistant_turn_before_any_user_turn_is_skipped() {
        let records = vec![
            rec("a0", None, Role::Assistant, "2026-01-01T00:00:00Z"),
            rec("u1", None, Role::User, "2026-01-01T00:01:00Z"),
            rec("a1", None, Role::Assistant, "2026-01-01T00:02:00Z"),
        ];
        let sessions = sessionize(records, DEFAULT_IDLE_GAP_HOURS);
        let turns = scorable_turns(&sessions[0]);
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].turn_id, "a1");
    }

    #[test]
    fn sidechain_turns_are_excluded() {
        let mut side = rec("a2", None, Role::Assistant, "2026-01-01T00:02:00Z");
        side.sidechain = true;
        let records = vec![
            rec("u1", None, Role::User, "2026-01-01T00:00:00Z"),
            rec("a1", None, Role::Assistant, "2026-01-01T00:01:00Z"),
            side,
        ];
        let sessions = sessionize(records, DEFAULT_IDLE_GAP_HOURS);
        let turns = scorable_turns(&sessions[0]);
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].turn_id, "a1");
    }
}
