//! Codex rollout logs: a flat list, no per-message id.
//!
//! Field names here were read off real rollout files. Because Codex carries no message
//! id, `turn_id` is synthesized as session plus the ordinal of the record among those
//! kept — *not* session-plus-timestamp. Timestamps are millisecond-precision and two kept
//! records can share one, which would collide two turns onto a single id and let one
//! silently overwrite the other's score. The ordinal is equally deterministic across
//! re-runs, which is the property the cache actually needs.

use crate::transcript::{Record, Role, SCHEMA};
use anyhow::Result;
use chrono::{DateTime, Utc};
use serde_json::Value;

pub const SOURCE: &str = "codex";

pub fn detect(sample: &str) -> bool {
    for line in sample.lines().take(50) {
        let Ok(v) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        if v.get("type").and_then(|t| t.as_str()) == Some("session_meta") {
            return true;
        }
        if v.get("type").and_then(|t| t.as_str()) == Some("response_item")
            && v.get("payload").is_some()
        {
            return true;
        }
    }
    false
}

pub fn parse(input: &str) -> Result<Vec<Record>> {
    let mut session: Option<String> = None;
    let mut ordinal: u32 = 0;
    let mut out = Vec::new();

    for line in input.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let Ok(v) = serde_json::from_str::<Value>(line) else {
            continue;
        };

        if v.get("type").and_then(|t| t.as_str()) == Some("session_meta") {
            session = v
                .get("payload")
                .and_then(|p| p.get("id"))
                .and_then(|i| i.as_str())
                .map(str::to_string);
            continue;
        }

        let Some(p) = v.get("payload") else { continue };
        if p.get("type").and_then(|t| t.as_str()) != Some("message") {
            continue;
        }

        let role = match p.get("role").and_then(|r| r.as_str()) {
            Some("user") => Role::User,
            Some("assistant") => Role::Assistant,
            _ => continue,
        };

        let text: String = p
            .get("content")
            .and_then(|c| c.as_array())
            .map(|blocks| {
                blocks
                    .iter()
                    .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                    .collect::<String>()
            })
            .unwrap_or_default();

        let trimmed = text.trim();
        if trimmed.is_empty() {
            continue;
        }
        // An <environment_context> block is injected by the harness, not typed by a person.
        let head: String = trimmed.chars().take(40).collect();
        if head.contains("environment_context") {
            continue;
        }

        let Some(ts) = v.get("timestamp").and_then(|t| t.as_str()) else {
            continue;
        };
        let Ok(timestamp) = DateTime::parse_from_rfc3339(ts).map(|t| t.with_timezone(&Utc)) else {
            continue;
        };

        let sid = session.clone().unwrap_or_else(|| "unknown".to_string());
        ordinal += 1;

        out.push(Record {
            schema: SCHEMA.to_string(),
            source: SOURCE.to_string(),
            session_id: sid.clone(),
            turn_id: format!("{sid}:{ordinal:04}"),
            role,
            text: trimmed.to_string(),
            timestamp,
            parent_id: None, // flat source: already linear
            model: None,
            actions: Vec::new(),
            sidechain: false,
        });
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const REAL_SHAPE: &str = r#"
{"type":"session_meta","payload":{"id":"0199b07d-60a4-7f93-bfd6-6cafc456607e"}}
{"type":"response_item","timestamp":"2025-10-04T18:30:30.860Z","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"why is the upload flaky?"}]}}
{"type":"response_item","timestamp":"2025-10-04T18:30:31.000Z","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Because the retry budget resets."}]}}
{"type":"response_item","timestamp":"2025-10-04T18:30:32.000Z","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"<environment_context>cwd=/tmp</environment_context>"}]}}
"#;

    #[test]
    fn detects_codex() {
        assert!(detect(REAL_SHAPE));
    }

    #[test]
    fn synthesizes_ordinal_ids_and_harvests_session() {
        let recs = parse(REAL_SHAPE).unwrap();
        assert_eq!(recs.len(), 2);
        assert_eq!(recs[0].turn_id, "0199b07d-60a4-7f93-bfd6-6cafc456607e:0001");
        assert_eq!(recs[1].turn_id, "0199b07d-60a4-7f93-bfd6-6cafc456607e:0002");
        assert_eq!(recs[0].session_id, "0199b07d-60a4-7f93-bfd6-6cafc456607e");
        assert!(recs[0].parent_id.is_none());
    }

    #[test]
    fn drops_environment_context_blocks() {
        let recs = parse(REAL_SHAPE).unwrap();
        assert!(!recs.iter().any(|r| r.text.contains("environment_context")));
    }

    /// The reason ids are ordinals: two kept records can share a millisecond.
    #[test]
    fn same_millisecond_records_do_not_collide() {
        let input = r#"
{"type":"session_meta","payload":{"id":"s"}}
{"type":"response_item","timestamp":"2025-10-04T18:30:30.860Z","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"one"}]}}
{"type":"response_item","timestamp":"2025-10-04T18:30:30.860Z","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"two"}]}}
"#;
        let recs = parse(input).unwrap();
        assert_eq!(recs.len(), 2);
        assert_ne!(recs[0].turn_id, recs[1].turn_id);
    }

    #[test]
    fn ids_are_stable_across_reruns() {
        let a = parse(REAL_SHAPE).unwrap();
        let b = parse(REAL_SHAPE).unwrap();
        let ids_a: Vec<&str> = a.iter().map(|r| r.turn_id.as_str()).collect();
        let ids_b: Vec<&str> = b.iter().map(|r| r.turn_id.as_str()).collect();
        assert_eq!(ids_a, ids_b);
    }
}
