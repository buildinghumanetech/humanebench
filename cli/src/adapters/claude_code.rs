//! Claude Code session logs: a tree linked through `parentUuid`.
//!
//! Field names here were read off real session transcripts. Most record types in a
//! Claude Code log are not conversation at all — see `DROP_TYPES`.

use crate::transcript::{Action, Record, Role, SCHEMA};
use anyhow::Result;
use chrono::{DateTime, Utc};
use serde_json::Value;
use std::collections::{HashMap, HashSet};

pub const SOURCE: &str = "claude-code";

/// Record types that are harness bookkeeping, not conversation.
const DROP_TYPES: &[&str] = &[
    "attachment",
    "file-history-delta",
    "file-history-snapshot",
    "ai-title",
    "mode",
    "permission-mode",
    "queue-operation",
    "bridge-session",
    "last-prompt",
    "system",
];

/// Recognize a Claude Code log from its first parseable record.
pub fn detect(sample: &str) -> bool {
    for line in sample.lines().take(50) {
        let Ok(v) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        if v.get("parentUuid").is_some() || v.get("isSidechain").is_some() {
            return true;
        }
        if v.get("uuid").is_some() && v.get("sessionId").is_some() {
            return true;
        }
    }
    false
}

/// Pull a short, human-legible summary out of a tool_use `input` object.
fn summarize_tool(name: &str, input: &Value) -> String {
    let pick = |k: &str| input.get(k).and_then(|v| v.as_str()).map(str::to_string);

    let raw = match name {
        "Bash" => pick("command"),
        "Read" | "Edit" | "Write" | "NotebookEdit" => pick("file_path"),
        "Grep" | "Glob" => pick("pattern"),
        "Task" | "Agent" => pick("description"),
        "WebFetch" => pick("url"),
        "ToolSearch" => pick("query"),
        _ => None,
    }
    .or_else(|| {
        // Fall back to the first short string field, so unknown tools still say something.
        input.as_object().and_then(|o| {
            o.values()
                .find_map(|v| v.as_str().filter(|s| !s.is_empty()))
                .map(str::to_string)
        })
    })
    .unwrap_or_default();

    let one_line = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    truncate(&one_line, 120)
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max).collect();
    out.push('…');
    out
}

/// Concatenate the `text` blocks of a content value. `thinking` and `tool_use` blocks are
/// deliberately excluded: the former is not the response, the latter is context.
fn text_blocks(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => {
            let mut buf = String::new();
            for b in blocks {
                if b.get("type").and_then(|t| t.as_str()) == Some("text") {
                    if let Some(t) = b.get("text").and_then(|t| t.as_str()) {
                        if !buf.is_empty() {
                            buf.push('\n');
                        }
                        buf.push_str(t);
                    }
                }
            }
            buf
        }
        _ => String::new(),
    }
}

/// True when a user record is a tool_result envelope — the harness, not the person.
fn is_tool_result(content: &Value) -> bool {
    content
        .as_array()
        .map(|blocks| {
            blocks
                .iter()
                .any(|b| b.get("type").and_then(|t| t.as_str()) == Some("tool_result"))
        })
        .unwrap_or(false)
}

/// Re-link a kept record to its nearest kept ancestor.
///
/// Most records in a Claude Code log are dropped — tool_result envelopes, tool-only
/// assistant turns, harness bookkeeping — and every one of them is a link in the
/// `parentUuid` chain. Leaving `parent_id` pointing at a dropped record orphans the
/// chain, and the newest-leaf walk then terminates at the first gap, discarding almost
/// the entire conversation. Walking up to the nearest survivor keeps genuine branch
/// discarding intact while not inventing a break that was never in the data.
fn nearest_kept_ancestor(
    start: Option<&str>,
    parent_of: &HashMap<String, Option<String>>,
    kept: &HashSet<String>,
) -> Option<String> {
    let mut cursor = start.map(str::to_string);
    let mut guard = 0usize;
    while let Some(id) = cursor {
        // Cycle / runaway guard: logs are machine-written but not guaranteed sane.
        guard += 1;
        if guard > 10_000 {
            return None;
        }
        if kept.contains(&id) {
            return Some(id);
        }
        cursor = parent_of.get(&id).cloned().flatten();
    }
    None
}

pub fn parse(input: &str) -> Result<Vec<Record>> {
    let mut out: Vec<Record> = Vec::new();
    // tool_use blocks seen since the last text-bearing assistant turn.
    let mut pending: Vec<Action> = Vec::new();
    // Every uuid -> parentUuid link in the file, kept and dropped alike.
    let mut parent_of: HashMap<String, Option<String>> = HashMap::new();

    for line in input.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let Ok(v) = serde_json::from_str::<Value>(line) else {
            continue; // a corrupt line in a log is not a reason to abort the file
        };

        // Record the link before any filtering: dropped records are still chain links.
        if let Some(uuid) = v.get("uuid").and_then(|u| u.as_str()) {
            parent_of.insert(
                uuid.to_string(),
                v.get("parentUuid")
                    .and_then(|p| p.as_str())
                    .map(str::to_string),
            );
        }

        let rtype = v.get("type").and_then(|t| t.as_str()).unwrap_or("");
        if DROP_TYPES.contains(&rtype) {
            continue;
        }
        if rtype != "user" && rtype != "assistant" {
            continue;
        }

        let message = v.get("message").cloned().unwrap_or(Value::Null);
        let content = message.get("content").cloned().unwrap_or(Value::Null);

        // Collect tool_use blocks regardless of whether this record carries text.
        if rtype == "assistant" {
            if let Some(blocks) = content.as_array() {
                for b in blocks {
                    if b.get("type").and_then(|t| t.as_str()) == Some("tool_use") {
                        let name = b
                            .get("name")
                            .and_then(|n| n.as_str())
                            .unwrap_or("tool")
                            .to_string();
                        let summary = b
                            .get("input")
                            .map(|i| summarize_tool(&name, i))
                            .unwrap_or_default();
                        pending.push(Action { name, summary });
                    }
                }
            }
        }

        if rtype == "user" && is_tool_result(&content) {
            continue; // the harness replying to itself
        }

        let text = text_blocks(&content);
        let trimmed = text.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with("<local-command-caveat") {
            continue;
        }

        let Some(ts) = v.get("timestamp").and_then(|t| t.as_str()) else {
            continue;
        };
        let Ok(timestamp) = DateTime::parse_from_rfc3339(ts).map(|t| t.with_timezone(&Utc)) else {
            continue;
        };

        let Some(uuid) = v.get("uuid").and_then(|u| u.as_str()) else {
            continue;
        };
        let session_id = v
            .get("sessionId")
            .and_then(|s| s.as_str())
            .unwrap_or("unknown")
            .to_string();

        let role = if rtype == "user" {
            Role::User
        } else {
            Role::Assistant
        };

        let mut rec = Record {
            schema: SCHEMA.to_string(),
            source: SOURCE.to_string(),
            session_id,
            turn_id: uuid.to_string(),
            role,
            text: trimmed.to_string(),
            timestamp,
            // uuid/parentUuid map straight on, which is why Claude Code needs no id synthesis.
            parent_id: v
                .get("parentUuid")
                .and_then(|p| p.as_str())
                .map(str::to_string),
            model: message
                .get("model")
                .and_then(|m| m.as_str())
                .map(str::to_string),
            actions: Vec::new(),
            sidechain: v
                .get("isSidechain")
                .and_then(|s| s.as_bool())
                .unwrap_or(false),
        };

        if role == Role::Assistant {
            rec.actions = std::mem::take(&mut pending);
        }

        out.push(rec);
    }

    // Re-link across everything that was dropped, so the tree stays walkable.
    let kept: HashSet<String> = out.iter().map(|r| r.turn_id.clone()).collect();
    for rec in &mut out {
        rec.parent_id = nearest_kept_ancestor(rec.parent_id.as_deref(), &parent_of, &kept);
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const REAL_SHAPE: &str = r#"
{"type":"queue-operation","uuid":"q1","timestamp":"2026-07-30T16:40:00.000Z"}
{"type":"user","uuid":"u1","parentUuid":null,"sessionId":"df03642d","timestamp":"2026-07-30T16:41:00.000Z","isSidechain":false,"message":{"content":"why is the upload flaky?"}}
{"type":"assistant","uuid":"a1","parentUuid":"u1","sessionId":"df03642d","timestamp":"2026-07-30T16:41:30.000Z","isSidechain":false,"message":{"model":"claude-opus-5","content":[{"type":"tool_use","id":"t1","name":"Bash","input":{"command":"git status"}},{"type":"tool_use","id":"t2","name":"Read","input":{"file_path":"src/app.ts"}}]}}
{"type":"user","uuid":"u2","parentUuid":"a1","sessionId":"df03642d","timestamp":"2026-07-30T16:41:40.000Z","isSidechain":false,"message":{"content":[{"type":"tool_result","tool_use_id":"t1","content":"clean"}]}}
{"type":"assistant","uuid":"a2","parentUuid":"u2","sessionId":"df03642d","timestamp":"2026-07-30T16:42:07.833Z","isSidechain":false,"message":{"model":"claude-opus-5","content":[{"type":"text","text":"The xhigh review …"}]}}
{"type":"attachment","uuid":"x1","timestamp":"2026-07-30T16:43:00.000Z"}
{"type":"last-prompt","uuid":"x2","timestamp":"2026-07-30T16:43:01.000Z"}
"#;

    #[test]
    fn detects_claude_code() {
        assert!(detect(REAL_SHAPE));
    }

    #[test]
    fn keeps_only_conversation_records() {
        let recs = parse(REAL_SHAPE).unwrap();
        let ids: Vec<&str> = recs.iter().map(|r| r.turn_id.as_str()).collect();
        // q1/x1/x2 dropped by type; u2 dropped as a tool_result; a1 has no text.
        assert_eq!(ids, vec!["u1", "a2"]);
    }

    #[test]
    fn folds_tool_uses_into_the_next_text_turn() {
        let recs = parse(REAL_SHAPE).unwrap();
        let a2 = recs.iter().find(|r| r.turn_id == "a2").unwrap();
        assert_eq!(a2.actions.len(), 2);
        assert_eq!(a2.actions[0].name, "Bash");
        assert_eq!(a2.actions[0].summary, "git status");
        assert_eq!(a2.actions[1].name, "Read");
        assert_eq!(a2.actions[1].summary, "src/app.ts");
    }

    #[test]
    fn maps_ids_and_model_straight_through() {
        let recs = parse(REAL_SHAPE).unwrap();
        let a2 = recs.iter().find(|r| r.turn_id == "a2").unwrap();
        assert_eq!(a2.model.as_deref(), Some("claude-opus-5"));
        assert_eq!(a2.session_id, "df03642d");
        assert!(!a2.sidechain);
    }

    /// Regression, found against a real session: the chain a2 -> u2 -> a1 -> u1 passes
    /// through two dropped records. Leaving parent_id pointing at the dropped u2 orphans
    /// the tree, and the newest-leaf walk then keeps a single record and discards the
    /// whole conversation.
    #[test]
    fn relinks_across_dropped_records() {
        let recs = parse(REAL_SHAPE).unwrap();
        let a2 = recs.iter().find(|r| r.turn_id == "a2").unwrap();
        assert_eq!(
            a2.parent_id.as_deref(),
            Some("u1"),
            "must re-link to the nearest KEPT ancestor, not the dropped tool_result"
        );

        // And the whole conversation must survive flattening.
        let (kept, discarded) = crate::transcript::flatten(recs);
        assert_eq!(kept.len(), 2, "flatten dropped the conversation");
        assert_eq!(discarded, 0);
    }

    #[test]
    fn root_record_keeps_a_null_parent() {
        let recs = parse(REAL_SHAPE).unwrap();
        let u1 = recs.iter().find(|r| r.turn_id == "u1").unwrap();
        assert!(u1.parent_id.is_none());
    }

    #[test]
    fn drops_local_command_caveat_records() {
        let input = r#"{"type":"user","uuid":"u9","sessionId":"s","timestamp":"2026-07-30T16:41:00.000Z","message":{"content":"<local-command-caveat>ran /clear</local-command-caveat>"}}"#;
        assert!(parse(input).unwrap().is_empty());
    }

    #[test]
    fn skips_corrupt_lines_without_aborting() {
        let input = format!("{{not json\n{}", REAL_SHAPE);
        assert_eq!(parse(&input).unwrap().len(), 2);
    }
}
