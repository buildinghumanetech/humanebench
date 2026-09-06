//! Claude app data export: `conversations.json`, flat `chat_messages` per conversation.
//!
//! ⚠ UNVERIFIED — same caveat as the ChatGPT adapter. The spec never inspected a real
//! Claude app export, so every field name here is illustrative and must be confirmed
//! against a real archive before this is trusted.

use crate::transcript::{Record, Role, SCHEMA};
use anyhow::Result;
use chrono::{DateTime, Utc};
use serde_json::Value;

pub const SOURCE: &str = "claude-app";

pub fn detect(sample: &str) -> bool {
    let Ok(v) = serde_json::from_str::<Value>(sample) else {
        return false;
    };
    let probe = |c: &Value| c.get("chat_messages").is_some();
    match &v {
        Value::Array(items) => items.first().map(probe).unwrap_or(false),
        other => probe(other),
    }
}

/// Claude app messages carry either a flat `text` or a `content` block array.
fn message_text(msg: &Value) -> String {
    if let Some(t) = msg.get("text").and_then(|t| t.as_str()) {
        if !t.trim().is_empty() {
            return t.to_string();
        }
    }
    msg.get("content")
        .and_then(|c| c.as_array())
        .map(|blocks| {
            blocks
                .iter()
                .filter(|b| b.get("type").and_then(|t| t.as_str()) == Some("text"))
                .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default()
}

pub fn parse(input: &str) -> Result<Vec<Record>> {
    let v: Value = serde_json::from_str(input)?;
    let conversations: Vec<&Value> = match &v {
        Value::Array(items) => items.iter().collect(),
        other => vec![other],
    };

    let mut out = Vec::new();

    for conv in conversations {
        let session_id = conv
            .get("uuid")
            .or_else(|| conv.get("id"))
            .and_then(|i| i.as_str())
            .unwrap_or("unknown")
            .to_string();

        let Some(messages) = conv.get("chat_messages").and_then(|m| m.as_array()) else {
            continue;
        };

        for (i, msg) in messages.iter().enumerate() {
            let role = match msg.get("sender").and_then(|s| s.as_str()) {
                Some("human") | Some("user") => Role::User,
                Some("assistant") => Role::Assistant,
                _ => continue,
            };

            let text = message_text(msg);
            if text.trim().is_empty() {
                continue;
            }

            let Some(timestamp) = msg
                .get("created_at")
                .and_then(|t| t.as_str())
                .and_then(|t| DateTime::parse_from_rfc3339(t).ok())
                .map(|t| t.with_timezone(&Utc))
            else {
                continue;
            };

            // Prefer the message uuid; fall back to an ordinal so ids stay stable and
            // never collide on a shared timestamp.
            let turn_id = msg
                .get("uuid")
                .and_then(|u| u.as_str())
                .map(|u| format!("{session_id}:{u}"))
                .unwrap_or_else(|| format!("{session_id}:{:04}", i + 1));

            out.push(Record {
                schema: SCHEMA.to_string(),
                source: SOURCE.to_string(),
                session_id: session_id.clone(),
                turn_id,
                role,
                text: text.trim().to_string(),
                timestamp,
                parent_id: None, // flat list
                model: None,
                actions: Vec::new(),
                sidechain: false,
            });
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHAPE: &str = r#"[{
      "uuid":"conv-1","name":"t",
      "chat_messages":[
        {"uuid":"m1","sender":"human","created_at":"2026-01-01T00:00:00Z","text":"hello"},
        {"uuid":"m2","sender":"assistant","created_at":"2026-01-01T00:00:05Z","text":"hi"},
        {"uuid":"m3","sender":"assistant","created_at":"2026-01-01T00:00:06Z","text":"   "}
      ]}]"#;

    #[test]
    fn detects_claude_app_export() {
        assert!(detect(SHAPE));
    }

    #[test]
    fn parses_flat_messages_and_drops_empty() {
        let recs = parse(SHAPE).unwrap();
        assert_eq!(recs.len(), 2);
        assert_eq!(recs[0].turn_id, "conv-1:m1");
        assert_eq!(recs[0].role, Role::User);
        assert!(recs[0].parent_id.is_none());
    }
}
