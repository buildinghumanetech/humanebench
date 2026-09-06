//! ChatGPT data export: `conversations.json` with a message mapping, tree-shaped.
//!
//! ⚠ UNVERIFIED. The spec marks every field name for this adapter as *illustrative* —
//! it was never confirmed against a real export archive. The structure below follows the
//! commonly-documented export shape, but until someone runs it against a real archive
//! treat a successful parse as a hypothesis, not a fact. `parse` reports how many
//! messages it recovered so a silent zero is visible rather than looking like an empty
//! history.

use crate::transcript::{Record, Role, SCHEMA};
use anyhow::Result;
use chrono::{DateTime, Utc};
use serde_json::Value;

pub const SOURCE: &str = "chatgpt";

pub fn detect(sample: &str) -> bool {
    let Ok(v) = serde_json::from_str::<Value>(sample) else {
        return false;
    };
    let probe = |c: &Value| c.get("mapping").is_some() && c.get("title").is_some();
    match &v {
        Value::Array(items) => items.first().map(probe).unwrap_or(false),
        other => probe(other),
    }
}

fn epoch_to_utc(v: Option<&Value>) -> Option<DateTime<Utc>> {
    let secs = v?.as_f64()?;
    DateTime::from_timestamp(secs as i64, ((secs.fract()) * 1e9) as u32)
}

fn parts_text(content: &Value) -> String {
    let Some(parts) = content.get("parts").and_then(|p| p.as_array()) else {
        return String::new();
    };
    parts
        .iter()
        .filter_map(|p| p.as_str())
        .collect::<Vec<_>>()
        .join("\n")
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
            .get("conversation_id")
            .or_else(|| conv.get("id"))
            .and_then(|i| i.as_str())
            .unwrap_or("unknown")
            .to_string();

        let Some(mapping) = conv.get("mapping").and_then(|m| m.as_object()) else {
            continue;
        };

        for (node_id, node) in mapping {
            let Some(msg) = node.get("message") else {
                continue;
            };
            if msg.is_null() {
                continue;
            }

            let role = match msg
                .get("author")
                .and_then(|a| a.get("role"))
                .and_then(|r| r.as_str())
            {
                Some("user") => Role::User,
                Some("assistant") => Role::Assistant,
                _ => continue, // system / tool authors are not conversation
            };

            let content = msg.get("content").cloned().unwrap_or(Value::Null);
            let text = parts_text(&content);
            if text.trim().is_empty() {
                continue;
            }

            let Some(timestamp) = epoch_to_utc(msg.get("create_time"))
                .or_else(|| epoch_to_utc(conv.get("create_time")))
            else {
                continue;
            };

            out.push(Record {
                schema: SCHEMA.to_string(),
                source: SOURCE.to_string(),
                session_id: session_id.clone(),
                turn_id: format!("{session_id}:{node_id}"),
                role,
                text: text.trim().to_string(),
                timestamp,
                // Tree-shaped: emit parent_id and let the engine flatten. ChatGPT branches
                // on edits and regenerations; the newest-leaf path is what the person saw.
                parent_id: node
                    .get("parent")
                    .and_then(|p| p.as_str())
                    .map(|p| format!("{session_id}:{p}")),
                model: msg
                    .get("metadata")
                    .and_then(|m| m.get("model_slug"))
                    .and_then(|m| m.as_str())
                    .map(str::to_string),
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
      "conversation_id":"c1","title":"t","create_time":1730000000.0,
      "mapping":{
        "root":{"id":"root","message":null,"parent":null},
        "n1":{"id":"n1","parent":"root","message":{"author":{"role":"user"},"create_time":1730000001.0,"content":{"content_type":"text","parts":["hello"]}}},
        "n2":{"id":"n2","parent":"n1","message":{"author":{"role":"assistant"},"create_time":1730000002.0,"metadata":{"model_slug":"gpt-4o"},"content":{"content_type":"text","parts":["hi"]}}},
        "n3":{"id":"n3","parent":"n1","message":{"author":{"role":"system"},"create_time":1730000003.0,"content":{"content_type":"text","parts":["sys"]}}}
      }}]"#;

    #[test]
    fn detects_chatgpt_export() {
        assert!(detect(SHAPE));
    }

    #[test]
    fn drops_null_and_system_messages() {
        let recs = parse(SHAPE).unwrap();
        assert_eq!(recs.len(), 2);
        assert!(recs.iter().all(|r| r.text == "hello" || r.text == "hi"));
    }

    #[test]
    fn namespaces_ids_by_conversation_and_links_parents() {
        let recs = parse(SHAPE).unwrap();
        let n2 = recs.iter().find(|r| r.text == "hi").unwrap();
        assert_eq!(n2.turn_id, "c1:n2");
        assert_eq!(n2.parent_id.as_deref(), Some("c1:n1"));
        assert_eq!(n2.model.as_deref(), Some("gpt-4o"));
    }
}
