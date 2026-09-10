//! MCP over stdio: the scored corpus as query tools.
//!
//! Read-only. The server cannot trigger scoring, because that would hand unbounded API
//! spend to an agent acting on its own judgement. There is no tool here that writes, and
//! `Store` is taken by shared reference so one cannot be added by accident.
//!
//! Excerpts are opt-in per call (`include_text`), never default: an agent may relay
//! results somewhere the person did not intend.

use crate::judge::{Tier, PRINCIPLES};
use crate::report::{self, principle_label};
use crate::store::{Filter, ScoredTurn, Store};
use anyhow::Result;
use chrono::{DateTime, Utc};
use serde_json::{json, Value};
use std::io::{BufRead, Write};

pub const PROTOCOL_VERSION: &str = "2024-11-05";
const MAX_LIMIT: usize = 200;

/// Tool definitions advertised to the client.
pub fn tool_definitions() -> Value {
    json!([
        {
            "name": "query_scores",
            "description": "Aggregate HumaneBench scores by principle, source, model, and time window. Returns means per principle for the turn tier and the session-rollup tier separately — they are never averaged together.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "since": {"type": "string", "description": "RFC 3339 lower bound on turn time"},
                    "until": {"type": "string", "description": "RFC 3339 upper bound on turn time"},
                    "source": {"type": "string", "description": "Filter to one source tag, e.g. claude-code"},
                    "model": {"type": "string", "description": "Filter to one assistant model"}
                },
                "additionalProperties": false
            }
        },
        {
            "name": "worst_turns",
            "description": "Lowest-scoring turns with the judge's rationales. Verbatim conversation text is returned ONLY when include_text is true.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "How many turns to return (default 10, max 200)"},
                    "since": {"type": "string"},
                    "source": {"type": "string"},
                    "model": {"type": "string"},
                    "principle": {"type": "string", "description": "Rank by one principle instead of the overall mean"},
                    "include_text": {"type": "boolean", "description": "Include verbatim excerpts. Off by default."}
                },
                "additionalProperties": false
            }
        },
        {
            "name": "principle_trend",
            "description": "One principle's time series, bucketed by day or week depending on span.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "principle": {"type": "string", "description": "One of the eight principle codes"},
                    "since": {"type": "string"},
                    "source": {"type": "string"}
                },
                "required": ["principle"],
                "additionalProperties": false
            }
        },
        {
            "name": "session_detail",
            "description": "One session's turn scores plus its rollup judgement.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "include_text": {"type": "boolean", "description": "Include verbatim excerpts. Off by default."}
                },
                "required": ["session_id"],
                "additionalProperties": false
            }
        }
    ])
}

fn parse_time(v: Option<&Value>) -> Option<DateTime<Utc>> {
    let s = v?.as_str()?;
    DateTime::parse_from_rfc3339(s)
        .ok()
        .map(|t| t.with_timezone(&Utc))
}

fn filter_from(args: &Value) -> Filter {
    Filter {
        since: parse_time(args.get("since")),
        until: parse_time(args.get("until")),
        source: args
            .get("source")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        model: args
            .get("model")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        tier: None,
        session_id: args
            .get("session_id")
            .and_then(|v| v.as_str())
            .map(str::to_string),
    }
}

fn principle_means(set: &[&ScoredTurn]) -> Value {
    let mut out = serde_json::Map::new();
    for code in PRINCIPLES {
        let vals: Vec<f64> = set
            .iter()
            .filter_map(|s| s.record.principle(code).map(|p| p.score))
            .collect();
        if vals.is_empty() {
            continue;
        }
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        out.insert(
            code.to_string(),
            json!({"label": principle_label(code), "mean": (mean * 1000.0).round() / 1000.0, "n": vals.len()}),
        );
    }
    Value::Object(out)
}

fn tool_query_scores(store: &Store, args: &Value) -> Result<Value> {
    let scores = store.scores(&filter_from(args))?;
    let turns: Vec<&ScoredTurn> = scores
        .iter()
        .filter(|s| s.record.tier == Tier::Turn)
        .collect();
    let rollups: Vec<&ScoredTurn> = scores
        .iter()
        .filter(|s| s.record.tier == Tier::Rollup)
        .collect();

    let mean_overall = |set: &[&ScoredTurn]| -> Option<f64> {
        if set.is_empty() {
            return None;
        }
        Some(set.iter().map(|s| s.record.overall()).sum::<f64>() / set.len() as f64)
    };

    Ok(json!({
        "turn_tier": {
            "n": turns.len(),
            "overall": mean_overall(&turns),
            "principles": principle_means(&turns),
        },
        "rollup_tier": {
            "n": rollups.len(),
            "overall": mean_overall(&rollups),
            "principles": principle_means(&rollups),
        },
        "facets": {
            "sources": store.distinct("source")?,
            "models": store.distinct("model")?,
            "judge_models": store.distinct("judge_model")?,
            "regimes": store.distinct("regime")?,
        },
        "caveat": "Single-judge scores are noisier than the benchmark's validated ensemble. \
                   Turn and rollup tiers are reported separately and must not be averaged together."
    }))
}

fn tool_worst_turns(store: &Store, args: &Value) -> Result<Value> {
    let mut filter = filter_from(args);
    filter.tier = Some(Tier::Turn);
    let scores = store.scores(&filter)?;

    let limit = args
        .get("limit")
        .and_then(|v| v.as_u64())
        .unwrap_or(10)
        .min(MAX_LIMIT as u64) as usize;
    let include_text = args
        .get("include_text")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let principle = args.get("principle").and_then(|v| v.as_str());

    let mut ranked: Vec<(&ScoredTurn, f64)> = scores
        .iter()
        .filter_map(|s| match principle {
            Some(code) => s.record.principle(code).map(|p| (s, p.score)),
            None => Some((s, s.record.overall())),
        })
        .collect();
    ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked.truncate(limit);

    let mut out = Vec::new();
    for (s, rank_score) in ranked {
        let negatives: Vec<Value> = s
            .record
            .principles
            .iter()
            .filter(|p| p.score < 0.0)
            .map(|p| json!({"principle": p.name, "label": principle_label(&p.name), "score": p.score, "rationale": p.rationale}))
            .collect();

        let mut entry = json!({
            "turn_id": s.record.turn_id,
            "session_id": s.record.session_id,
            "timestamp": s.timestamp.to_rfc3339(),
            "source": s.source,
            "model": s.model,
            "overall": (s.record.overall() * 1000.0).round() / 1000.0,
            "rank_score": rank_score,
            "confidence": s.record.confidence,
            "negatives": negatives,
            "globalViolations": s.record.global_violations,
        });

        if include_text {
            if let Some(text) = store.turn_text(&s.record.turn_id)? {
                entry["excerpt"] = json!(text);
            }
        }
        out.push(entry);
    }

    Ok(json!({
        "turns": out,
        "include_text": include_text,
        "note": if include_text {
            "Verbatim conversation text included at the caller's explicit request."
        } else {
            "Excerpts withheld. Pass include_text: true to include verbatim conversation text."
        }
    }))
}

fn tool_principle_trend(store: &Store, args: &Value) -> Result<Value> {
    let code = args
        .get("principle")
        .and_then(|v| v.as_str())
        .unwrap_or_default();
    if !PRINCIPLES.contains(&code) {
        anyhow::bail!(
            "unknown principle {code:?}; expected one of: {}",
            PRINCIPLES.join(", ")
        );
    }
    let scores = store.scores(&filter_from(args))?;
    let points = report::trend(&scores, code);

    Ok(json!({
        "principle": code,
        "label": principle_label(code),
        "points": points.iter().map(|p| json!({
            "bucket": p.bucket,
            "at": p.at.to_rfc3339(),
            "mean": (p.value * 1000.0).round() / 1000.0,
            "n": p.n,
        })).collect::<Vec<_>>(),
    }))
}

fn tool_session_detail(store: &Store, args: &Value) -> Result<Value> {
    let session_id = args
        .get("session_id")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();
    let include_text = args
        .get("include_text")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let scores = store.scores(&Filter {
        session_id: Some(session_id.clone()),
        ..Default::default()
    })?;

    let mut turns = Vec::new();
    let mut rollup = Value::Null;

    for s in &scores {
        let principles: Vec<Value> = s
            .record
            .principles
            .iter()
            .map(|p| json!({"principle": p.name, "score": p.score, "rationale": p.rationale}))
            .collect();

        let mut entry = json!({
            "turn_id": s.record.turn_id,
            "timestamp": s.timestamp.to_rfc3339(),
            "overall": (s.record.overall() * 1000.0).round() / 1000.0,
            "confidence": s.record.confidence,
            "principles": principles,
            "globalViolations": s.record.global_violations,
        });

        if include_text {
            if let Some(text) = store.turn_text(&s.record.turn_id)? {
                entry["excerpt"] = json!(text);
            }
        }

        match s.record.tier {
            Tier::Rollup => rollup = entry,
            Tier::Turn => turns.push(entry),
        }
    }

    Ok(json!({
        "session_id": session_id,
        "turn_count": turns.len(),
        "turns": turns,
        "rollup": rollup,
        "include_text": include_text,
    }))
}

pub fn call_tool(store: &Store, name: &str, args: &Value) -> Result<Value> {
    match name {
        "query_scores" => tool_query_scores(store, args),
        "worst_turns" => tool_worst_turns(store, args),
        "principle_trend" => tool_principle_trend(store, args),
        "session_detail" => tool_session_detail(store, args),
        // Deliberately absent: anything that scores. See the module doc.
        "score" | "ingest" => anyhow::bail!(
            "this server is read-only; scoring is a deliberate, human-invoked command \
             (`humanebench score`) so that API spend is never delegated to an agent"
        ),
        other => anyhow::bail!("unknown tool {other:?}"),
    }
}

fn error_response(id: Value, code: i64, message: String) -> Value {
    json!({"jsonrpc":"2.0","id":id,"error":{"code":code,"message":message}})
}

/// Handle one JSON-RPC request. Returns `None` for notifications, which take no reply.
pub fn handle_request(store: &Store, req: &Value) -> Option<Value> {
    let method = req.get("method").and_then(|m| m.as_str()).unwrap_or("");
    let id = req.get("id").cloned();

    // A notification has no id and never gets a response.
    let id = id?;

    match method {
        "initialize" => Some(json!({
            "jsonrpc":"2.0","id":id,
            "result":{
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name":"humanebench","version": env!("CARGO_PKG_VERSION")},
                "instructions": "Read-only access to a locally-scored HumaneBench corpus. \
                                 Verbatim conversation text is withheld unless include_text is \
                                 passed. This server cannot trigger scoring."
            }
        })),
        "ping" => Some(json!({"jsonrpc":"2.0","id":id,"result":{}})),
        "tools/list" => Some(json!({
            "jsonrpc":"2.0","id":id,
            "result":{"tools": tool_definitions()}
        })),
        "tools/call" => {
            let params = req.get("params").cloned().unwrap_or(json!({}));
            let name = params.get("name").and_then(|n| n.as_str()).unwrap_or("");
            let args = params.get("arguments").cloned().unwrap_or(json!({}));

            match call_tool(store, name, &args) {
                Ok(value) => Some(json!({
                    "jsonrpc":"2.0","id":id,
                    "result":{
                        "content":[{"type":"text","text": serde_json::to_string_pretty(&value)
                            .unwrap_or_else(|_| value.to_string())}],
                        "isError": false
                    }
                })),
                Err(e) => Some(json!({
                    "jsonrpc":"2.0","id":id,
                    "result":{
                        "content":[{"type":"text","text": format!("error: {e}")}],
                        "isError": true
                    }
                })),
            }
        }
        other => Some(error_response(
            id,
            -32601,
            format!("method not found: {other}"),
        )),
    }
}

/// Serve newline-delimited JSON-RPC over stdio until EOF.
pub fn serve(store: &Store) -> Result<()> {
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();

    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let req: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                let resp = error_response(Value::Null, -32700, format!("parse error: {e}"));
                writeln!(stdout, "{resp}")?;
                stdout.flush()?;
                continue;
            }
        };
        if let Some(resp) = handle_request(store, &req) {
            writeln!(stdout, "{resp}")?;
            stdout.flush()?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::judge::{PrincipleScore, ScoreRecord};
    use crate::store::score_record;
    use crate::transcript::{Record, Role};
    use chrono::TimeZone;

    fn judgement_scores(scores: &[f64]) -> Vec<PrincipleScore> {
        PRINCIPLES
            .iter()
            .zip(scores.iter())
            .map(|(n, s)| PrincipleScore {
                name: n.to_string(),
                score: *s,
                rationale: if *s < 0.0 {
                    Some(format!("bad {n}"))
                } else {
                    None
                },
            })
            .collect()
    }

    fn seeded() -> Store {
        let mut store = Store::open_in_memory().unwrap();
        let t = Utc.with_ymd_and_hms(2026, 1, 1, 12, 0, 0).unwrap();

        let recs = vec![
            Record::new(
                "claude-code",
                "s1",
                "t1",
                Role::Assistant,
                "VERBATIM-ONE",
                t,
            ),
            Record::new(
                "claude-code",
                "s1",
                "t2",
                Role::Assistant,
                "VERBATIM-TWO",
                t,
            ),
        ];
        store.upsert_records(&recs).unwrap();

        let mut a = score_record(
            "t1",
            "s1",
            Tier::Turn,
            "h1",
            "openrouter/m",
            "single",
            crate::judge::Judgement {
                principles: judgement_scores(&[1.0; 8]),
                global_violations: vec![],
                confidence: 0.9,
            },
        );
        a.principles = judgement_scores(&[1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);

        let b = ScoreRecord {
            principles: judgement_scores(&[-1.0, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
            ..score_record(
                "t2",
                "s1",
                Tier::Turn,
                "h2",
                "openrouter/m",
                "single",
                crate::judge::Judgement {
                    principles: judgement_scores(&[0.5; 8]),
                    global_violations: vec![],
                    confidence: 0.7,
                },
            )
        };

        let r = ScoreRecord {
            principles: judgement_scores(&[0.5, 0.5, 0.5, 0.5, -1.0, 0.5, 0.5, 0.5]),
            ..score_record(
                "s1:rollup",
                "s1",
                Tier::Rollup,
                "h3",
                "openrouter/m",
                "single",
                crate::judge::Judgement {
                    principles: judgement_scores(&[0.5; 8]),
                    global_violations: vec![],
                    confidence: 0.6,
                },
            )
        };

        store
            .insert_score_at(&a, "claude-code", Some("opus"), t)
            .unwrap();
        store
            .insert_score_at(&b, "claude-code", Some("opus"), t)
            .unwrap();
        store
            .insert_score_at(&r, "claude-code", Some("opus"), t)
            .unwrap();
        store
    }

    fn call(store: &Store, name: &str, args: Value) -> Value {
        call_tool(store, name, &args).unwrap()
    }

    #[test]
    fn initialize_advertises_tools_capability() {
        let store = seeded();
        let resp = handle_request(
            &store,
            &json!({"jsonrpc":"2.0","id":1,"method":"initialize"}),
        )
        .unwrap();
        assert_eq!(resp["result"]["protocolVersion"], PROTOCOL_VERSION);
        assert!(resp["result"]["capabilities"]["tools"].is_object());
    }

    #[test]
    fn notifications_get_no_reply() {
        let store = seeded();
        assert!(handle_request(
            &store,
            &json!({"jsonrpc":"2.0","method":"notifications/initialized"})
        )
        .is_none());
    }

    #[test]
    fn tools_list_exposes_exactly_the_four_query_tools() {
        let store = seeded();
        let resp = handle_request(
            &store,
            &json!({"jsonrpc":"2.0","id":2,"method":"tools/list"}),
        )
        .unwrap();
        let names: Vec<String> = resp["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(
            names,
            vec![
                "query_scores",
                "worst_turns",
                "principle_trend",
                "session_detail"
            ]
        );
    }

    #[test]
    fn unknown_method_is_a_jsonrpc_error() {
        let store = seeded();
        let resp =
            handle_request(&store, &json!({"jsonrpc":"2.0","id":3,"method":"nope"})).unwrap();
        assert_eq!(resp["error"]["code"], -32601);
    }

    #[test]
    fn query_scores_keeps_tiers_apart() {
        let store = seeded();
        let v = call(&store, "query_scores", json!({}));
        assert_eq!(v["turn_tier"]["n"], 2);
        assert_eq!(v["rollup_tier"]["n"], 1);
        // rollup healthy_relationships is -1.0 and must not be diluted by turn scores
        assert_eq!(
            v["rollup_tier"]["principles"]["healthy_relationships"]["mean"],
            -1.0
        );
        assert_eq!(
            v["turn_tier"]["principles"]["healthy_relationships"]["mean"],
            0.5
        );
    }

    #[test]
    fn worst_turns_withholds_excerpts_by_default() {
        let store = seeded();
        let v = call(&store, "worst_turns", json!({}));
        let s = v.to_string();
        assert!(
            !s.contains("VERBATIM"),
            "excerpt leaked without include_text"
        );
        assert!(s.contains("Pass include_text"));
    }

    #[test]
    fn worst_turns_includes_excerpts_only_on_explicit_request() {
        let store = seeded();
        let v = call(&store, "worst_turns", json!({"include_text": true}));
        assert!(v.to_string().contains("VERBATIM-TWO"));
    }

    #[test]
    fn worst_turns_ranks_lowest_first_and_excludes_rollups() {
        let store = seeded();
        let v = call(&store, "worst_turns", json!({}));
        let turns = v["turns"].as_array().unwrap();
        assert_eq!(turns[0]["turn_id"], "t2");
        assert!(turns.iter().all(|t| t["turn_id"] != "s1:rollup"));
    }

    #[test]
    fn worst_turns_can_rank_by_one_principle() {
        let store = seeded();
        let v = call(
            &store,
            "worst_turns",
            json!({"principle":"respect_attention","limit":1}),
        );
        assert_eq!(v["turns"][0]["turn_id"], "t2");
        assert_eq!(v["turns"][0]["rank_score"], -1.0);
    }

    #[test]
    fn worst_turns_limit_is_capped() {
        let store = seeded();
        let v = call(&store, "worst_turns", json!({"limit": 99999}));
        assert!(v["turns"].as_array().unwrap().len() <= MAX_LIMIT);
    }

    #[test]
    fn principle_trend_rejects_an_unknown_principle() {
        let store = seeded();
        assert!(call_tool(&store, "principle_trend", &json!({"principle":"vibes"})).is_err());
    }

    #[test]
    fn principle_trend_returns_a_series() {
        let store = seeded();
        let v = call(
            &store,
            "principle_trend",
            json!({"principle":"respect_attention"}),
        );
        assert_eq!(v["principle"], "respect_attention");
        assert!(!v["points"].as_array().unwrap().is_empty());
    }

    #[test]
    fn session_detail_separates_turns_from_rollup() {
        let store = seeded();
        let v = call(&store, "session_detail", json!({"session_id":"s1"}));
        assert_eq!(v["turn_count"], 2);
        assert_eq!(v["rollup"]["turn_id"], "s1:rollup");
        assert!(!v.to_string().contains("VERBATIM"));
    }

    /// The load-bearing safety property: no tool can spend money.
    #[test]
    fn server_refuses_to_trigger_scoring() {
        let store = seeded();
        let err = call_tool(&store, "score", &json!({}))
            .unwrap_err()
            .to_string();
        assert!(err.contains("read-only"));
        assert!(call_tool(&store, "ingest", &json!({})).is_err());

        let names: Vec<String> = tool_definitions()
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap().to_string())
            .collect();
        assert!(!names
            .iter()
            .any(|n| n.contains("score") && n != "query_scores"));
    }

    #[test]
    fn tool_errors_come_back_as_is_error_not_a_transport_failure() {
        let store = seeded();
        let resp = handle_request(
            &store,
            &json!({
                "jsonrpc":"2.0","id":9,"method":"tools/call",
                "params":{"name":"principle_trend","arguments":{"principle":"vibes"}}
            }),
        )
        .unwrap();
        assert_eq!(resp["result"]["isError"], true);
    }
}
