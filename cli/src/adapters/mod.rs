//! Built-in adapters.
//!
//! Adapters are the only module anyone should need to touch to add a source — and the
//! normalized schema means they often won't need to touch even that. A converter written
//! in any language that emits `humanebench.transcript/v1` enters the engine at exactly
//! the same point as everything here.

pub mod chatgpt;
pub mod claude_app;
pub mod claude_code;
pub mod codex;
pub mod hermes;

use crate::transcript::{parse_jsonl, Record};
use anyhow::{anyhow, bail, Context, Result};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    ClaudeCode,
    Codex,
    ChatGpt,
    ClaudeApp,
    Hermes,
    /// Pre-normalized: already speaks the contract.
    Normalized,
}

impl Source {
    pub fn parse_name(s: &str) -> Result<Source> {
        Ok(match s.to_ascii_lowercase().replace('_', "-").as_str() {
            "claude-code" | "claudecode" => Source::ClaudeCode,
            "codex" => Source::Codex,
            "chatgpt" | "openai" => Source::ChatGpt,
            "claude-app" | "claude" => Source::ClaudeApp,
            "hermes" => Source::Hermes,
            "normalized" | "humanebench" | "jsonl" => Source::Normalized,
            other => bail!(
                "unknown source {other:?}; expected one of: claude-code, codex, chatgpt, \
                 claude-app, hermes, normalized"
            ),
        })
    }

    pub fn tag(&self) -> &'static str {
        match self {
            Source::ClaudeCode => claude_code::SOURCE,
            Source::Codex => codex::SOURCE,
            Source::ChatGpt => chatgpt::SOURCE,
            Source::ClaudeApp => claude_app::SOURCE,
            Source::Hermes => hermes::SOURCE,
            Source::Normalized => "normalized",
        }
    }

    /// Whether this adapter's field names were confirmed against real data. Unverified
    /// adapters warn on use rather than quietly producing plausible-looking nonsense.
    pub fn is_unverified(&self) -> bool {
        matches!(self, Source::ChatGpt | Source::ClaudeApp)
    }
}

/// True when the first parseable record actually declares our schema in its `schema`
/// field. A substring search is not good enough: any transcript that merely *mentions*
/// the schema tag — including a conversation about this tool — would match it, and the
/// file would then be parsed with the wrong adapter and rejected line by line.
fn declares_schema(sample: &str) -> bool {
    for line in sample.lines().take(50) {
        if line.trim().is_empty() {
            continue;
        }
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        return v.get("schema").and_then(|s| s.as_str()) == Some(crate::transcript::SCHEMA);
    }
    false
}

/// Sniff a source from file content. Order matters: the two adapters whose field names
/// were confirmed against real data get first refusal.
pub fn detect(sample: &str) -> Option<Source> {
    if declares_schema(sample) {
        return Some(Source::Normalized);
    }
    if claude_code::detect(sample) {
        return Some(Source::ClaudeCode);
    }
    if codex::detect(sample) {
        return Some(Source::Codex);
    }
    if chatgpt::detect(sample) {
        return Some(Source::ChatGpt);
    }
    if claude_app::detect(sample) {
        return Some(Source::ClaudeApp);
    }
    None
}

pub fn parse_with(source: Source, input: &str) -> Result<Vec<Record>> {
    match source {
        Source::ClaudeCode => claude_code::parse(input),
        Source::Codex => codex::parse(input),
        Source::ChatGpt => chatgpt::parse(input),
        Source::ClaudeApp => claude_app::parse(input),
        Source::Hermes => hermes::parse(input),
        Source::Normalized => parse_jsonl(input),
    }
}

/// Parse content whose source may be unknown, returning the source actually used.
pub fn parse_auto(input: &str, forced: Option<Source>) -> Result<(Source, Vec<Record>)> {
    let source = match forced {
        Some(s) => s,
        None => detect(input).ok_or_else(|| {
            anyhow!(
                "could not detect the transcript format.\n\n\
                 Pass --source to force it, or convert to the documented schema and pipe \
                 it in with --stdin (see `humanebench schema`)."
            )
        })?,
    };
    let records = parse_with(source, input)?;
    Ok((source, records))
}

/// Candidate transcript roots under `home`, in a stable order.
///
/// Only the adapters whose field names were confirmed against real data get an entry.
/// Consumer chat history has no on-disk location to find — it arrives as an export
/// archive the user passes explicitly — so guessing a root for it would only produce
/// misleading "not present" lines.
fn roots_under(home: &Path) -> Vec<(PathBuf, &'static str)> {
    vec![
        (
            home.join(".claude").join("projects"),
            Source::ClaudeCode.tag(),
        ),
        (home.join(".codex").join("sessions"), Source::Codex.tag()),
    ]
}

/// Where to look when `ingest` is given no path.
pub fn default_roots() -> Vec<(PathBuf, &'static str)> {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    roots_under(Path::new(&home))
}

/// Files worth trying inside a directory or export archive.
fn is_candidate(path: &Path) -> bool {
    match path.extension().and_then(|e| e.to_str()) {
        Some("jsonl") => true,
        Some("json") => path
            .file_name()
            .and_then(|n| n.to_str())
            .map(|n| n == "conversations.json")
            .unwrap_or(false),
        _ => false,
    }
}

fn collect_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_files(&path, out)?;
        } else if is_candidate(&path) {
            out.push(path);
        }
    }
    Ok(())
}

/// Read `conversations.json` out of an export zip. Consumer chat history is only
/// reachable through a manual export archive — there is no live tail and no API.
fn read_zip(path: &Path) -> Result<Vec<(String, String)>> {
    let file = fs::File::open(path)?;
    let mut archive = zip::ZipArchive::new(file)
        .with_context(|| format!("{} is not a readable zip archive", path.display()))?;

    let mut out = Vec::new();
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        if !entry.is_file() {
            continue;
        }
        let name = entry.name().to_string();
        if !name.ends_with("conversations.json") && !name.ends_with(".jsonl") {
            continue;
        }
        let mut buf = String::new();
        if entry.read_to_string(&mut buf).is_ok() && !buf.trim().is_empty() {
            out.push((name, buf));
        }
    }

    if out.is_empty() {
        bail!(
            "no conversations.json or .jsonl found inside {}",
            path.display()
        );
    }
    Ok(out)
}

/// Ingest a path: a file, an export zip, or a directory to walk.
/// Returns the records plus a per-file note of what was read.
pub fn ingest_path(path: &Path, forced: Option<Source>) -> Result<(Vec<Record>, Vec<String>)> {
    let mut records = Vec::new();
    let mut notes = Vec::new();

    if !path.exists() {
        bail!("{} does not exist", path.display());
    }

    let mut blobs: Vec<(String, String)> = Vec::new();

    if path.is_dir() {
        let mut files = Vec::new();
        collect_files(path, &mut files)?;
        files.sort();
        if files.is_empty() {
            bail!(
                "found no .jsonl or conversations.json files under {}",
                path.display()
            );
        }
        for f in files {
            let content = fs::read_to_string(&f).unwrap_or_default();
            if !content.trim().is_empty() {
                blobs.push((f.display().to_string(), content));
            }
        }
    } else if path.extension().and_then(|e| e.to_str()) == Some("zip") {
        blobs = read_zip(path)?;
    } else {
        let content =
            fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        blobs.push((path.display().to_string(), content));
    }

    for (name, content) in blobs {
        match parse_auto(&content, forced) {
            Ok((source, mut recs)) => {
                notes.push(format!(
                    "{}: {} records via {}",
                    name,
                    recs.len(),
                    source.tag()
                ));
                records.append(&mut recs);
            }
            Err(e) => {
                notes.push(format!("{name}: skipped ({e})"));
            }
        }
    }

    if records.is_empty() {
        bail!(
            "no usable conversation records found.\n\n{}",
            notes.join("\n")
        );
    }

    Ok((records, notes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalized_wins_detection() {
        let line = r#"{"schema":"humanebench.transcript/v1","source":"x","session_id":"s","turn_id":"t","role":"user","text":"hi","timestamp":"2026-01-01T00:00:00Z"}"#;
        assert_eq!(detect(line), Some(Source::Normalized));
    }

    /// Regression: a real Claude Code session that merely *discusses* the schema was
    /// misdetected as pre-normalized, and then failed to parse line by line.
    #[test]
    fn a_transcript_that_merely_mentions_the_schema_is_not_normalized() {
        let line = r#"{"type":"assistant","uuid":"a","parentUuid":null,"sessionId":"s","timestamp":"2026-01-01T00:00:00Z","message":{"content":[{"type":"text","text":"the tag is humanebench.transcript/v1"}]}}"#;
        assert_eq!(detect(line), Some(Source::ClaudeCode));
    }

    #[test]
    fn detects_each_known_source() {
        assert_eq!(
            detect(r#"{"type":"assistant","uuid":"a","parentUuid":null,"sessionId":"s"}"#),
            Some(Source::ClaudeCode)
        );
        assert_eq!(
            detect(r#"{"type":"session_meta","payload":{"id":"s"}}"#),
            Some(Source::Codex)
        );
    }

    #[test]
    fn unknown_format_is_an_error_not_a_guess() {
        assert_eq!(detect("just some text"), None);
        assert!(parse_auto("just some text", None).is_err());
    }

    #[test]
    fn source_names_round_trip() {
        assert_eq!(
            Source::parse_name("claude-code").unwrap(),
            Source::ClaudeCode
        );
        assert_eq!(Source::parse_name("CODEX").unwrap(), Source::Codex);
        assert!(Source::parse_name("nope").is_err());
    }

    #[test]
    fn known_roots_cover_the_verified_adapters_only() {
        let roots = roots_under(Path::new("/home/ada"));
        let paths: Vec<String> = roots.iter().map(|(p, _)| p.display().to_string()).collect();
        assert_eq!(
            paths,
            vec!["/home/ada/.claude/projects", "/home/ada/.codex/sessions"]
        );
        for (_, tag) in &roots {
            let source = Source::parse_name(tag).unwrap();
            assert!(!source.is_unverified(), "{tag} has no discoverable root");
        }
    }

    #[test]
    fn unverified_adapters_are_flagged() {
        assert!(Source::ChatGpt.is_unverified());
        assert!(Source::ClaudeApp.is_unverified());
        assert!(!Source::ClaudeCode.is_unverified());
        assert!(!Source::Codex.is_unverified());
    }
}
