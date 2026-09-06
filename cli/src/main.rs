//! HumaneBench CLI — score your own conversation history against the HumaneBench v3
//! rubric and emit a local HTML report.
//!
//! Ingest, score, and report are separate commands on purpose. Scoring is the only one
//! that costs money, so it is the only one you have to consciously invoke — and
//! re-rendering a report never re-spends. Runs only when invoked: no daemon, no hook, no
//! background scoring.

mod adapters;
mod judge;
mod mcp;
mod report;
mod store;
mod transcript;

use adapters::Source;
use anyhow::{bail, Context, Result};
use chrono::{DateTime, Duration, Utc};
use clap::{Parser, Subcommand};
use judge::{Judge, Provider, Tier};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::PathBuf;
use store::{Filter, Store};

#[derive(Parser)]
#[command(
    name = "humanebench",
    about = "Score your own conversation history against the HumaneBench v3 rubric",
    version
)]
struct Cli {
    /// Path to the local store.
    #[arg(long, global = true)]
    db: Option<PathBuf>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Detect source, normalize, and write transcripts to the local store.
    Ingest {
        /// File, directory, or export .zip. Omit to sweep the known transcript folders.
        path: Option<PathBuf>,
        /// Force the source instead of detecting it.
        #[arg(long)]
        source: Option<String>,
        /// Read pre-normalized JSONL from your own adapter.
        #[arg(long)]
        stdin: bool,
        /// Idle gap, in hours, that splits one thread into separate sessions.
        #[arg(long, default_value_t = transcript::DEFAULT_IDLE_GAP_HOURS)]
        idle_gap_hours: i64,
    },
    /// Judge everything not already cached.
    Score {
        /// Print the call count and token estimate without spending anything.
        #[arg(long)]
        dry_run: bool,
        /// Judge backend.
        #[arg(long, value_parser = ["openrouter", "vertex"], default_value = "vertex")]
        provider: String,
        /// Judge model, as the chosen provider names it.
        #[arg(long)]
        model: Option<String>,
        /// Stop after this many judge calls.
        #[arg(long)]
        limit: Option<usize>,
        /// Skip only the turn tier, or only the rollup tier.
        #[arg(long, value_parser = ["turn", "rollup", "both"], default_value = "both")]
        tier: String,
        /// Only score turns at or after this point, e.g. 3d, 12w, 6m, or an RFC 3339 date.
        #[arg(long)]
        since: Option<String>,
        /// Assume consent has already been given (for non-interactive use).
        #[arg(long)]
        yes: bool,
        #[arg(long, default_value_t = transcript::DEFAULT_IDLE_GAP_HOURS)]
        idle_gap_hours: i64,
    },
    /// Render HTML from cached scores. Never calls the judge.
    Report {
        /// Time window, e.g. 30d, 12w, 6m, or an RFC 3339 date.
        #[arg(long)]
        since: Option<String>,
        #[arg(long)]
        source: Option<String>,
        #[arg(long)]
        model: Option<String>,
        /// Output path.
        #[arg(long, default_value = "report.html")]
        out: PathBuf,
        /// Emit the excerpt-free artifact instead.
        #[arg(long)]
        share: bool,
        #[arg(long, default_value_t = transcript::DEFAULT_IDLE_GAP_HOURS)]
        idle_gap_hours: i64,
    },
    /// Serve the scored corpus over stdio as MCP.
    Mcp,
    /// Print the normalized transcript contract.
    Schema,
}

fn parse_since(s: &str) -> Result<DateTime<Utc>> {
    if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
        return Ok(dt.with_timezone(&Utc));
    }
    if let Ok(d) = chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        return Ok(DateTime::from_naive_utc_and_offset(
            d.and_time(chrono::NaiveTime::MIN),
            Utc,
        ));
    }
    let (num, unit) = s.split_at(s.len().saturating_sub(1));
    let n: i64 = num
        .parse()
        .with_context(|| format!("could not read a time window from {s:?}"))?;
    let dur = match unit {
        "d" => Duration::days(n),
        "w" => Duration::weeks(n),
        "m" => Duration::days(n * 30),
        "y" => Duration::days(n * 365),
        _ => bail!("unknown time unit {unit:?}; use d, w, m, or y (e.g. 30d)"),
    };
    Ok(Utc::now() - dur)
}

/// The trust boundary, stated exactly. One arrow crosses it, and only after this.
fn consent_text(destination: &str, model: &str, calls: usize) -> String {
    format!(
        "\n\
         ── One-time consent ────────────────────────────────────────────────\n\
         \n\
         Scoring sends conversation text off this machine. Specifically:\n\
         \n\
           WHAT LEAVES:   for each turn, the assistant's response text, the user\n\
                          prompt it was answering, a one-line summary of any tools\n\
                          it called, and the HumaneBench rubric. For each session,\n\
                          the conversation arc in full.\n\
           WHERE TO:      {destination}\n\
                          Judge model: {model}\n\
           WHAT COMES BACK: eight principle scores, rationales for negative scores,\n\
                          and a confidence value. Stored locally in SQLite.\n\
           HOW OFTEN:     once per turn, ever. Results are cached by content hash,\n\
                          so re-running never re-sends the same turn.\n\
           WHAT NEVER LEAVES: the report files, the store, and any turn you have\n\
                          not scored. Nothing is uploaded; sharing a report is a\n\
                          separate, deliberate command.\n\
         \n\
         This run would make {calls} judge call(s).\n\
         \n\
         Type 'yes' to consent and continue: "
    )
}

fn prompt_consent(destination: &str, model: &str, calls: usize) -> Result<bool> {
    use std::io::Write;
    print!("{}", consent_text(destination, model, calls));
    std::io::stdout().flush()?;
    let mut line = String::new();
    std::io::stdin().read_line(&mut line)?;
    Ok(line.trim().eq_ignore_ascii_case("yes"))
}

fn read_stdin() -> Result<String> {
    let mut buf = String::new();
    std::io::stdin()
        .read_to_string(&mut buf)
        .context("reading normalized JSONL from stdin")?;
    Ok(buf)
}

/// Total branch records discarded by the flattening rule, for the report's disclosure.
fn discarded_branches(records: &[transcript::Record]) -> usize {
    let mut by_session: BTreeMap<String, Vec<transcript::Record>> = BTreeMap::new();
    for r in records {
        by_session
            .entry(r.session_id.clone())
            .or_default()
            .push(r.clone());
    }
    by_session
        .into_values()
        .map(|thread| transcript::flatten(thread).1)
        .sum()
}

/// Sweep the known transcript roots, skipping the ones that aren't on this machine.
///
/// One unreadable root does not sink the sweep: it is recorded as a note and the rest
/// still ingest, the same way one unparseable file does inside a single root.
fn ingest_roots(
    candidates: &[(PathBuf, &'static str)],
    forced: Option<Source>,
) -> Result<(Vec<transcript::Record>, Vec<String>)> {
    let (found, absent): (Vec<_>, Vec<_>) = candidates.iter().partition(|(p, _)| p.is_dir());

    if found.is_empty() {
        let looked: Vec<String> = candidates
            .iter()
            .map(|(p, tag)| format!("    {} ({tag})", p.display()))
            .collect();
        bail!(
            "found none of the known transcript folders. Looked in:\n{}\n\n\
             Give a path explicitly (`humanebench ingest <path>`), or pipe normalized \
             JSONL with `--stdin` (see `humanebench schema`).",
            looked.join("\n")
        );
    }

    // A root can hold thousands of files and hundreds of megabytes, so name what is
    // being swept before the per-file notes start arriving.
    for (path, tag) in &found {
        eprintln!("  scanning {} ({tag})", path.display());
    }
    for (path, tag) in &absent {
        eprintln!("  skipping {} ({tag}): not present", path.display());
    }

    let mut records = Vec::new();
    let mut notes = Vec::new();
    for (path, _) in &found {
        match adapters::ingest_path(path, forced) {
            Ok((mut recs, mut root_notes)) => {
                records.append(&mut recs);
                notes.append(&mut root_notes);
            }
            Err(e) => notes.push(format!("{}: skipped ({e})", path.display())),
        }
    }

    if records.is_empty() {
        bail!(
            "no usable conversation records found in the known transcript folders.\n\n{}",
            notes.join("\n")
        );
    }
    Ok((records, notes))
}

fn cmd_ingest(
    store: &mut Store,
    path: Option<PathBuf>,
    source: Option<String>,
    use_stdin: bool,
    idle_gap_hours: i64,
) -> Result<()> {
    let forced = source.as_deref().map(Source::parse_name).transpose()?;

    let (records, notes) = if use_stdin {
        let input = read_stdin()?;
        if input.trim().is_empty() {
            bail!("--stdin was passed but nothing arrived on stdin");
        }
        let (src, recs) = adapters::parse_auto(&input, forced.or(Some(Source::Normalized)))?;
        let note = format!("stdin: {} records via {}", recs.len(), src.tag());
        (recs, vec![note])
    } else if let Some(path) = path {
        adapters::ingest_path(&path, forced)?
    } else {
        ingest_roots(&adapters::default_roots(), forced)?
    };

    for note in &notes {
        eprintln!("  {note}");
    }

    if let Some(src) = forced {
        if src.is_unverified() {
            eprintln!(
                "\n  ⚠ The {} adapter's field names were never confirmed against a real export.\n    \
                 Check the ingested counts below look right before trusting any scores.",
                src.tag()
            );
        }
    }

    let n = store.upsert_records(&records)?;
    let dropped = discarded_branches(&records);
    let sessions = transcript::sessionize(records, idle_gap_hours);
    let scorable: usize = sessions
        .iter()
        .map(|s| transcript::scorable_turns(s).len())
        .sum();

    println!("Ingested {n} records into {} session(s).", sessions.len());
    println!("{scorable} assistant turn(s) are scorable.");
    if dropped > 0 {
        println!("{dropped} branch record(s) discarded (edits, regenerations, abandoned paths).");
    }
    println!("\nNext: `humanebench score --dry-run` to see what scoring would cost.");
    Ok(())
}

struct Plan {
    turns: Vec<(transcript::ScorableTurn, String, String)>,
    rollups: Vec<(transcript::Session, String, String)>,
}

/// Work out what still needs judging. Cache hits never appear here.
///
/// `since` selects what to score and never reaches a prompt: a session that straddles the
/// cutoff is rolled up whole. Trimming it to the in-window records would change the rollup
/// prompt and so its content hash, and a wider window later would re-judge the overlap
/// instead of reusing it.
fn build_plan(
    store: &Store,
    judge_model: &str,
    idle_gap_hours: i64,
    tier: &str,
    since: Option<DateTime<Utc>>,
) -> Result<Plan> {
    let records = store.all_records()?;
    let sessions = transcript::sessionize(records, idle_gap_hours);

    let mut turns = Vec::new();
    let mut rollups = Vec::new();

    for session in sessions {
        // Sessionizing and prompt-pairing both run over the whole corpus first: the
        // cutoff must not shift idle-gap boundaries or strand a turn from the user
        // prompt it was answering.
        let in_window: Vec<transcript::ScorableTurn> = transcript::scorable_turns(&session)
            .into_iter()
            .filter(|t| since.is_none_or(|cutoff| t.timestamp >= cutoff))
            .collect();
        // A session with nothing scorable in the window has no arc worth rolling up.
        let worth_rolling_up = !in_window.is_empty();

        if tier == "both" || tier == "turn" {
            for turn in in_window {
                let prompt = judge::assemble_turn_prompt(&turn);
                let hash = judge::content_hash(&prompt, judge_model);
                if !store.has_score(&hash)? {
                    turns.push((turn, prompt, hash));
                }
            }
        }
        if (tier == "both" || tier == "rollup") && worth_rolling_up {
            let prompt = judge::rollup::assemble_rollup_prompt(&session);
            let hash = judge::content_hash(&prompt, judge_model);
            if !store.has_score(&hash)? {
                rollups.push((session, prompt, hash));
            }
        }
    }

    Ok(Plan { turns, rollups })
}

/// The model to call and the label recorded against every score, from flags alone.
/// `--dry-run` reports the judge through this and constructs nothing, which is what keeps
/// it usable on a machine that has no credentials at all.
fn resolve_judge(provider: Provider, model: Option<String>) -> (String, String) {
    let model = model.unwrap_or_else(|| provider.default_model().to_string());
    let label = provider.label(&model);
    (model, label)
}

#[allow(clippy::too_many_arguments)]
fn cmd_score(
    store: &mut Store,
    dry_run: bool,
    provider: String,
    model: Option<String>,
    limit: Option<usize>,
    tier: String,
    since: Option<String>,
    yes: bool,
    idle_gap_hours: i64,
) -> Result<()> {
    if store.count_turns()? == 0 {
        bail!("the store is empty — run `humanebench ingest <path>` first");
    }

    let provider = Provider::parse_name(&provider)?;
    let (model, judge_model) = resolve_judge(provider, model);

    let since = since.as_deref().map(parse_since).transpose()?;
    let window = since.map(|s| format!("since {}", s.format("%Y-%m-%d %H:%M UTC")));

    let plan = build_plan(store, &judge_model, idle_gap_hours, &tier, since)?;
    let total = plan.turns.len() + plan.rollups.len();

    if dry_run {
        let turn_tokens: usize = plan
            .turns
            .iter()
            .map(|(_, p, _)| judge::estimate_tokens(p))
            .sum();
        let rollup_tokens: usize = plan
            .rollups
            .iter()
            .map(|(_, p, _)| judge::estimate_tokens(p))
            .sum();

        println!("Judge model: {judge_model}");
        println!("Regime:      {} (single judge)", judge::REGIME);
        match &window {
            Some(w) => println!("Window:      {w} — everything older stays pending"),
            None => println!("Window:      the whole pending corpus"),
        }
        println!();
        println!("Turn-tier calls needed:   {}", plan.turns.len());
        println!("Rollup calls needed:      {}", plan.rollups.len());
        println!("Total calls:              {total}");
        println!();
        println!("Estimated input tokens:   ~{}", turn_tokens + rollup_tokens);
        println!("  turn tier:              ~{turn_tokens}");
        println!("  rollups:                ~{rollup_tokens}");
        println!();
        println!(
            "Token counts are a crude estimate (chars/4) and exclude output. Multiply by your \n\
             judge model's input price to get a cost. Cached turns are already excluded, so \n\
             this is what a real run would spend, not what the whole corpus would cost."
        );
        return Ok(());
    }

    if total == 0 {
        match &window {
            Some(w) => println!("Nothing pending {w} — that slice is already scored."),
            None => println!("Everything is already scored — nothing to do."),
        }
        println!("Re-render any time with `humanebench report` (free, never re-judges).");
        return Ok(());
    }

    // Above the consent prompt on purpose: the call count it quotes is meaningless
    // without the slice it covers.
    match &window {
        Some(w) => println!("Scoring {total} pending item(s) {w}; everything older stays pending."),
        None => println!("Scoring {total} pending item(s) — the whole pending corpus."),
    }

    let judge = Judge::new(provider, &model)?;
    let destination = judge.destination();
    let judge_model = judge.labelled_model();

    if !store.consent_granted(&destination)? {
        if yes {
            store.grant_consent(&destination)?;
        } else if prompt_consent(&destination, judge.model(), total)? {
            store.grant_consent(&destination)?;
            println!("Consent recorded for {destination}. This will not be asked again.\n");
        } else {
            println!("No consent given. Nothing was sent and nothing was scored.");
            return Ok(());
        }
    }

    let cap = limit.unwrap_or(usize::MAX);
    let mut done = 0usize;
    let mut failed = 0usize;
    let mut spent = judge::Usage::default();

    for (turn, prompt, hash) in &plan.turns {
        if done >= cap {
            break;
        }
        eprint!("\r  scoring turn {}/{}…", done + 1, total.min(cap));
        match judge_one(&judge, prompt) {
            Ok((j, usage)) => {
                spent.prompt_tokens += usage.prompt_tokens;
                spent.completion_tokens += usage.completion_tokens;
                let rec = store::score_record(
                    &turn.turn_id,
                    &turn.session_id,
                    Tier::Turn,
                    hash,
                    &judge_model,
                    judge::REGIME,
                    j,
                );
                store
                    .insert_score_at(&rec, &turn.source, turn.model.as_deref(), turn.timestamp)
                    .with_context(|| format!("storing the score for turn {}", turn.turn_id))?;
            }
            Err(e) => {
                failed += 1;
                eprintln!("\n  ! turn {} failed: {e}", turn.turn_id);
            }
        }
        done += 1;
    }

    for (session, prompt, hash) in &plan.rollups {
        if done >= cap {
            break;
        }
        eprint!("\r  scoring rollup {}/{}…", done + 1, total.min(cap));
        match judge_one(&judge, prompt) {
            Ok((j, usage)) => {
                spent.prompt_tokens += usage.prompt_tokens;
                spent.completion_tokens += usage.completion_tokens;
                let rec = store::score_record(
                    &format!("{}:rollup", session.session_id),
                    &session.session_id,
                    Tier::Rollup,
                    hash,
                    &judge_model,
                    judge::REGIME,
                    j,
                );
                let ts = session.started_at().unwrap_or_else(Utc::now);
                store
                    .insert_score_at(&rec, &session.source, None, ts)
                    .with_context(|| {
                        format!(
                            "storing the rollup score for session {}",
                            session.session_id
                        )
                    })?;
            }
            Err(e) => {
                failed += 1;
                eprintln!("\n  ! rollup {} failed: {e}", session.session_id);
            }
        }
        done += 1;
    }

    eprintln!("\r                                          ");
    println!("Scored {} of {total} pending item(s).", done - failed);
    if failed > 0 {
        println!("{failed} failed and were not cached; re-run to retry just those.");
    }
    // The real number, as reported by the API — not the crude dry-run estimate.
    if spent.prompt_tokens > 0 || spent.completion_tokens > 0 {
        println!(
            "Tokens actually used: {} input + {} output. Multiply by your judge model's \
             prices for the real cost of this run.",
            spent.prompt_tokens, spent.completion_tokens
        );
    }
    println!("\nNext: `humanebench report --out report.html`");
    Ok(())
}

fn judge_one(judge: &Judge, prompt: &str) -> Result<(judge::Judgement, judge::Usage)> {
    let (raw, usage) = judge.complete(prompt)?;
    Ok((judge::parse_judgement(&raw)?, usage))
}

fn cmd_report(
    store: &Store,
    since: Option<String>,
    source: Option<String>,
    model: Option<String>,
    out: PathBuf,
    share: bool,
    idle_gap_hours: i64,
) -> Result<()> {
    let since = since.as_deref().map(parse_since).transpose()?;
    let filter = Filter {
        since,
        until: None,
        source: source.clone(),
        model: model.clone(),
        tier: None,
        session_id: None,
    };

    let scores = store.scores(&filter)?;
    if scores.is_empty() {
        bail!(
            "no cached scores match that filter.\n\n\
             Run `humanebench score` first, or widen --since / --source / --model."
        );
    }

    // Excerpts are only needed for the full report.
    let mut excerpts = BTreeMap::new();
    if !share {
        for s in &scores {
            if let Some(text) = store.turn_text(&s.record.turn_id)? {
                excerpts.insert(s.record.turn_id.clone(), text);
            }
        }
    }

    let records = store.all_records()?;
    let unverified: Vec<String> = records
        .iter()
        .map(|r| r.source.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .filter(|s| {
            Source::parse_name(s)
                .map(|src| src.is_unverified())
                .unwrap_or(false)
        })
        .collect();

    let mut filter_note = Vec::new();
    if let Some(s) = &since {
        filter_note.push(format!("since {}", s.format("%Y-%m-%d")));
    }
    if let Some(s) = &source {
        filter_note.push(format!("source {s}"));
    }
    if let Some(m) = &model {
        filter_note.push(format!("model {m}"));
    }

    let input = report::ReportInput {
        judge_models: store.distinct("judge_model")?,
        regimes: store.distinct("regime")?,
        sources: store.distinct("source")?,
        generated_at: Utc::now(),
        discarded_branches: discarded_branches(&records),
        filter_note: if filter_note.is_empty() {
            None
        } else {
            Some(filter_note.join(", "))
        },
        unverified_sources: unverified,
        excerpts,
        scores,
    };

    let html = if share {
        report::render_share(&input)
    } else {
        report::render_full(&input)
    };

    std::fs::write(&out, html).with_context(|| format!("writing {}", out.display()))?;

    let turns = input
        .scores
        .iter()
        .filter(|s| s.record.tier == Tier::Turn)
        .count();
    let rollups = input.scores.len() - turns;

    println!(
        "Wrote {} ({} turn scores, {} rollups).",
        out.display(),
        turns,
        rollups
    );
    if share {
        println!("Shared summary: excerpts, judge reasoning, and citations are excluded.");
    } else {
        println!("Private report: contains verbatim excerpts. `--share` emits the safe artifact.");
    }
    let _ = idle_gap_hours;
    Ok(())
}

fn cmd_schema() {
    println!(
        r#"The normalized transcript schema — humanebench.transcript/v1

One JSONL record per message. This is the entire contract: an adapter that emits this is
a first-class citizen with no code in the binary.

  schema      req  Literal "humanebench.transcript/v1".
  source      req  Free-form origin tag ("codex", "chatgpt"). Surfaced, never parsed.
  session_id  req  Groups records into one thread before idle-gap splitting.
  turn_id     req  Stable and unique ACROSS RUNS. Re-running an adapter must produce the
                   same ids or the cache misses and you pay twice.
  role        req  "user" or "assistant". Anything else is dropped.
  text        req  What the person wrote or saw. Never emit empty/whitespace-only records.
  timestamp   req  RFC 3339. Drives idle-gap splitting and the trend axis.
  parent_id   opt  Parent turn_id. Present for trees, absent for flat sources.
  model       opt  Model that produced an assistant turn.
  actions     opt  Array of {{name, summary}} for tools/MCP/skills used for this turn.
                   Context for the judge, never scored on its own.
  sidechain   opt  Bool. True marks a turn no human saw; excluded from scoring.

Example:

  {{"schema":"humanebench.transcript/v1","source":"codex",
   "session_id":"0199b07d-60a4-7f93-bfd6-6cafc456607e",
   "turn_id":"0199b07d-...:0007","role":"user","text":"why is the upload flaky?",
   "timestamp":"2025-10-04T18:30:30.860Z"}}

Pipe it in:

  your-converter < logs | humanebench ingest --stdin --source yourtool

Flattening: for tree-shaped sources, either emit parent_id and let the binary walk from
the newest leaf to the root, or flatten yourself and emit a linear file. Both are valid;
doing it twice is not."#
    );
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let db_path = cli.db.unwrap_or_else(store::default_db_path);

    match cli.command {
        Command::Schema => {
            cmd_schema();
            Ok(())
        }
        Command::Ingest {
            path,
            source,
            stdin,
            idle_gap_hours,
        } => {
            let mut store = Store::open(&db_path)?;
            cmd_ingest(&mut store, path, source, stdin, idle_gap_hours)
        }
        Command::Score {
            dry_run,
            provider,
            model,
            limit,
            tier,
            since,
            yes,
            idle_gap_hours,
        } => {
            let mut store = Store::open(&db_path)?;
            cmd_score(
                &mut store,
                dry_run,
                provider,
                model,
                limit,
                tier,
                since,
                yes,
                idle_gap_hours,
            )
        }
        Command::Report {
            since,
            source,
            model,
            out,
            share,
            idle_gap_hours,
        } => {
            let store = Store::open(&db_path)?;
            cmd_report(&store, since, source, model, out, share, idle_gap_hours)
        }
        Command::Mcp => {
            let store = Store::open(&db_path)?;
            mcp::serve(&store)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use transcript::{Record, Role};

    #[test]
    fn parses_relative_and_absolute_since() {
        assert!(parse_since("30d").is_ok());
        assert!(parse_since("12w").is_ok());
        assert!(parse_since("6m").is_ok());
        assert!(parse_since("2026-01-01").is_ok());
        assert!(parse_since("2026-01-01T00:00:00Z").is_ok());
        assert!(parse_since("nonsense").is_err());
        assert!(parse_since("30x").is_err());
    }

    #[test]
    fn since_30d_is_about_a_month_ago() {
        let t = parse_since("30d").unwrap();
        let delta = Utc::now() - t;
        assert!(delta.num_days() >= 29 && delta.num_days() <= 31);
    }

    #[test]
    fn consent_text_states_the_data_flow_concretely() {
        let t = consent_text(
            judge::openrouter::DESTINATION,
            "anthropic/claude-sonnet-4.5",
            42,
        );
        assert!(t.contains("WHAT LEAVES"));
        assert!(t.contains("OpenRouter"));
        assert!(t.contains("anthropic/claude-sonnet-4.5"));
        assert!(t.contains("42 judge call"));
        assert!(t.contains("WHAT NEVER LEAVES"));
        assert!(t.contains("cached by content hash"));
    }

    /// The project is the disclosure that matters: a company GCP project must be on
    /// screen before anyone types "yes".
    #[test]
    fn consent_text_names_the_vertex_project_and_location() {
        let t = consent_text(
            "Google Vertex AI (project acme-prod, location europe-west4)",
            "gemini-3.1-pro-preview",
            7,
        );
        assert!(t.contains("acme-prod"));
        assert!(t.contains("europe-west4"));
        assert!(!t.contains("OpenRouter"));
    }

    /// --dry-run is documented as needing no key, so the label it prints must resolve
    /// from flags alone — including when the model falls back to the provider default.
    #[test]
    fn dry_run_labels_resolve_without_credentials() {
        let (model, label) = resolve_judge(Provider::Vertex, None);
        assert_eq!(model, judge::vertex::DEFAULT_MODEL);
        assert_eq!(label, format!("vertex/{}", judge::vertex::DEFAULT_MODEL));

        let (_, label) = resolve_judge(
            Provider::OpenRouter,
            Some("anthropic/claude-sonnet-4.5".into()),
        );
        assert_eq!(label, "openrouter/anthropic/claude-sonnet-4.5");
    }

    fn ts(s: &str) -> DateTime<Utc> {
        DateTime::parse_from_rfc3339(s).unwrap().with_timezone(&Utc)
    }

    /// With nothing to discover, the error has to carry every route forward: where it
    /// looked, and the two ways to ingest that don't rely on discovery at all.
    #[test]
    fn absent_roots_name_where_it_looked_and_the_other_routes() {
        let roots = vec![
            (
                PathBuf::from("/nonexistent-home/.claude/projects"),
                "claude-code",
            ),
            (PathBuf::from("/nonexistent-home/.codex/sessions"), "codex"),
        ];
        let err = ingest_roots(&roots, None).unwrap_err().to_string();
        assert!(err.contains("/nonexistent-home/.claude/projects"), "{err}");
        assert!(err.contains("/nonexistent-home/.codex/sessions"), "{err}");
        assert!(err.contains("humanebench ingest <path>"), "{err}");
        assert!(err.contains("--stdin"), "{err}");
    }

    #[test]
    fn counts_discarded_branches() {
        let mut a = Record::new("t", "s1", "a", Role::User, "a", ts("2026-01-01T00:00:00Z"));
        a.parent_id = None;
        let mut b = Record::new(
            "t",
            "s1",
            "b",
            Role::Assistant,
            "b",
            ts("2026-01-01T00:01:00Z"),
        );
        b.parent_id = Some("a".into());
        let mut b2 = Record::new(
            "t",
            "s1",
            "b2",
            Role::Assistant,
            "b2",
            ts("2026-01-01T00:02:00Z"),
        );
        b2.parent_id = Some("a".into());
        let mut c = Record::new("t", "s1", "c", Role::User, "c", ts("2026-01-01T00:03:00Z"));
        c.parent_id = Some("b".into());

        assert_eq!(discarded_branches(&[a, b, b2, c]), 1);
    }

    #[test]
    fn plan_is_empty_once_everything_is_cached() {
        let mut store = Store::open_in_memory().unwrap();
        let recs = vec![
            Record::new(
                "claude-code",
                "s1",
                "u1",
                Role::User,
                "hi",
                ts("2026-01-01T00:00:00Z"),
            ),
            Record::new(
                "claude-code",
                "s1",
                "a1",
                Role::Assistant,
                "hello",
                ts("2026-01-01T00:01:00Z"),
            ),
        ];
        store.upsert_records(&recs).unwrap();

        let plan = build_plan(&store, "openrouter/m", 6, "both", None).unwrap();
        assert_eq!(plan.turns.len(), 1);
        assert_eq!(plan.rollups.len(), 1);

        // Cache both, then the plan must be empty.
        let judgement = judge::Judgement {
            principles: judge::PRINCIPLES
                .iter()
                .map(|n| judge::PrincipleScore {
                    name: n.to_string(),
                    score: 0.5,
                    rationale: None,
                })
                .collect(),
            global_violations: vec![],
            confidence: 0.8,
        };
        for (_, _, hash) in &plan.turns {
            let rec = store::score_record(
                "a1",
                "s1",
                Tier::Turn,
                hash,
                "openrouter/m",
                "single",
                judgement.clone(),
            );
            store.insert_score(&rec, "claude-code", None).unwrap();
        }
        for (_, _, hash) in &plan.rollups {
            let rec = store::score_record(
                "s1:rollup",
                "s1",
                Tier::Rollup,
                hash,
                "openrouter/m",
                "single",
                judgement.clone(),
            );
            store.insert_score(&rec, "claude-code", None).unwrap();
        }

        let plan2 = build_plan(&store, "openrouter/m", 6, "both", None).unwrap();
        assert_eq!(
            plan2.turns.len(),
            0,
            "a cached turn must never be re-judged"
        );
        assert_eq!(plan2.rollups.len(), 0);
    }

    #[test]
    fn changing_the_judge_model_invalidates_the_plan() {
        let mut store = Store::open_in_memory().unwrap();
        let recs = vec![
            Record::new(
                "claude-code",
                "s1",
                "u1",
                Role::User,
                "hi",
                ts("2026-01-01T00:00:00Z"),
            ),
            Record::new(
                "claude-code",
                "s1",
                "a1",
                Role::Assistant,
                "hello",
                ts("2026-01-01T00:01:00Z"),
            ),
        ];
        store.upsert_records(&recs).unwrap();

        let plan = build_plan(&store, "openrouter/model-a", 6, "both", None).unwrap();
        let judgement = judge::Judgement {
            principles: judge::PRINCIPLES
                .iter()
                .map(|n| judge::PrincipleScore {
                    name: n.to_string(),
                    score: 0.5,
                    rationale: None,
                })
                .collect(),
            global_violations: vec![],
            confidence: 0.8,
        };
        for (_, _, hash) in &plan.turns {
            let rec = store::score_record(
                "a1",
                "s1",
                Tier::Turn,
                hash,
                "openrouter/model-a",
                "single",
                judgement.clone(),
            );
            store.insert_score(&rec, "claude-code", None).unwrap();
        }

        let plan_b = build_plan(&store, "openrouter/model-b", 6, "turn", None).unwrap();
        assert_eq!(
            plan_b.turns.len(),
            1,
            "a model swap must re-judge, not serve stale scores"
        );
    }

    #[test]
    fn tier_filter_limits_the_plan() {
        let mut store = Store::open_in_memory().unwrap();
        let recs = vec![
            Record::new(
                "claude-code",
                "s1",
                "u1",
                Role::User,
                "hi",
                ts("2026-01-01T00:00:00Z"),
            ),
            Record::new(
                "claude-code",
                "s1",
                "a1",
                Role::Assistant,
                "hello",
                ts("2026-01-01T00:01:00Z"),
            ),
        ];
        store.upsert_records(&recs).unwrap();

        assert_eq!(
            build_plan(&store, "m", 6, "turn", None)
                .unwrap()
                .rollups
                .len(),
            0
        );
        assert_eq!(
            build_plan(&store, "m", 6, "rollup", None)
                .unwrap()
                .turns
                .len(),
            0
        );
    }

    /// One session entirely before the cutoff, one straddling it.
    fn straddling_store() -> Store {
        let mut store = Store::open_in_memory().unwrap();
        let recs = vec![
            Record::new(
                "claude-code",
                "s0",
                "u0",
                Role::User,
                "old question",
                ts("2026-01-01T00:00:00Z"),
            ),
            Record::new(
                "claude-code",
                "s0",
                "a0",
                Role::Assistant,
                "old answer",
                ts("2026-01-01T00:01:00Z"),
            ),
            Record::new(
                "claude-code",
                "s1",
                "u1",
                Role::User,
                "first question",
                ts("2026-01-01T00:00:00Z"),
            ),
            Record::new(
                "claude-code",
                "s1",
                "a1",
                Role::Assistant,
                "first answer",
                ts("2026-01-01T00:01:00Z"),
            ),
            Record::new(
                "claude-code",
                "s1",
                "u2",
                Role::User,
                "later question",
                ts("2026-01-01T02:00:00Z"),
            ),
            Record::new(
                "claude-code",
                "s1",
                "a2",
                Role::Assistant,
                "later answer",
                ts("2026-01-01T02:01:00Z"),
            ),
        ];
        store.upsert_records(&recs).unwrap();
        store
    }

    #[test]
    fn since_shrinks_the_plan_to_the_window() {
        let store = straddling_store();
        let cutoff = ts("2026-01-01T01:00:00Z");

        let all = build_plan(&store, "m", 6, "both", None).unwrap();
        assert_eq!(all.turns.len(), 3);
        assert_eq!(all.rollups.len(), 2);

        let windowed = build_plan(&store, "m", 6, "both", Some(cutoff)).unwrap();
        let turn_ids: Vec<&str> = windowed
            .turns
            .iter()
            .map(|(t, _, _)| t.turn_id.as_str())
            .collect();
        assert_eq!(turn_ids, ["a2"]);

        let rollup_ids: Vec<&str> = windowed
            .rollups
            .iter()
            .map(|(s, _, _)| s.session_id.as_str())
            .collect();
        assert_eq!(
            rollup_ids,
            ["s1"],
            "a session whose only turns fell outside the cutoff must not be rolled up"
        );
    }

    /// The straddling session is rolled up whole, so the cutoff leaves both hashes alone
    /// and a later, wider run reuses the overlap instead of paying for it again.
    #[test]
    fn since_never_enters_the_content_hash() {
        let store = straddling_store();
        let all = build_plan(&store, "m", 6, "both", None).unwrap();
        let windowed =
            build_plan(&store, "m", 6, "both", Some(ts("2026-01-01T01:00:00Z"))).unwrap();

        let hash_of = |plan: &Plan, id: &str| {
            plan.turns
                .iter()
                .find(|(t, _, _)| t.turn_id == id)
                .map(|(_, _, h)| h.clone())
                .unwrap()
        };
        assert_eq!(hash_of(&all, "a2"), hash_of(&windowed, "a2"));

        let rollup_hash_of = |plan: &Plan, id: &str| {
            plan.rollups
                .iter()
                .find(|(s, _, _)| s.session_id == id)
                .map(|(_, _, h)| h.clone())
                .unwrap()
        };
        assert_eq!(rollup_hash_of(&all, "s1"), rollup_hash_of(&windowed, "s1"));

        // Pay for the narrow window, then widen it: the overlap must not come back.
        let judgement = judge::Judgement {
            principles: judge::PRINCIPLES
                .iter()
                .map(|n| judge::PrincipleScore {
                    name: n.to_string(),
                    score: 0.5,
                    rationale: None,
                })
                .collect(),
            global_violations: vec![],
            confidence: 0.8,
        };
        for (turn, _, hash) in &windowed.turns {
            let rec = store::score_record(
                &turn.turn_id,
                &turn.session_id,
                Tier::Turn,
                hash,
                "m",
                "single",
                judgement.clone(),
            );
            store.insert_score(&rec, "claude-code", None).unwrap();
        }
        for (session, _, hash) in &windowed.rollups {
            let rec = store::score_record(
                &format!("{}:rollup", session.session_id),
                &session.session_id,
                Tier::Rollup,
                hash,
                "m",
                "single",
                judgement.clone(),
            );
            store.insert_score(&rec, "claude-code", None).unwrap();
        }

        let widened = build_plan(&store, "m", 6, "both", None).unwrap();
        assert!(widened.turns.iter().all(|(t, _, _)| t.turn_id != "a2"));
        assert!(widened.rollups.iter().all(|(s, _, _)| s.session_id != "s1"));
    }
}
