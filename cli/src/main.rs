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

/// One judge call and every score row it will produce.
///
/// Subjects share a job when their assembled prompts are byte-identical. That happens for
/// real: "thanks" answered "You're welcome!" in two different sessions assembles to the
/// same bytes, and both turns deserve a score even though only one call is warranted.
struct Job<T> {
    prompt: String,
    hash: String,
    subjects: Vec<T>,
    /// A judgement the store already holds for these exact bytes. When it is present the
    /// rows are filled from it and nothing is sent — the prompt has already been paid for.
    reuse: Option<judge::Judgement>,
}

impl<T> Job<T> {
    fn needs_a_call(&self) -> bool {
        self.reuse.is_none()
    }
}

struct Plan {
    turns: Vec<Job<transcript::ScorableTurn>>,
    rollups: Vec<Job<transcript::Session>>,
}

/// Jobs that would cost a call.
fn pending_calls<T>(jobs: &[Job<T>]) -> usize {
    jobs.iter().filter(|j| j.needs_a_call()).count()
}

/// Estimated input tokens for the jobs that would cost a call.
fn pending_tokens<T>(jobs: &[Job<T>]) -> usize {
    jobs.iter()
        .filter(|j| j.needs_a_call())
        .map(|j| judge::estimate_tokens(&j.prompt))
        .sum()
}

impl Plan {
    /// What a run would actually spend.
    fn calls(&self) -> usize {
        pending_calls(&self.turns) + pending_calls(&self.rollups)
    }

    /// Rows a run would write. Never fewer than [`Plan::calls`], and more whenever the
    /// corpus repeats itself.
    fn rows(&self) -> usize {
        self.turns.iter().map(|j| j.subjects.len()).sum::<usize>()
            + self.rollups.iter().map(|j| j.subjects.len()).sum::<usize>()
    }
}

/// Add a subject to the plan, folding it into an existing job when another subject has
/// already produced these exact bytes.
fn enqueue<T>(
    jobs: &mut Vec<Job<T>>,
    by_hash: &mut BTreeMap<String, usize>,
    store: &Store,
    prompt: String,
    hash: String,
    subject: T,
) -> Result<()> {
    if let Some(&i) = by_hash.get(&hash) {
        jobs[i].subjects.push(subject);
        return Ok(());
    }
    let reuse = store.judgement_for_hash(&hash)?;
    by_hash.insert(hash.clone(), jobs.len());
    jobs.push(Job {
        prompt,
        hash,
        subjects: vec![subject],
        reuse,
    });
    Ok(())
}

/// The stable identity of a session's rollup row — see [`store::rollup_identity`].
fn rollup_identity_of(session: &transcript::Session) -> String {
    store::rollup_identity(
        &session.session_id,
        session.started_at().unwrap_or_else(Utc::now),
    )
}

/// The session as the rollup judge is shown it: labelled with the thread root rather than
/// the sub-session number.
///
/// `assemble_rollup_prompt` prints the session id as context, so the positional `#N` that
/// `sessionize` assigns lands inside the hashed bytes. The first idle-gap split of a
/// thread renames `sA` to `sA#1`, and a chunk whose content has not changed by a single
/// character re-hashes and is billed all over again. Labelling every chunk with the
/// thread root keeps those bytes stable across the split. Nothing is lost: the id is
/// context the prompt itself calls "not itself scored", and the sub-session number is
/// what the row's `session_id` column records.
fn as_rollup_subject(session: &transcript::Session) -> transcript::Session {
    transcript::Session {
        session_id: store::thread_root(&session.session_id).to_string(),
        source: session.source.clone(),
        records: session.records.clone(),
    }
}

/// A rollup row is stamped at the *end* of the arc it judges, not the start.
///
/// `score --since T` rolls a straddling session up whole, so a session that began long
/// before `T` can be judged entirely because of turns after it. Stamping that row at the
/// session start put it outside the very window that selected it, and `report --since T`
/// then filtered out the rollup just paid for. The last instant the arc covers is always
/// inside the window that selected the session.
fn rollup_timestamp(session: &transcript::Session) -> DateTime<Utc> {
    session
        .records
        .last()
        .map(|r| r.timestamp)
        .or_else(|| session.started_at())
        .unwrap_or_else(Utc::now)
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
    // Deduplication runs across the whole corpus, not per session — the whole point is
    // that the repeat lives in a *different* session.
    let mut turns_by_hash = BTreeMap::new();
    let mut rollups_by_hash = BTreeMap::new();

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
                if store.is_scored(&turn.turn_id, Tier::Turn, judge_model, &hash)? {
                    continue;
                }
                enqueue(&mut turns, &mut turns_by_hash, store, prompt, hash, turn)?;
            }
        }
        if (tier == "both" || tier == "rollup") && worth_rolling_up {
            let prompt = judge::rollup::assemble_rollup_prompt(&as_rollup_subject(&session));
            let hash = judge::content_hash(&prompt, judge_model);
            let identity = rollup_identity_of(&session);
            if !store.is_scored(&identity, Tier::Rollup, judge_model, &hash)? {
                enqueue(
                    &mut rollups,
                    &mut rollups_by_hash,
                    store,
                    prompt,
                    hash,
                    session,
                )?;
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
    let total = plan.calls();
    let rows = plan.rows();

    if dry_run {
        let turn_tokens = pending_tokens(&plan.turns);
        let rollup_tokens = pending_tokens(&plan.rollups);
        let turn_calls = pending_calls(&plan.turns);
        let rollup_calls = pending_calls(&plan.rollups);

        println!("Judge model: {judge_model}");
        println!("Regime:      {} (single judge)", judge::REGIME);
        match &window {
            Some(w) => println!("Window:      {w} — everything older stays pending"),
            None => println!("Window:      the whole pending corpus"),
        }
        println!();
        println!("Turn-tier calls needed:   {turn_calls}");
        println!("Rollup calls needed:      {rollup_calls}");
        println!("Total calls:              {total}");
        println!("Scores this would write:  {rows}");
        println!();
        println!("Estimated input tokens:   ~{}", turn_tokens + rollup_tokens);
        println!("  turn tier:              ~{turn_tokens}");
        println!("  rollups:                ~{rollup_tokens}");
        println!();
        if rows > total {
            println!(
                "{} score(s) need no call of their own: the corpus repeats itself, and \n\
                 identical text is judged once and recorded against every turn it covers.\n",
                rows - total
            );
        }
        println!(
            "Token counts are a crude estimate (chars/4) and exclude output. Multiply by your \n\
             judge model's input price to get a cost. Cached turns are already excluded, so \n\
             this is what a real run would spend, not what the whole corpus would cost."
        );
        return Ok(());
    }

    if rows == 0 {
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
        Some(w) => println!("Scoring {rows} pending item(s) {w}; everything older stays pending."),
        None => println!("Scoring {rows} pending item(s) — the whole pending corpus."),
    }
    if rows > total {
        println!(
            "{} of them repeat text already judged and will be filled from the store.",
            rows - total
        );
    }

    // Only build a judge — and only ask for consent — if something actually has to be
    // sent. A run that is entirely reuse must work on a machine with no credentials.
    let judge = if total > 0 {
        let judge = Judge::new(provider, &model)?;
        let destination = judge.destination();
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
        Some(judge)
    } else {
        None
    };
    let judge_model = match &judge {
        Some(j) => j.labelled_model(),
        None => judge_model,
    };

    let mut run = Run {
        judge: judge.as_ref(),
        cap: limit.unwrap_or(usize::MAX),
        total,
        calls: 0,
        failed: 0,
        written: 0,
        reused: 0,
        spent: judge::Usage::default(),
    };

    for job in &plan.turns {
        let judgement = match run.step(job, "turn", &job.subjects[0].turn_id) {
            Step::Judged(j) => j,
            Step::Skip => continue,
            Step::Stop => break,
        };
        for turn in &job.subjects {
            let rec = store::score_record(
                &turn.turn_id,
                &turn.session_id,
                Tier::Turn,
                &job.hash,
                &judge_model,
                judge::REGIME,
                judgement.clone(),
            );
            store
                .insert_score_at(&rec, &turn.source, turn.model.as_deref(), turn.timestamp)
                .with_context(|| format!("storing the score for turn {}", turn.turn_id))?;
            run.written += 1;
        }
    }

    for job in &plan.rollups {
        let judgement = match run.step(job, "rollup", &job.subjects[0].session_id) {
            Step::Judged(j) => j,
            Step::Skip => continue,
            Step::Stop => break,
        };
        for session in &job.subjects {
            let rec = store::score_record(
                &format!("{}:rollup", session.session_id),
                &session.session_id,
                Tier::Rollup,
                &job.hash,
                &judge_model,
                judge::REGIME,
                judgement.clone(),
            );
            store
                .insert_score_as(
                    &rec,
                    &rollup_identity_of(session),
                    &session.source,
                    None,
                    rollup_timestamp(session),
                )
                .with_context(|| {
                    format!(
                        "storing the rollup score for session {}",
                        session.session_id
                    )
                })?;
            run.written += 1;
        }
    }

    eprintln!("\r                                          ");
    println!(
        "Scored {} pending item(s) in {} judge call(s).",
        run.written,
        run.calls - run.failed
    );
    if run.reused > 0 {
        println!(
            "{} of them reused a judgement already in the store — no call, no cost.",
            run.reused
        );
    }
    if run.failed > 0 {
        println!(
            "{} call(s) failed and were not cached; re-run to retry just those.",
            run.failed
        );
    }
    // The real number, as reported by the API — not the crude dry-run estimate.
    if run.spent.prompt_tokens > 0 || run.spent.completion_tokens > 0 {
        println!(
            "Tokens actually used: {} input + {} output. Multiply by your judge model's \
             prices for the real cost of this run.",
            run.spent.prompt_tokens, run.spent.completion_tokens
        );
    }
    // A run in which nothing survived is a failed run, and must not exit 0.
    score_outcome(run.calls, run.failed)?;
    println!("\nNext: `humanebench report --out report.html`");
    Ok(())
}

/// Whether a completed `score` run counts as a success.
///
/// A partial failure stays a success: the scores that landed are real, they are cached,
/// and the printout says to re-run for the rest. A run where every single call failed has
/// nothing to show and almost always means a bad key, no network, or a wrong model id —
/// exiting 0 there reports success to a script that has just scored nothing.
fn score_outcome(calls: usize, failed: usize) -> Result<()> {
    if calls > 0 && failed == calls {
        bail!(
            "every judge call failed ({failed} of {calls}) — nothing was scored. Check the \
             API key, the network, and the --model id, then re-run."
        );
    }
    Ok(())
}

/// What a job produced.
enum Step {
    /// Record this judgement against every subject of the job.
    Judged(judge::Judgement),
    /// This job failed. Nothing is recorded for it and the run carries on.
    Skip,
    /// The `--limit` budget is spent.
    Stop,
}

/// The running totals of one `score` invocation.
struct Run<'a> {
    judge: Option<&'a Judge>,
    cap: usize,
    total: usize,
    calls: usize,
    failed: usize,
    written: usize,
    reused: usize,
    spent: judge::Usage,
}

impl Run<'_> {
    /// The judgement for one job: the one the store already holds for these bytes, or a
    /// fresh call.
    fn step<T>(&mut self, job: &Job<T>, label: &str, subject: &str) -> Step {
        if let Some(j) = &job.reuse {
            self.reused += job.subjects.len();
            return Step::Judged(j.clone());
        }
        if self.calls >= self.cap {
            return Step::Stop;
        }
        // `total > 0` is what built the judge, and `total` counts exactly the jobs that
        // reach this line — so this is unreachable rather than a silent no-op.
        let Some(judge) = self.judge else {
            return Step::Stop;
        };
        eprint!(
            "\r  scoring {label} {}/{}…",
            self.calls + 1,
            self.total.min(self.cap)
        );
        self.calls += 1;
        match judge_one(judge, &job.prompt) {
            Ok((j, usage)) => {
                self.spent.prompt_tokens += usage.prompt_tokens;
                self.spent.completion_tokens += usage.completion_tokens;
                Step::Judged(j)
            }
            Err(e) => {
                self.failed += 1;
                eprintln!("\n  ! {label} {subject} failed: {e}");
                Step::Skip
            }
        }
    }
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

    fn a_judgement() -> judge::Judgement {
        judge::Judgement {
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
        }
    }

    /// Record a judgement for everything in a plan, exactly the way `cmd_score` does —
    /// same identities, same timestamps, one judgement fanned out over every subject of a
    /// job. Tests that reach into the store by hand instead would not prove the caching
    /// they claim to.
    fn pay_for(store: &Store, plan: &Plan, judge_model: &str) {
        for job in &plan.turns {
            for turn in &job.subjects {
                let rec = store::score_record(
                    &turn.turn_id,
                    &turn.session_id,
                    Tier::Turn,
                    &job.hash,
                    judge_model,
                    judge::REGIME,
                    a_judgement(),
                );
                store
                    .insert_score_at(&rec, &turn.source, turn.model.as_deref(), turn.timestamp)
                    .unwrap();
            }
        }
        for job in &plan.rollups {
            for session in &job.subjects {
                let rec = store::score_record(
                    &format!("{}:rollup", session.session_id),
                    &session.session_id,
                    Tier::Rollup,
                    &job.hash,
                    judge_model,
                    judge::REGIME,
                    a_judgement(),
                );
                store
                    .insert_score_as(
                        &rec,
                        &rollup_identity_of(session),
                        &session.source,
                        None,
                        rollup_timestamp(session),
                    )
                    .unwrap();
            }
        }
    }

    fn turn_ids_in(plan: &Plan) -> Vec<&str> {
        plan.turns
            .iter()
            .flat_map(|j| j.subjects.iter().map(|t| t.turn_id.as_str()))
            .collect()
    }

    fn rollup_ids_in(plan: &Plan) -> Vec<&str> {
        plan.rollups
            .iter()
            .flat_map(|j| j.subjects.iter().map(|s| s.session_id.as_str()))
            .collect()
    }

    fn turn_hash(plan: &Plan, turn_id: &str) -> String {
        plan.turns
            .iter()
            .find(|j| j.subjects.iter().any(|t| t.turn_id == turn_id))
            .map(|j| j.hash.clone())
            .unwrap()
    }

    fn rollup_hash(plan: &Plan, session_id: &str) -> String {
        plan.rollups
            .iter()
            .find(|j| j.subjects.iter().any(|s| s.session_id == session_id))
            .map(|j| j.hash.clone())
            .unwrap()
    }

    fn exchange(session: &str, n: &str, prompt: &str, reply: &str, at: &str) -> Vec<Record> {
        let at = ts(at);
        vec![
            Record::new(
                "claude-code",
                session,
                format!("u{n}"),
                Role::User,
                prompt,
                at,
            ),
            Record::new(
                "claude-code",
                session,
                format!("a{n}"),
                Role::Assistant,
                reply,
                at + Duration::minutes(1),
            ),
        ]
    }

    // ---- F2: identical turns in different sessions --------------------------

    /// Two sessions containing the same exchange assemble to the same bytes. One judge
    /// call is right; one score row is not — the second turn would be billed for and then
    /// be invisible to every report.
    #[test]
    fn identical_turns_in_two_sessions_are_one_call_and_two_scores() {
        let mut store = Store::open_in_memory().unwrap();
        let mut recs = exchange(
            "sA",
            "1",
            "thanks",
            "You're welcome!",
            "2026-01-01T00:00:00Z",
        );
        recs.extend(exchange(
            "sB",
            "2",
            "thanks",
            "You're welcome!",
            "2026-03-01T00:00:00Z",
        ));
        store.upsert_records(&recs).unwrap();

        let plan = build_plan(&store, "m", 6, "turn", None).unwrap();
        assert_eq!(
            plan.turns.len(),
            1,
            "byte-identical prompts must collapse to a single judge call"
        );
        assert_eq!(plan.calls(), 1);
        assert_eq!(
            plan.rows(),
            2,
            "…and still produce a score for each of the two turns"
        );
        assert_eq!(turn_ids_in(&plan), ["a1", "a2"]);

        pay_for(&store, &plan, "m");

        let scored = store.scores(&Filter::default()).unwrap();
        let ids: Vec<&str> = scored.iter().map(|s| s.record.turn_id.as_str()).collect();
        assert_eq!(
            ids,
            ["a1", "a2"],
            "both turns must reach the report, not just the first one seen"
        );
        assert_eq!(
            scored[0].record.content_hash, scored[1].record.content_hash,
            "they share a content hash — that is what made them one call"
        );

        // And a second run must not re-judge either of them.
        assert_eq!(build_plan(&store, "m", 6, "turn", None).unwrap().calls(), 0);
    }

    /// The same exchange appearing later, after the first was already paid for, costs
    /// nothing: the judgement is already in the store.
    #[test]
    fn a_repeat_of_an_already_judged_exchange_costs_no_call() {
        let mut store = Store::open_in_memory().unwrap();
        store
            .upsert_records(&exchange(
                "sA",
                "1",
                "thanks",
                "You're welcome!",
                "2026-01-01T00:00:00Z",
            ))
            .unwrap();
        let plan = build_plan(&store, "m", 6, "turn", None).unwrap();
        pay_for(&store, &plan, "m");

        store
            .upsert_records(&exchange(
                "sB",
                "2",
                "thanks",
                "You're welcome!",
                "2026-03-01T00:00:00Z",
            ))
            .unwrap();
        let plan = build_plan(&store, "m", 6, "turn", None).unwrap();
        assert_eq!(
            turn_ids_in(&plan),
            ["a2"],
            "the new turn still needs a score"
        );
        assert_eq!(
            plan.calls(),
            0,
            "but not a judge call — these exact bytes have already been paid for"
        );
        assert_eq!(plan.rows(), 1);

        pay_for(&store, &plan, "m");
        assert_eq!(store.scores(&Filter::default()).unwrap().len(), 2);
    }

    // ---- F3: rollups supersede ----------------------------------------------

    /// Score, ingest more of the same session, re-score. The session must end up with one
    /// rollup row, judged on the complete arc — not one per generation, averaged together.
    #[test]
    fn re_scoring_a_grown_session_leaves_one_rollup_carrying_the_latest_arc() {
        let mut store = Store::open_in_memory().unwrap();
        store
            .upsert_records(&exchange(
                "sA",
                "1",
                "first question",
                "first answer",
                "2026-01-01T00:00:00Z",
            ))
            .unwrap();
        let first = build_plan(&store, "m", 6, "rollup", None).unwrap();
        assert_eq!(first.calls(), 1);
        pay_for(&store, &first, "m");
        let first_hash = rollup_hash(&first, "sA");

        // More of the same conversation arrives.
        store
            .upsert_records(&exchange(
                "sA",
                "2",
                "second question",
                "second answer",
                "2026-01-01T00:30:00Z",
            ))
            .unwrap();
        let second = build_plan(&store, "m", 6, "rollup", None).unwrap();
        assert_eq!(
            second.calls(),
            1,
            "the arc grew, so it genuinely needs re-judging"
        );
        assert_ne!(rollup_hash(&second, "sA"), first_hash);
        pay_for(&store, &second, "m");

        let rollups = store
            .scores(&Filter {
                tier: Some(Tier::Rollup),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(
            rollups.len(),
            1,
            "one rollup per session — the stub arc must not linger and be averaged in"
        );
        assert_eq!(
            rollups[0].record.content_hash,
            rollup_hash(&second, "sA"),
            "the surviving row must be the one judged on the complete arc"
        );
    }

    /// An idle gap renames `sA` to `sA#1`. The unchanged chunk must neither be re-billed
    /// nor leave its pre-rename rollup behind as a row nothing can supersede.
    #[test]
    fn an_idle_gap_split_neither_rebills_nor_orphans_the_first_chunk() {
        let mut store = Store::open_in_memory().unwrap();
        store
            .upsert_records(&exchange(
                "sA",
                "1",
                "first question",
                "first answer",
                "2026-01-01T00:00:00Z",
            ))
            .unwrap();
        let before = build_plan(&store, "m", 6, "rollup", None).unwrap();
        assert_eq!(rollup_ids_in(&before), ["sA"]);
        pay_for(&store, &before, "m");

        // A week later the same thread resumes: `sessionize` now splits it in two, and
        // the first chunk is renumbered `sA#1`.
        store
            .upsert_records(&exchange(
                "sA",
                "2",
                "much later question",
                "much later answer",
                "2026-01-08T00:00:00Z",
            ))
            .unwrap();
        let after = build_plan(&store, "m", 6, "rollup", None).unwrap();
        assert_eq!(rollup_ids_in(&after), ["sA#2"], "only the new chunk is new");
        pay_for(&store, &after, "m");

        let rollups = store
            .scores(&Filter {
                tier: Some(Tier::Rollup),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(
            rollups.len(),
            2,
            "two chunks, two rollups — the renamed one must not become a third"
        );
    }

    // ---- F18: a rollup survives the window that selected it -------------------

    /// `score --since T` rolls a straddling session up whole. `report --since T` must then
    /// still show it: paying for a rollup the same window hides is a broken round trip.
    #[test]
    fn a_rollup_survives_the_since_window_that_paid_for_it() {
        let store = straddling_store();
        let cutoff = ts("2026-01-01T01:00:00Z");

        let plan = build_plan(&store, "m", 6, "both", Some(cutoff)).unwrap();
        assert_eq!(rollup_ids_in(&plan), ["s1"]);
        pay_for(&store, &plan, "m");

        let visible = store
            .scores(&Filter {
                since: Some(cutoff),
                tier: Some(Tier::Rollup),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(
            rollup_ids_in(&plan).len(),
            visible.len(),
            "every rollup the window billed for must be visible through the same window"
        );
        assert_eq!(visible[0].record.session_id, "s1");
    }

    // ---- F23: a run that scored nothing is not a success ----------------------

    #[test]
    fn a_run_where_every_call_failed_is_an_error() {
        // A bad API key fails every call. Exiting 0 there tells a script the corpus is
        // scored when not one score was written.
        assert!(score_outcome(4, 4).is_err());
        assert!(
            score_outcome(4, 1).is_ok(),
            "a partial failure still scored"
        );
        assert!(
            score_outcome(0, 0).is_ok(),
            "an all-cache run made no calls"
        );
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

        // Pay for both, then the plan must be empty.
        pay_for(&store, &plan, "openrouter/m");

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
        pay_for(&store, &plan, "openrouter/model-a");

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
        let turn_ids = turn_ids_in(&windowed);
        assert_eq!(turn_ids, ["a2"]);

        let rollup_ids = rollup_ids_in(&windowed);
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

        assert_eq!(turn_hash(&all, "a2"), turn_hash(&windowed, "a2"));
        assert_eq!(rollup_hash(&all, "s1"), rollup_hash(&windowed, "s1"));

        // Pay for the narrow window, then widen it: the overlap must not come back.
        pay_for(&store, &windowed, "m");

        let widened = build_plan(&store, "m", 6, "both", None).unwrap();
        assert!(!turn_ids_in(&widened).contains(&"a2"));
        assert!(!rollup_ids_in(&widened).contains(&"s1"));
    }
}
