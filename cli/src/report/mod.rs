//! HTML report rendering — full and `--share`.
//!
//! Never calls the judge, so it is free and instant to re-render. Everything here is
//! derived from cached scores.
//!
//! The two modes are two artifacts rather than one redacted one. Verbatim excerpts are
//! what make a low score believable to yourself, so the default keeps them; sharing is a
//! separate, deliberate command that drops excerpts *and* judge reasoning (reasoning
//! restates what the turn said).

mod suggest;

pub use suggest::{suggestions, Suggestion};

use crate::judge::{Tier, PRINCIPLES};
use crate::store::ScoredTurn;
use chrono::{DateTime, Datelike, Duration, Utc};
use std::collections::BTreeMap;

/// Human-readable principle names.
pub fn principle_label(code: &str) -> &'static str {
    match code {
        "respect_attention" => "Respect User Attention",
        "meaningful_choices" => "Enable Meaningful Choices",
        "enhance_capabilities" => "Enhance Human Capabilities",
        "dignity_safety" => "Protect Dignity & Safety",
        "healthy_relationships" => "Foster Healthy Relationships",
        "longterm_wellbeing" => "Prioritize Long-Term Wellbeing",
        "transparency_honesty" => "Be Transparent & Honest",
        "equity_inclusion" => "Design for Equity & Inclusion",
        _ => "Unknown Principle",
    }
}

/// Which principles a single turn cannot show. Surfaced in the report so a turn-only
/// corpus doesn't read as if all eight were measured equally.
pub const LONGITUDINAL: [&str; 4] = [
    "healthy_relationships",
    "longterm_wellbeing",
    "meaningful_choices",
    "respect_attention",
];

pub struct ReportInput {
    pub scores: Vec<ScoredTurn>,
    pub excerpts: BTreeMap<String, String>,
    pub judge_models: Vec<String>,
    pub regimes: Vec<String>,
    pub sources: Vec<String>,
    pub generated_at: DateTime<Utc>,
    pub discarded_branches: usize,
    pub filter_note: Option<String>,
    pub unverified_sources: Vec<String>,
}

pub struct Aggregates {
    pub turn_count: usize,
    pub rollup_count: usize,
    pub turn_overall: Option<f64>,
    pub rollup_overall: Option<f64>,
    pub turn_by_principle: BTreeMap<String, f64>,
    pub rollup_by_principle: BTreeMap<String, f64>,
    pub mean_confidence: Option<f64>,
    pub span: Option<(DateTime<Utc>, DateTime<Utc>)>,
}

fn mean(xs: &[f64]) -> Option<f64> {
    if xs.is_empty() {
        None
    } else {
        Some(xs.iter().sum::<f64>() / xs.len() as f64)
    }
}

pub fn aggregate(scores: &[ScoredTurn]) -> Aggregates {
    let turns: Vec<&ScoredTurn> = scores
        .iter()
        .filter(|s| s.record.tier == Tier::Turn)
        .collect();
    let rollups: Vec<&ScoredTurn> = scores
        .iter()
        .filter(|s| s.record.tier == Tier::Rollup)
        .collect();

    // Missing-principle policy, and it is the *opposite* of the Python scorer's.
    // `humanebench/scorer.py:130,136` treats a principle with no usable scores as 0 and
    // averages that 0 into the HumaneScore; the filter_map below EXCLUDES it, and `mean`
    // returns None for an empty set so the principle is omitted from the report entirely.
    // This cannot currently produce a divergence: `judge::parse_judgement` rejects any
    // judgement that does not carry exactly the eight expected principles, so a stored
    // record always has all eight. The invariant is upstream, though, and this layer would
    // diverge the moment it is relaxed. See `rubric/README.md` and the report caveats.
    let by_principle = |set: &[&ScoredTurn]| -> BTreeMap<String, f64> {
        let mut out = BTreeMap::new();
        for code in PRINCIPLES {
            let vals: Vec<f64> = set
                .iter()
                .filter_map(|s| s.record.principle(code).map(|p| p.score))
                .collect();
            if let Some(m) = mean(&vals) {
                out.insert(code.to_string(), m);
            }
        }
        out
    };

    let span = if scores.is_empty() {
        None
    } else {
        let min = scores.iter().map(|s| s.timestamp).min().unwrap();
        let max = scores.iter().map(|s| s.timestamp).max().unwrap();
        Some((min, max))
    };

    Aggregates {
        turn_count: turns.len(),
        rollup_count: rollups.len(),
        turn_overall: mean(&turns.iter().map(|s| s.record.overall()).collect::<Vec<_>>()),
        rollup_overall: mean(
            &rollups
                .iter()
                .map(|s| s.record.overall())
                .collect::<Vec<_>>(),
        ),
        turn_by_principle: by_principle(&turns),
        rollup_by_principle: by_principle(&rollups),
        mean_confidence: mean(
            &scores
                .iter()
                .map(|s| s.record.confidence)
                .collect::<Vec<_>>(),
        ),
        span,
    }
}

/// One point on a trend line.
pub struct TrendPoint {
    pub bucket: String,
    pub at: DateTime<Utc>,
    pub value: f64,
    pub n: usize,
}

/// Bucket turn-tier scores over time. Daily for short spans, weekly for long ones, so the
/// axis stays legible either way.
pub fn trend(scores: &[ScoredTurn], principle: &str) -> Vec<TrendPoint> {
    let turns: Vec<&ScoredTurn> = scores
        .iter()
        .filter(|s| s.record.tier == Tier::Turn)
        .collect();
    if turns.is_empty() {
        return Vec::new();
    }
    let min = turns.iter().map(|s| s.timestamp).min().unwrap();
    let max = turns.iter().map(|s| s.timestamp).max().unwrap();
    let weekly = (max - min) > Duration::days(60);

    let mut buckets: BTreeMap<String, (Vec<f64>, DateTime<Utc>)> = BTreeMap::new();
    for s in turns {
        let Some(p) = s.record.principle(principle) else {
            continue;
        };
        let key = if weekly {
            let iso = s.timestamp.iso_week();
            format!("{}-W{:02}", iso.year(), iso.week())
        } else {
            s.timestamp.format("%Y-%m-%d").to_string()
        };
        let entry = buckets
            .entry(key)
            .or_insert_with(|| (Vec::new(), s.timestamp));
        entry.0.push(p.score);
        if s.timestamp < entry.1 {
            entry.1 = s.timestamp;
        }
    }

    buckets
        .into_iter()
        .filter_map(|(bucket, (vals, at))| {
            mean(&vals).map(|value| TrendPoint {
                bucket,
                at,
                value,
                n: vals.len(),
            })
        })
        .collect()
}

/// A turn ranked by how badly it scored.
pub struct WorstTurn<'a> {
    pub scored: &'a ScoredTurn,
    pub overall: f64,
    pub negatives: Vec<(&'a str, f64, Option<&'a str>)>,
}

pub fn worst_turns<'a>(scores: &'a [ScoredTurn], limit: usize) -> Vec<WorstTurn<'a>> {
    let mut ranked: Vec<WorstTurn> = scores
        .iter()
        .filter(|s| s.record.tier == Tier::Turn)
        .map(|s| WorstTurn {
            scored: s,
            overall: s.record.overall(),
            negatives: s
                .record
                .principles
                .iter()
                .filter(|p| p.score < 0.0)
                .map(|p| (p.name.as_str(), p.score, p.rationale.as_deref()))
                .collect(),
        })
        .filter(|w| !w.negatives.is_empty() || w.overall < 0.5)
        .collect();

    ranked.sort_by(|a, b| {
        a.overall
            .partial_cmp(&b.overall)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.negatives.len().cmp(&a.negatives.len()))
    });
    ranked.truncate(limit);
    ranked
}

// ---- HTML -------------------------------------------------------------------

pub fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(c),
        }
    }
    out
}

/// Map a -1..1 score onto the report's diverging scale.
fn score_color(score: f64) -> &'static str {
    if score >= 0.75 {
        "#2f855a"
    } else if score >= 0.25 {
        "#68a678"
    } else if score >= -0.25 {
        "#b7952f"
    } else if score >= -0.75 {
        "#c0673a"
    } else {
        "#a33a30"
    }
}

fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let head: String = s.chars().take(max).collect();
        format!("{head}…")
    }
}

/// Horizontal bar chart of the eight principles, as inline SVG.
fn principle_bars(by_principle: &BTreeMap<String, f64>) -> String {
    let row_h = 30.0;
    let label_w = 210.0;
    let bar_w = 320.0;
    let h = row_h * PRINCIPLES.len() as f64 + 34.0;
    let width = label_w + bar_w + 60.0;
    let mid = label_w + bar_w / 2.0;

    let mut svg = format!(
        r#"<svg class="chart" viewBox="0 0 {width} {h}" role="img" aria-label="Score by principle">"#
    );
    // Axis: -1 .. 0 .. +1
    svg.push_str(&format!(
        r#"<line x1="{mid}" y1="18" x2="{mid}" y2="{}" stroke="currentColor" stroke-opacity=".25"/>"#,
        h - 16.0
    ));
    for (tick, label) in [(-1.0, "−1"), (0.0, "0"), (1.0, "+1")] {
        let x = mid + tick * (bar_w / 2.0);
        svg.push_str(&format!(
            r#"<text x="{x:.1}" y="12" class="tick" text-anchor="middle">{label}</text>"#
        ));
    }

    for (i, code) in PRINCIPLES.iter().enumerate() {
        let y = 24.0 + i as f64 * row_h;
        let label = principle_label(code);
        svg.push_str(&format!(
            r#"<text x="{}" y="{:.1}" class="lbl" text-anchor="end">{}</text>"#,
            label_w - 12.0,
            y + 14.0,
            escape(label)
        ));
        match by_principle.get(*code) {
            Some(v) => {
                let half = bar_w / 2.0;
                let len = (v.abs() * half).max(1.5);
                let x = if *v >= 0.0 { mid } else { mid - len };
                svg.push_str(&format!(
                    r#"<rect x="{x:.1}" y="{:.1}" width="{len:.1}" height="16" rx="3" fill="{}"/>"#,
                    y + 3.0,
                    score_color(*v)
                ));
                svg.push_str(&format!(
                    r#"<text x="{:.1}" y="{:.1}" class="val">{:+.2}</text>"#,
                    label_w + bar_w + 8.0,
                    y + 15.0,
                    v
                ));
            }
            None => {
                svg.push_str(&format!(
                    r#"<text x="{:.1}" y="{:.1}" class="val muted">no data</text>"#,
                    label_w + bar_w + 8.0,
                    y + 15.0
                ));
            }
        }
    }
    svg.push_str("</svg>");
    svg
}

/// Sparkline-style trend for one principle.
fn trend_chart(points: &[TrendPoint], label: &str) -> String {
    if points.len() < 2 {
        return format!(
            r#"<div class="trend"><h4>{}</h4><p class="muted small">Not enough data points to plot a trend.</p></div>"#,
            escape(label)
        );
    }
    let w = 420.0;
    let h = 90.0;
    let pad = 6.0;
    let n = points.len() as f64;

    let to_xy = |i: usize, v: f64| {
        let x = pad + (i as f64 / (n - 1.0)) * (w - 2.0 * pad);
        // score domain is fixed at -1..1 so charts are comparable across principles
        let y = pad + ((1.0 - v) / 2.0) * (h - 2.0 * pad);
        (x, y)
    };

    let mut path = String::new();
    for (i, p) in points.iter().enumerate() {
        let (x, y) = to_xy(i, p.value);
        path.push_str(&format!(
            "{}{:.1},{:.1}",
            if i == 0 { "M" } else { "L" },
            x,
            y
        ));
    }

    let (_, zero_y) = to_xy(0, 0.0);
    let mut dots = String::new();
    for (i, p) in points.iter().enumerate() {
        let (x, y) = to_xy(i, p.value);
        dots.push_str(&format!(
            r#"<circle cx="{x:.1}" cy="{y:.1}" r="2.5" fill="{}"><title>{}: {:+.2} (n={})</title></circle>"#,
            score_color(p.value),
            escape(&p.bucket),
            p.value,
            p.n
        ));
    }

    format!(
        r##"<div class="trend"><h4>{label}</h4>
<svg class="spark" viewBox="0 0 {w} {h}" role="img" aria-label="Trend for {label}">
<line x1="{pad}" y1="{zero_y:.1}" x2="{:.1}" y2="{zero_y:.1}" stroke="currentColor" stroke-opacity=".2" stroke-dasharray="3 3"/>
<path d="{path}" fill="none" stroke="#4a6fa5" stroke-width="1.8"/>
{dots}
</svg>
<p class="muted small">{} → {}</p></div>"##,
        w - pad,
        escape(&points[0].bucket),
        escape(&points[points.len() - 1].bucket),
        label = escape(label),
    )
}

const STYLE: &str = r#"
:root { color-scheme: light dark; --fg:#1a1a1a; --bg:#fdfdfc; --muted:#6b6b6b; --line:#e3e1dd; --card:#ffffff; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e8e6e3; --bg:#16161a; --muted:#9a9a9a; --line:#2c2c33; --card:#1d1d22; }
}
* { box-sizing: border-box; }
body { margin:0; padding:2.5rem 1.25rem 4rem; background:var(--bg); color:var(--fg);
  font:15px/1.6 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif; }
main { max-width: 860px; margin: 0 auto; }
h1 { font-size:1.7rem; margin:0 0 .25rem; letter-spacing:-.02em; }
h2 { font-size:1.15rem; margin:2.5rem 0 .75rem; padding-bottom:.35rem; border-bottom:1px solid var(--line); }
h3 { font-size:1rem; margin:1.5rem 0 .5rem; }
h4 { font-size:.85rem; margin:0 0 .35rem; font-weight:600; color:var(--muted); }
.sub { color:var(--muted); margin:0 0 1.5rem; font-size:.9rem; }
.muted { color:var(--muted); }
.small { font-size:.8rem; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:1rem 1.15rem; margin:.75rem 0; }
.headline { display:flex; gap:2rem; flex-wrap:wrap; align-items:baseline; }
.headline .n { font-size:2.4rem; font-weight:650; letter-spacing:-.03em; }
.chart { width:100%; height:auto; overflow:visible; }
.chart text { fill:currentColor; font-size:11px; }
.chart .lbl { font-size:11.5px; }
.chart .val { font-size:11px; font-variant-numeric:tabular-nums; }
.chart .tick { font-size:10px; opacity:.6; }
.trends { display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:1rem; }
.trend { border:1px solid var(--line); border-radius:8px; padding:.75rem .85rem; background:var(--card); }
.spark { width:100%; height:auto; }
.turn { border:1px solid var(--line); border-left:3px solid var(--line); border-radius:8px; padding:.9rem 1rem; margin:.75rem 0; background:var(--card); }
.turn .meta { font-size:.78rem; color:var(--muted); display:flex; gap:.75rem; flex-wrap:wrap; margin-bottom:.5rem; }
.pill { display:inline-block; padding:.1rem .45rem; border-radius:99px; font-size:.72rem; font-variant-numeric:tabular-nums; color:#fff; }
blockquote { margin:.6rem 0; padding:.6rem .85rem; border-left:2px solid var(--line); background:rgba(128,128,128,.06);
  border-radius:0 6px 6px 0; white-space:pre-wrap; word-break:break-word; font-size:.88rem; }
.rationale { font-size:.85rem; margin:.35rem 0 0; }
.note { border-left:3px solid #b7952f; background:rgba(183,149,47,.07); padding:.75rem 1rem; border-radius:0 8px 8px 0; margin:1rem 0; font-size:.87rem; }
.scroll { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-size:.85rem; }
th,td { text-align:left; padding:.4rem .6rem; border-bottom:1px solid var(--line); }
th { color:var(--muted); font-weight:600; }
td.num { text-align:right; font-variant-numeric:tabular-nums; }
footer { margin-top:3rem; padding-top:1rem; border-top:1px solid var(--line); color:var(--muted); font-size:.8rem; }
ul { padding-left:1.1rem; }
li { margin:.3rem 0; }
"#;

fn header(input: &ReportInput, agg: &Aggregates, share: bool) -> String {
    let span = agg
        .span
        .map(|(a, b)| format!("{} → {}", a.format("%Y-%m-%d"), b.format("%Y-%m-%d")))
        .unwrap_or_else(|| "no scored turns".to_string());

    let overall = agg
        .turn_overall
        .map(|v| format!("{v:+.2}"))
        .unwrap_or_else(|| "—".into());

    let title = if share {
        "HumaneBench — shared summary"
    } else {
        "HumaneBench — personal report"
    };

    format!(
        r#"<h1>{title}</h1>
<p class="sub">{span} · {} turns scored · {} session rollups{}</p>
<div class="card headline">
  <div><div class="n">{overall}</div><div class="muted small">overall, turn tier (−1 … +1)</div></div>
  <div><div class="n">{}</div><div class="muted small">overall, session rollups</div></div>
  <div><div class="n">{}</div><div class="muted small">mean judge confidence</div></div>
</div>"#,
        agg.turn_count,
        agg.rollup_count,
        input
            .filter_note
            .as_ref()
            .map(|f| format!(" · {}", escape(f)))
            .unwrap_or_default(),
        agg.rollup_overall
            .map(|v| format!("{v:+.2}"))
            .unwrap_or_else(|| "—".into()),
        agg.mean_confidence
            .map(|v| format!("{v:.2}"))
            .unwrap_or_else(|| "—".into()),
    )
}

fn caveats(input: &ReportInput, agg: &Aggregates) -> String {
    let judges = if input.judge_models.is_empty() {
        "unknown".to_string()
    } else {
        input.judge_models.join(", ")
    };
    let regimes = if input.regimes.is_empty() {
        "single".to_string()
    } else {
        input.regimes.join(", ")
    };

    let mut notes = vec![format!(
        "<strong>Single judge.</strong> Scores come from one model ({}) in the <code>{}</code> \
         regime, so they are noisier than the benchmark's validated ensemble. Treat individual \
         turn scores as indicative and the aggregates as the signal.",
        escape(&judges),
        escape(&regimes)
    )];

    if agg.rollup_count == 0 && agg.turn_count > 0 {
        notes.push(format!(
            "<strong>No session rollups in this view.</strong> Four principles — {} — are \
             inherently longitudinal and a per-turn judge is structurally blind to them. \
             Their turn-tier numbers above are the weakest part of this report.",
            LONGITUDINAL
                .iter()
                .map(|c| principle_label(c))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }

    if input.discarded_branches > 0 {
        notes.push(format!(
            "<strong>Alternatives dropped.</strong> {} branch record(s) — edits, regenerations, \
             and abandoned paths — were discarded during ingest. Only the conversation actually \
             seen and kept was scored.",
            input.discarded_branches
        ));
    }

    if !input.unverified_sources.is_empty() {
        notes.push(format!(
            "<strong>Unverified adapter(s):</strong> {}. The field names for these export formats \
             were never confirmed against a real archive, so their records may be incomplete.",
            escape(&input.unverified_sources.join(", "))
        ));
    }

    notes.push(
        "<strong>Not comparable to published benchmark numbers — it is a different \
         statistic.</strong> The benchmark scores each sample on the <em>one</em> principle its \
         prompt was built to stress, so a principle's mean is taken only over turns that \
         actually engage it. This report scores <em>every</em> turn on all eight and means \
         them, so each principle's denominator is dominated by turns where that principle is \
         barely in play. It also uses one judge where the benchmark ensembles across models, \
         and it drops a missing principle from the mean where the benchmark scores it 0 and \
         averages it in. Same rubric, same −1.0…+1.0 scale, incomparable numbers — the gap is \
         arithmetic, not a claim about which assistant is more humane."
            .to_string(),
    );

    notes
        .into_iter()
        .map(|n| format!(r#"<div class="note">{n}</div>"#))
        .collect::<Vec<_>>()
        .join("\n")
}

fn overview_section(agg: &Aggregates) -> String {
    let mut out = String::from("<h2>Score overview</h2>");
    out.push_str(&format!(
        r#"<div class="card"><h3>Turn tier · {} turns</h3><div class="scroll">{}</div></div>"#,
        agg.turn_count,
        principle_bars(&agg.turn_by_principle)
    ));
    if agg.rollup_count > 0 {
        out.push_str(&format!(
            r#"<div class="card"><h3>Session rollup tier · {} sessions</h3>
<p class="muted small">Kept separate on purpose: averaging a rollup together with turn scores would be meaningless.</p>
<div class="scroll">{}</div></div>"#,
            agg.rollup_count,
            principle_bars(&agg.rollup_by_principle)
        ));
    }
    out
}

fn trends_section(scores: &[ScoredTurn]) -> String {
    let charts: Vec<String> = PRINCIPLES
        .iter()
        .map(|code| trend_chart(&trend(scores, code), principle_label(code)))
        .collect();
    format!(
        r#"<h2>Per-principle trend</h2><div class="trends">{}</div>"#,
        charts.join("\n")
    )
}

fn worst_section(input: &ReportInput, limit: usize) -> String {
    let worst = worst_turns(&input.scores, limit);
    if worst.is_empty() {
        return String::from(
            "<h2>Lowest-scoring turns</h2><p class=\"muted\">No turn scored below the neutral band.</p>",
        );
    }

    let mut out = String::from("<h2>Lowest-scoring turns</h2>");
    for w in worst {
        let s = w.scored;
        let excerpt = input
            .excerpts
            .get(&s.record.turn_id)
            .map(|t| truncate_chars(t, 1200))
            .unwrap_or_else(|| "(excerpt unavailable)".to_string());

        let negatives: String = w
            .negatives
            .iter()
            .map(|(code, score, rationale)| {
                format!(
                    r#"<div class="rationale"><span class="pill" style="background:{}">{:+.1}</span>
<strong>{}</strong>{}</div>"#,
                    score_color(*score),
                    score,
                    escape(principle_label(code)),
                    rationale
                        .map(|r| format!(" — {}", escape(r)))
                        .unwrap_or_default()
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        let violations = if s.record.global_violations.is_empty() {
            String::new()
        } else {
            format!(
                r#"<p class="rationale"><strong>Global violations:</strong> {}</p>"#,
                escape(&s.record.global_violations.join("; "))
            )
        };

        out.push_str(&format!(
            r#"<div class="turn" style="border-left-color:{}">
<div class="meta"><span>{}</span><span>{}</span>{}<span>overall {:+.2}</span><span>confidence {:.2}</span></div>
{negatives}
{violations}
<blockquote>{}</blockquote>
</div>"#,
            score_color(w.overall),
            s.timestamp.format("%Y-%m-%d %H:%M UTC"),
            escape(&s.source),
            s.model
                .as_ref()
                .map(|m| format!("<span>{}</span>", escape(m)))
                .unwrap_or_default(),
            w.overall,
            s.record.confidence,
            escape(&excerpt),
        ));
    }
    out
}

fn suggestions_section(sugs: &[Suggestion], share: bool) -> String {
    let visible: Vec<&Suggestion> = if share {
        // Omitted when unintelligible without its evidence; otherwise citations stripped.
        sugs.iter().filter(|s| !s.evidence_dependent).collect()
    } else {
        sugs.iter().collect()
    };

    if visible.is_empty() {
        return String::from(
            "<h2>Suggestions</h2><p class=\"muted\">Nothing rose to the level of a recommendation.</p>",
        );
    }

    let mut out = String::from("<h2>Suggestions</h2>");
    out.push_str(
        r#"<p class="muted small">Human-reviewed by design. This tool never writes to CLAUDE.md, AGENTS.md, or any config file.</p>"#,
    );

    for s in visible {
        let citations = if share || s.citations.is_empty() {
            String::new()
        } else {
            format!(
                r#"<p class="muted small">Motivated by {} turn(s): {}</p>"#,
                s.citations.len(),
                s.citations
                    .iter()
                    .map(|c| format!("<code>{}</code>", escape(c)))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        out.push_str(&format!(
            r#"<div class="card"><h3>{}</h3><p>{}</p>{citations}</div>"#,
            escape(&s.title),
            escape(&s.recommendation)
        ));
    }
    out
}

fn footer(input: &ReportInput, share: bool) -> String {
    let mode = if share {
        "Shared summary: chat text, verbatim excerpts, and judge reasoning are excluded."
    } else {
        "Private report: contains verbatim excerpts of your conversations."
    };
    format!(
        r#"<footer><p>{mode}</p>
<p>Generated {} · sources: {} · rubric: HumaneBench v3.0 (compiled into the binary)</p></footer>"#,
        input.generated_at.format("%Y-%m-%d %H:%M UTC"),
        escape(&if input.sources.is_empty() {
            "none".to_string()
        } else {
            input.sources.join(", ")
        }),
    )
}

fn page(title: &str, body: String) -> String {
    format!(
        r#"<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{}</title>
<style>{STYLE}</style>
</head><body><main>{body}</main></body></html>"#,
        escape(title)
    )
}

/// The full, private report.
pub fn render_full(input: &ReportInput) -> String {
    let agg = aggregate(&input.scores);
    let sugs = suggestions(&input.scores, &input.excerpts);
    let body = format!(
        "{}{}{}{}{}{}",
        header(input, &agg, false),
        overview_section(&agg),
        trends_section(&input.scores),
        worst_section(input, 15),
        suggestions_section(&sugs, false),
        footer(input, false),
    ) + &caveats(input, &agg);
    page("HumaneBench — personal report", body)
}

/// The excerpt-free artifact: score overview and trend charts only.
pub fn render_share(input: &ReportInput) -> String {
    let agg = aggregate(&input.scores);
    let sugs = suggestions(&input.scores, &input.excerpts);
    let body = format!(
        "{}{}{}{}{}",
        header(input, &agg, true),
        overview_section(&agg),
        trends_section(&input.scores),
        suggestions_section(&sugs, true),
        footer(input, true),
    ) + &caveats(input, &agg);
    page("HumaneBench — shared summary", body)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::judge::{PrincipleScore, ScoreRecord};
    use chrono::TimeZone;

    fn scored(turn_id: &str, day: u32, scores: &[f64], tier: Tier) -> ScoredTurn {
        ScoredTurn {
            record: ScoreRecord {
                turn_id: turn_id.into(),
                session_id: "s1".into(),
                tier,
                content_hash: format!("blake3:{turn_id}"),
                judge_model: "openrouter/anthropic/claude-sonnet-4.5".into(),
                regime: "single".into(),
                scored_at: Utc.with_ymd_and_hms(2026, 1, day, 0, 0, 0).unwrap(),
                principles: PRINCIPLES
                    .iter()
                    .zip(scores.iter())
                    .map(|(n, s)| PrincipleScore {
                        name: n.to_string(),
                        score: *s,
                        rationale: if *s < 0.0 {
                            Some(format!("problem with {n}"))
                        } else {
                            None
                        },
                    })
                    .collect(),
                global_violations: vec![],
                confidence: 0.8,
            },
            source: "claude-code".into(),
            model: Some("claude-opus-5".into()),
            timestamp: Utc.with_ymd_and_hms(2026, 1, day, 12, 0, 0).unwrap(),
        }
    }

    fn input(scores: Vec<ScoredTurn>) -> ReportInput {
        let mut excerpts = BTreeMap::new();
        for s in &scores {
            excerpts.insert(
                s.record.turn_id.clone(),
                format!("SECRET-EXCERPT-{}", s.record.turn_id),
            );
        }
        ReportInput {
            scores,
            excerpts,
            judge_models: vec!["openrouter/anthropic/claude-sonnet-4.5".into()],
            regimes: vec!["single".into()],
            sources: vec!["claude-code".into()],
            generated_at: Utc.with_ymd_and_hms(2026, 7, 30, 0, 0, 0).unwrap(),
            discarded_branches: 3,
            filter_note: None,
            unverified_sources: vec![],
        }
    }

    fn sample() -> Vec<ScoredTurn> {
        vec![
            scored(
                "t1",
                1,
                &[1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 0.5],
                Tier::Turn,
            ),
            scored(
                "t2",
                2,
                &[-1.0, -0.5, -0.5, 0.5, -1.0, -0.5, 0.5, 0.5],
                Tier::Turn,
            ),
            scored(
                "t3",
                3,
                &[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                Tier::Turn,
            ),
            scored(
                "r1",
                3,
                &[0.5, -0.5, 0.5, 0.5, -1.0, -0.5, 0.5, 0.5],
                Tier::Rollup,
            ),
        ]
    }

    #[test]
    fn aggregates_keep_tiers_apart() {
        let agg = aggregate(&sample());
        assert_eq!(agg.turn_count, 3);
        assert_eq!(agg.rollup_count, 1);
        // turn-tier respect_attention = mean(1.0, -1.0, 0.5)
        let ra = agg.turn_by_principle.get("respect_attention").unwrap();
        assert!((ra - (1.0 - 1.0 + 0.5) / 3.0).abs() < 1e-9);
        // rollup tier is computed separately, never folded in
        assert!(
            (agg.rollup_by_principle
                .get("healthy_relationships")
                .unwrap()
                - -1.0)
                .abs()
                < 1e-9
        );
    }

    #[test]
    fn worst_turns_rank_lowest_first() {
        let s = sample();
        let w = worst_turns(&s, 10);
        assert_eq!(w[0].scored.record.turn_id, "t2");
        assert!(!w[0].negatives.is_empty());
    }

    #[test]
    fn worst_turns_exclude_rollups() {
        let s = sample();
        let w = worst_turns(&s, 10);
        assert!(w.iter().all(|x| x.scored.record.tier == Tier::Turn));
    }

    #[test]
    fn trend_buckets_by_day_for_short_spans() {
        let t = trend(&sample(), "respect_attention");
        assert_eq!(t.len(), 3);
        assert_eq!(t[0].bucket, "2026-01-01");
        assert!((t[1].value - -1.0).abs() < 1e-9);
    }

    /// Self-contained means "nothing here can cause a network fetch" — not "the string
    /// http:// never appears". Real transcripts contain URLs, and an excerpt quoting one
    /// is escaped body text, not a resource reference.
    #[test]
    fn full_report_is_self_contained_html() {
        let mut i = input(sample());
        i.excerpts.insert(
            "t2".into(),
            "see https://example.com/thing for details".into(),
        );
        let html = render_full(&i);

        assert!(html.starts_with("<!doctype html>"));
        assert!(html.contains("<style>"));
        assert!(html.contains("<svg"));

        // Nothing that fetches.
        for construct in [
            "<script", "<link", "<iframe", "@import", "url(", "src=", "href=",
        ] {
            assert!(
                !html.contains(construct),
                "report contains a network-capable construct: {construct}"
            );
        }
        // The quoted URL survives as escaped text.
        assert!(html.contains("example.com/thing"));
    }

    #[test]
    fn full_report_keeps_excerpts_and_reasoning() {
        let html = render_full(&input(sample()));
        assert!(
            html.contains("SECRET-EXCERPT-t2"),
            "full report must keep excerpts"
        );
        assert!(
            html.contains("problem with"),
            "full report must keep judge reasoning"
        );
    }

    /// The privacy-critical test.
    #[test]
    fn share_report_drops_excerpts_and_reasoning() {
        let html = render_share(&input(sample()));
        assert!(
            !html.contains("SECRET-EXCERPT"),
            "share leaked a verbatim excerpt"
        );
        assert!(
            !html.contains("problem with"),
            "share leaked judge reasoning"
        );
        assert!(
            !html.contains("Lowest-scoring turns"),
            "share leaked the worst-turn section"
        );
    }

    #[test]
    fn share_report_keeps_scores_and_trends() {
        let html = render_share(&input(sample()));
        assert!(html.contains("Score overview"));
        assert!(html.contains("Per-principle trend"));
        assert!(html.contains("<svg"));
    }

    #[test]
    fn share_report_strips_citations_from_suggestions() {
        let html = render_share(&input(sample()));
        assert!(!html.contains("Motivated by"), "share leaked citations");
    }

    #[test]
    fn report_states_single_judge_and_dropped_alternatives() {
        let html = render_full(&input(sample()));
        assert!(html.contains("Single judge"));
        assert!(html.contains("noisier"));
        assert!(html.contains("Alternatives dropped"));
        assert!(html.contains("3 branch record"));
    }

    /// The benchmark, the root rubric and this CLI's own judge prompt all say "Enable".
    /// A drifted label here silently renames a principle in every report and MCP payload.
    #[test]
    fn principle_labels_match_the_rubric_wording() {
        assert_eq!(
            principle_label("meaningful_choices"),
            "Enable Meaningful Choices"
        );
        for code in PRINCIPLES {
            assert_ne!(
                principle_label(code),
                "Unknown Principle",
                "no label for {code}"
            );
        }
    }

    /// The caveat has to give the *mechanical* reason the numbers differ, not a story about
    /// corpus composition — the arithmetic is what makes them incomparable.
    #[test]
    fn report_states_why_scores_are_not_benchmark_comparable() {
        let html = render_full(&input(sample()));
        assert!(html.contains("it is a different statistic"));
        assert!(html.contains("all eight"), "must name the denominator");
        assert!(
            html.contains("ensembles across models"),
            "must name the single-judge divergence"
        );
        assert!(
            html.contains("scores it 0 and averages it in"),
            "must name the opposite missing-data policy"
        );
    }

    #[test]
    fn report_labels_judge_model_and_regime() {
        let html = render_full(&input(sample()));
        assert!(html.contains("claude-sonnet-4.5"));
        assert!(html.contains("single"));
    }

    #[test]
    fn html_escaping_blocks_injection_from_transcript_text() {
        let mut i = input(sample());
        i.excerpts
            .insert("t2".into(), "<script>alert(1)</script>".into());
        let html = render_full(&i);
        assert!(!html.contains("<script>alert(1)</script>"));
        assert!(html.contains("&lt;script&gt;"));
    }

    #[test]
    fn empty_corpus_renders_without_panicking() {
        let html = render_full(&input(vec![]));
        assert!(html.contains("no scored turns"));
        let shared = render_share(&input(vec![]));
        assert!(shared.contains("<svg") || shared.contains("Score overview"));
    }
}
