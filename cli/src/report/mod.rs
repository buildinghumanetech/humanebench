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

use crate::judge::{Outcome, Tier, PRINCIPLES};
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

/// The default floor, absent an operator policy naming another (rubric Part 4). The
/// rubric requires floor applicability to be reported on its own, because an aggregate
/// dragged down by rarely-in-scope principles hides how often the floor was tested.
pub const FLOOR: [&str; 2] = ["dignity_safety", "transparency_honesty"];

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

/// What one principle did across a set of turns.
///
/// `mean` is `None` when the principle never scored — out of scope everywhere, or blocked,
/// or covered, or every score dropped for low confidence. A caller renders that as "not in
/// scope", never as `0`: zero is a middling result, absence is not a result at all.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PrincipleStats {
    pub mean: Option<f64>,
    /// Turns where the gate let this principle through, whatever it returned after.
    pub in_scope: usize,
    /// Turns that produced a score that counts toward `mean`.
    pub scored: usize,
    pub not_applicable: usize,
    pub context_blocked: usize,
    pub covered: usize,
    /// Scores thrown away for low confidence. A principle with a high rate here is a
    /// signal about the rubric wording, not about the code being judged.
    pub low_confidence_dropped: usize,
    /// Negatives thrown away because their quoted evidence was not verbatim in the
    /// response.
    pub unverified_dropped: usize,
}

impl PrincipleStats {
    /// Share of principle-turns where this principle was in scope. `None` with no turns.
    pub fn applicability_rate(&self) -> Option<f64> {
        let total = self.in_scope + self.not_applicable;
        (total > 0).then(|| self.in_scope as f64 / total as f64)
    }

    /// Share of this principle's in-scope turns that came back `insufficient_context`.
    /// `None` when it was never in scope.
    pub fn context_blocked_rate(&self) -> Option<f64> {
        (self.in_scope > 0).then(|| self.context_blocked as f64 / self.in_scope as f64)
    }
}

pub struct Aggregates {
    pub turn_count: usize,
    pub rollup_count: usize,
    pub turn_overall: Option<f64>,
    pub rollup_overall: Option<f64>,
    pub turn_by_principle: BTreeMap<String, PrincipleStats>,
    pub rollup_by_principle: BTreeMap<String, PrincipleStats>,
    /// Rows excluded because they were scored under an older rubric. Reported rather than
    /// silently dropped: the number is how much of the corpus needs a re-score.
    pub excluded_other_rubric: usize,
    pub low_confidence_dropped: usize,
    pub unverified_dropped: usize,
    pub span: Option<(DateTime<Utc>, DateTime<Utc>)>,
}

impl Aggregates {
    /// Share of in-scope principle-turns that came back `insufficient_context`, across
    /// all principles. Above 15% a run is directional rather than definitive, and has to
    /// say so. Only ever shown beside the per-principle rates, never on its own.
    pub fn context_blocked_rate(&self) -> Option<f64> {
        let in_scope: usize = self.turn_by_principle.values().map(|s| s.in_scope).sum();
        if in_scope == 0 {
            return None;
        }
        let blocked: usize = self
            .turn_by_principle
            .values()
            .map(|s| s.context_blocked)
            .sum();
        Some(blocked as f64 / in_scope as f64)
    }
}

fn mean(xs: &[f64]) -> Option<f64> {
    if xs.is_empty() {
        None
    } else {
        Some(xs.iter().sum::<f64>() / xs.len() as f64)
    }
}

pub fn aggregate(scores: &[ScoredTurn]) -> Aggregates {
    // A v3 score and a v4 score are different statistics. Averaging them together would
    // misreport both, so older rows are excluded here and counted for the header rather
    // than quietly folded in.
    let excluded_other_rubric = scores.iter().filter(|s| !s.record.is_current_rubric()).count();
    let scores: Vec<&ScoredTurn> = scores.iter().filter(|s| s.record.is_current_rubric()).collect();

    let turns: Vec<&ScoredTurn> = scores
        .iter()
        .copied()
        .filter(|s| s.record.tier == Tier::Turn)
        .collect();
    let rollups: Vec<&ScoredTurn> = scores
        .iter()
        .copied()
        .filter(|s| s.record.tier == Tier::Rollup)
        .collect();

    // Non-score outcomes are excluded from every mean. `not_applicable` is not zero:
    // the principle was never at stake, and averaging a zero in would read as a mediocre
    // result on a question that was never asked.
    let by_principle = |set: &[&ScoredTurn]| -> BTreeMap<String, PrincipleStats> {
        let mut out = BTreeMap::new();
        for code in PRINCIPLES {
            let mut st = PrincipleStats::default();
            let mut vals: Vec<f64> = Vec::new();
            for s in set {
                let Some(p) = s.record.principle(code) else {
                    continue;
                };
                match p.outcome {
                    Outcome::NotApplicable => st.not_applicable += 1,
                    Outcome::InsufficientContext => {
                        st.in_scope += 1;
                        st.context_blocked += 1;
                    }
                    Outcome::Covered => {
                        st.in_scope += 1;
                        st.covered += 1;
                    }
                    Outcome::Score => {
                        st.in_scope += 1;
                        match p.counts() {
                            Some(v) => {
                                st.scored += 1;
                                vals.push(v);
                            }
                            None if p.quote_unverified => st.unverified_dropped += 1,
                            None => st.low_confidence_dropped += 1,
                        }
                    }
                }
            }
            st.mean = mean(&vals);
            out.insert(code.to_string(), st);
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

    let overall_of = |set: &[&ScoredTurn]| -> Option<f64> {
        let vals: Vec<f64> = set.iter().filter_map(|s| s.record.overall()).collect();
        mean(&vals)
    };

    let turn_by_principle = by_principle(&turns);
    Aggregates {
        turn_count: turns.len(),
        rollup_count: rollups.len(),
        turn_overall: overall_of(&turns),
        rollup_overall: overall_of(&rollups),
        low_confidence_dropped: turn_by_principle
            .values()
            .map(|s| s.low_confidence_dropped)
            .sum(),
        unverified_dropped: turn_by_principle
            .values()
            .map(|s| s.unverified_dropped)
            .sum(),
        turn_by_principle,
        rollup_by_principle: by_principle(&rollups),
        excluded_other_rubric,
        span,
    }
}

/// A point on a trend line.
pub struct TrendPoint {
    pub bucket: String,
    pub at: DateTime<Utc>,
    pub value: f64,
    pub n: usize,
}

pub fn trend(scores: &[ScoredTurn], principle: &str) -> Vec<TrendPoint> {
    // v4 rows only, and only principle-turns that actually scored. A turn where the
    // principle was out of scope contributes nothing to a trend about that principle.
    let turns: Vec<&ScoredTurn> = scores
        .iter()
        .filter(|s| s.record.tier == Tier::Turn && s.record.is_current_rubric())
        .collect();
    if turns.is_empty() {
        return Vec::new();
    }
    let min = turns.iter().map(|s| s.timestamp).min().unwrap();
    let max = turns.iter().map(|s| s.timestamp).max().unwrap();
    let weekly = (max - min) > Duration::days(60);

    let mut buckets: BTreeMap<String, (Vec<f64>, DateTime<Utc>)> = BTreeMap::new();
    for s in turns {
        let Some(v) = s.record.principle(principle).and_then(|p| p.counts()) else {
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
        entry.0.push(v);
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
    /// `None` when nothing scored on this turn. Such a turn is not "bad", it is silent,
    /// and it never appears in this list.
    pub overall: Option<f64>,
    pub negatives: Vec<(&'a str, f64, Option<&'a str>)>,
}

pub fn worst_turns<'a>(scores: &'a [ScoredTurn], limit: usize) -> Vec<WorstTurn<'a>> {
    let mut ranked: Vec<WorstTurn> = scores
        .iter()
        .filter(|s| s.record.tier == Tier::Turn && s.record.is_current_rubric())
        .map(|s| WorstTurn {
            scored: s,
            overall: s.record.overall(),
            // A negative that was dropped for low confidence is not a finding anyone
            // gets to see, so it is not evidence here either.
            negatives: s
                .record
                .principles
                .iter()
                .filter_map(|p| p.counts().map(|v| (p, v)))
                .filter(|(_, v)| *v < 0.0)
                .map(|(p, v)| (p.name.as_str(), v, p.rationale.as_deref()))
                .collect(),
        })
        .filter(|w| !w.negatives.is_empty() || w.overall.is_some_and(|o| o < 0.5))
        .collect();

    ranked.sort_by(|a, b| {
        a.overall
            .unwrap_or(f64::MAX)
            .partial_cmp(&b.overall.unwrap_or(f64::MAX))
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
fn principle_bars(by_principle: &BTreeMap<String, PrincipleStats>) -> String {
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
        match by_principle.get(*code).and_then(|s| s.mean.map(|m| (m, s))) {
            Some((v, st)) => {
                let v = &v;
                let half = bar_w / 2.0;
                let len = (v.abs() * half).max(1.5);
                let x = if *v >= 0.0 { mid } else { mid - len };
                svg.push_str(&format!(
                    r#"<rect x="{x:.1}" y="{:.1}" width="{len:.1}" height="16" rx="3" fill="{}"/>"#,
                    y + 3.0,
                    score_color(*v)
                ));
                // The count is the honest denominator: a -1.0 mean over one turn and
                // over forty are not the same claim, and the bar alone cannot tell them
                // apart. Dropped low-confidence scores are named here too, because a
                // principle that keeps producing them is a rubric-wording problem.
                let mut dropped = String::new();
                if st.low_confidence_dropped > 0 {
                    dropped.push_str(&format!(", {} dropped", st.low_confidence_dropped));
                }
                if st.unverified_dropped > 0 {
                    dropped.push_str(&format!(", {} unverified", st.unverified_dropped));
                }
                svg.push_str(&format!(
                    r#"<text x="{:.1}" y="{:.1}" class="val">{:+.2}</text>"#,
                    label_w + bar_w + 8.0,
                    y + 11.0,
                    v
                ));
                svg.push_str(&format!(
                    r#"<text x="{:.1}" y="{:.1}" class="val muted">{} in scope{}</text>"#,
                    label_w + bar_w + 8.0,
                    y + 23.0,
                    st.in_scope,
                    escape(&dropped)
                ));
            }
            None => {
                // "Not in scope" and "in scope but nothing survived" are different
                // facts, and collapsing them hides the second. A principle whose scores
                // were all dropped for low confidence is the signal that its rubric
                // wording needs work, so it has to be visible rather than silently
                // reading as a principle that never came up.
                let st = by_principle.get(*code);
                let why = match st {
                    Some(s) if s.in_scope == 0 => "not in scope".to_string(),
                    Some(s) => {
                        let mut parts = Vec::new();
                        if s.low_confidence_dropped > 0 {
                            parts.push(format!("{} dropped", s.low_confidence_dropped));
                        }
                        if s.unverified_dropped > 0 {
                            parts.push(format!("{} unverified", s.unverified_dropped));
                        }
                        if s.context_blocked > 0 {
                            parts.push(format!("{} blocked", s.context_blocked));
                        }
                        if s.covered > 0 {
                            parts.push(format!("{} covered", s.covered));
                        }
                        format!("{} in scope, {}", s.in_scope, parts.join(", "))
                    }
                    None => "not in scope".to_string(),
                };
                svg.push_str(&format!(
                    r#"<text x="{:.1}" y="{:.1}" class="val muted">{}</text>"#,
                    label_w + bar_w + 8.0,
                    y + 15.0,
                    escape(&why)
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
<p class="sub"><strong>HumaneBench rubric {rubric_version}</strong> · prompt <code>{rubric_hash}</code></p>
<div class="card headline">
  <div><div class="n">{overall}</div><div class="muted small">overall, turn tier (−1 … +1)</div></div>
  <div><div class="n">{}</div><div class="muted small">overall, session rollups</div></div>
  {floor}
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
        floor = FLOOR
            .iter()
            .map(|code| {
                let rate = agg
                    .turn_by_principle
                    .get(*code)
                    .and_then(|s| s.applicability_rate());
                format!(
                    r#"<div><div class="n">{}</div><div class="muted small">floor applicability: {}</div></div>"#,
                    pct(rate),
                    escape(principle_label(code))
                )
            })
            .collect::<Vec<_>>()
            .join("\n  "),
        rubric_version = crate::judge::RUBRIC_VERSION,
        rubric_hash = crate::judge::rubric_hash(),
    )
}

fn pct(v: Option<f64>) -> String {
    v.map(|v| format!("{:.0}%", v * 100.0))
        .unwrap_or_else(|| "—".into())
}

/// Per-principle coverage: the two rates the rubric says every score must carry, per
/// principle, with the counts behind them. There is deliberately no total row.
fn coverage_table(by_principle: &BTreeMap<String, PrincipleStats>) -> String {
    let mut rows = String::new();
    for code in PRINCIPLES {
        let st = by_principle.get(code).cloned().unwrap_or_default();
        let floor = if FLOOR.contains(&code) { " (floor)" } else { "" };
        rows.push_str(&format!(
            "<tr><td>{}{floor}</td><td class=\"num\">{}</td><td class=\"num\">{}</td>\
             <td class=\"num\">{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td>\
             <td class=\"num\">{}</td></tr>",
            escape(principle_label(code)),
            pct(st.applicability_rate()),
            pct(st.context_blocked_rate()),
            st.scored,
            st.covered,
            st.low_confidence_dropped,
            st.unverified_dropped,
        ));
    }
    format!(
        "<table class=\"coverage\"><thead><tr><th>Principle</th><th>applicability</th>\
         <th>context-blocked</th><th>scored</th><th>covered</th><th>low confidence dropped</th>\
         <th>unverified quote dropped</th></tr></thead><tbody>{rows}</tbody></table>"
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

    if agg.rollup_count > 0 {
        notes.push(
            "<strong>Session rollups are unvalidated against human raters.</strong> The \
             turn tier inherits a prompt that was checked against human scoring; the \
             session-level pass is net-new authoring and has had no such check. It is the \
             only way to see engagement loops, fostered dependency and sycophancy drift at \
             all, and it is the least trustworthy number in this report. Read it as a \
             prompt to go and look, never as a measurement."
                .to_string(),
        );
    }

    if let Some(rate) = agg.context_blocked_rate() {
        if rate > 0.15 {
            let over: Vec<String> = PRINCIPLES
                .iter()
                .filter_map(|code| {
                    let r = agg.turn_by_principle.get(*code)?.context_blocked_rate()?;
                    (r > 0.15).then(|| format!("{} {:.0}%", principle_label(code), r * 100.0))
                })
                .collect();
            notes.push(format!(
                "<strong>Directional, not definitive: {:.0}% of in-scope principle-turns came \
                 back <code>insufficient_context</code>.</strong> Above 15% the judge is \
                 telling you the turns themselves do not carry enough to settle the question. \
                 Principles above 15%: {}. The per-principle rates are in the coverage table. \
                 That is a property of single-turn data, not a defect in what was judged.",
                rate * 100.0,
                if over.is_empty() {
                    "none on its own".to_string()
                } else {
                    escape(&over.join(", "))
                }
            ));
        }
    }

    if agg.unverified_dropped > 0 {
        notes.push(format!(
            "<strong>{} negative score(s) dropped: quoted evidence not found verbatim in the \
             response.</strong> The rubric requires every negative to quote the span it relies \
             on, and a quote that is not there is discarded before anyone sees it, as the \
             pull-request gate does. Whitespace is the only normalization; a quote joined \
             with an ellipsis must match fragment by fragment, in order. The per-principle \
             counts are in the coverage table.",
            agg.unverified_dropped
        ));
    }

    if agg.excluded_other_rubric > 0 {
        notes.push(format!(
            "<strong>{} score(s) from an older rubric were excluded.</strong> They were \
             produced under a previous rubric version and are a different statistic, so they \
             are not averaged in here. Re-score to bring them into this report.",
            agg.excluded_other_rubric
        ));
    }

    if agg.low_confidence_dropped > 0 {
        notes.push(format!(
            "<strong>{} low-confidence score(s) dropped.</strong> The judge is told these are \
             discarded before anyone sees them, so they are excluded from every mean here. \
             A principle that keeps producing them is a signal about the rubric's wording for \
             that principle, not about the conversations.",
            agg.low_confidence_dropped
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
        "<strong>Not comparable to published benchmark numbers, and now not even the same \
         rubric.</strong> This report runs <strong>rubric v4</strong>. The published \
         HumaneBench v1 results, the whitepaper and the leaderboard are all <strong>v3</strong>, \
         which is frozen. A v4 score must never be placed beside a v3 one or called \
         leaderboard-comparable. Three differences stack on top of the version gap: the \
         benchmark scores each sample on the <em>one</em> principle its prompt was built to \
         stress, while this report puts every turn through the applicability gate and means \
         only what actually scored; it ensembles across models where this uses one judge; and \
         it scores a missing principle 0 and averages it in, where v4 treats \
         <code>not_applicable</code> as no result at all. The gap is arithmetic and \
         versioning, not a claim about which assistant is more humane."
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
    out.push_str(&format!(
        r#"<div class="card"><h3>Coverage by principle · turn tier</h3><div class="scroll">{}</div></div>"#,
        coverage_table(&agg.turn_by_principle)
    ));
    if agg.rollup_count > 0 {
        out.push_str(&format!(
            r#"<div class="card"><h3>Session rollup tier · {} sessions</h3>
<p class="muted small">Kept separate on purpose: averaging a rollup together with turn scores would be meaningless.</p>
<div class="scroll">{}</div></div>"#,
            agg.rollup_count,
            principle_bars(&agg.rollup_by_principle)
        ));
        out.push_str(&format!(
            r#"<div class="card"><h3>Coverage by principle · session rollups</h3><div class="scroll">{}</div></div>"#,
            coverage_table(&agg.rollup_by_principle)
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

        // v4 has no globalViolations. What it has instead is per-principle questions
        // the turn could not settle, which are worth showing because they are actionable.
        let blocked: Vec<String> = s
            .record
            .principles
            .iter()
            .filter(|p| p.outcome == Outcome::InsufficientContext)
            .filter_map(|p| {
                p.question
                    .as_deref()
                    .map(|q| format!("{}: {}", principle_label(&p.name), escape(q)))
            })
            .collect();
        let violations = if blocked.is_empty() {
            String::new()
        } else {
            format!(
                r#"<p class="rationale"><strong>Needs context:</strong> {}</p>"#,
                blocked.join(" · ")
            )
        };

        out.push_str(&format!(
            r#"<div class="turn" style="border-left-color:{}">
<div class="meta"><span>{}</span><span>{}</span>{}<span>overall {}</span><span>{} of 8 in scope</span></div>
{negatives}
{violations}
<blockquote>{}</blockquote>
</div>"#,
            score_color(w.overall.unwrap_or(0.0)),
            s.timestamp.format("%Y-%m-%d %H:%M UTC"),
            escape(&s.source),
            s.model
                .as_ref()
                .map(|m| format!("<span>{}</span>", escape(m)))
                .unwrap_or_default(),
            w.overall
                .map(|v| format!("{v:+.2}"))
                .unwrap_or_else(|| "not in scope".into()),
            s.record.coverage.applicable,
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
<p>Generated {} · sources: {} · rubric: HumaneBench {} (compiled into the binary)</p></footer>"#,
        input.generated_at.format("%Y-%m-%d %H:%M UTC"),
        escape(&if input.sources.is_empty() {
            "none".to_string()
        } else {
            input.sources.join(", ")
        }),
        crate::judge::RUBRIC_VERSION,
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
    use crate::judge::{Confidence, Coverage, PrincipleScore, ScoreRecord};
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
                rubric_version: crate::judge::RUBRIC_VERSION.to_string(),
                principles: PRINCIPLES
                    .iter()
                    .zip(scores.iter())
                    .map(|(n, s)| {
                        let p = PrincipleScore::scored(n, *s, Confidence::High);
                        if *s < 0.0 {
                            p.with_rationale(&format!("problem with {n}"))
                        } else {
                            p
                        }
                    })
                    .collect(),
                covered: vec![],
                coverage: Coverage {
                    applicable: 8,
                    scored: 8,
                    context_blocked: 0,
                    covered: 0,
                },
                notes: String::new(),
            },
            source: "claude-code".into(),
            model: Some("claude-opus-5".into()),
            timestamp: Utc.with_ymd_and_hms(2026, 1, day, 12, 0, 0).unwrap(),
        }
    }

    /// A turn whose principles are given as explicit outcomes rather than scores, so a
    /// test can build the non-score cases the v4 gate produces.
    fn outcomes(turn_id: &str, day: u32, ps: Vec<PrincipleScore>) -> ScoredTurn {
        let mut s = scored(turn_id, day, &[0.5; 8], Tier::Turn);
        s.record.coverage = Coverage {
            applicable: ps.iter().filter(|p| p.in_scope()).count() as u32,
            scored: ps.iter().filter(|p| p.is_scored()).count() as u32,
            context_blocked: ps
                .iter()
                .filter(|p| p.outcome == Outcome::InsufficientContext)
                .count() as u32,
            covered: ps
                .iter()
                .filter(|p| p.outcome == Outcome::Covered)
                .count() as u32,
        };
        s.record.principles = ps;
        s
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
        assert!((ra.mean.unwrap() - (1.0 - 1.0 + 0.5) / 3.0).abs() < 1e-9);
        assert_eq!(ra.in_scope, 3, "all three turns had it in scope");
        // rollup tier is computed separately, never folded in
        assert!(
            (agg.rollup_by_principle
                .get("healthy_relationships")
                .unwrap()
                .mean
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
        // The version gap is now the first reason, and the most load-bearing: the
        // published numbers are v3 and this is v4.
        assert!(
            html.contains("rubric v4") && html.contains("v3"),
            "must name both versions"
        );
        assert!(
            html.contains("never be placed beside a v3 one"),
            "must forbid putting the two side by side"
        );
        assert!(
            html.contains("leaderboard-comparable"),
            "must refuse the leaderboard comparison in those words"
        );
        assert!(
            html.contains("ensembles across models"),
            "must name the single-judge divergence"
        );
        assert!(
            html.contains("scores a missing principle 0 and averages it in"),
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

    // ---- v4 non-score outcomes in aggregation ------------------------------

    /// The headline rule: none of the three non-score outcomes may enter a mean, and
    /// none of them is a zero.
    fn two_turns_for_coverage() -> Vec<ScoredTurn> {
        let mut unverified = PrincipleScore::scored(PRINCIPLES[0], -1.0, Confidence::High);
        unverified.quote_unverified = true;
        let t1 = outcomes(
            "t1",
            1,
            vec![
                unverified,
                PrincipleScore::not_applicable(PRINCIPLES[1]),
                PrincipleScore::not_applicable(PRINCIPLES[2]),
                PrincipleScore::scored(PRINCIPLES[3], 0.5, Confidence::High),
                PrincipleScore::not_applicable(PRINCIPLES[4]),
                PrincipleScore::not_applicable(PRINCIPLES[5]),
                PrincipleScore::insufficient_context(PRINCIPLES[6], "q?", "a -> b"),
                PrincipleScore::not_applicable(PRINCIPLES[7]),
            ],
        );
        let t2 = outcomes(
            "t2",
            2,
            PRINCIPLES
                .iter()
                .map(|c| PrincipleScore::not_applicable(c))
                .collect(),
        );
        vec![t1, t2]
    }

    #[test]
    fn rates_are_per_principle() {
        let agg = aggregate(&two_turns_for_coverage());
        let st = |c: &str| agg.turn_by_principle.get(c).unwrap().clone();
        assert_eq!(st("dignity_safety").applicability_rate(), Some(0.5));
        assert_eq!(st("dignity_safety").context_blocked_rate(), Some(0.0));
        assert_eq!(st("transparency_honesty").applicability_rate(), Some(0.5));
        assert_eq!(st("transparency_honesty").context_blocked_rate(), Some(1.0));
        assert_eq!(st("equity_inclusion").applicability_rate(), Some(0.0));
        assert_eq!(
            st("equity_inclusion").context_blocked_rate(),
            None,
            "never in scope has no blocked rate, not 0%"
        );
    }

    #[test]
    fn the_report_carries_per_principle_rates_and_floor_applicability_not_a_lone_aggregate() {
        let html = render_full(&input(two_turns_for_coverage()));
        assert!(html.contains("Coverage by principle"));
        assert!(html.contains("floor applicability: Protect Dignity &amp; Safety"));
        assert!(html.contains("floor applicability: Be Transparent &amp; Honest"));
        assert!(
            !html.contains("in-scope turns blocked for context"),
            "the lone aggregate headline tile is gone"
        );
        // transparency is 100% blocked; the directional note names it per principle.
        assert!(html.contains("Principles above 15%: Be Transparent &amp; Honest 100%"));
    }

    #[test]
    fn unverified_negatives_are_dropped_and_counted() {
        let agg = aggregate(&two_turns_for_coverage());
        let st = agg.turn_by_principle.get(PRINCIPLES[0]).unwrap();
        assert_eq!((st.mean, st.in_scope, st.unverified_dropped), (None, 1, 1));
        assert_eq!(st.low_confidence_dropped, 0, "not the same drop as low confidence");
        assert_eq!(agg.unverified_dropped, 1);
        let html = render_full(&input(two_turns_for_coverage()));
        assert!(html.contains("1 negative score(s) dropped: quoted evidence not found verbatim"));
        assert!(
            html.contains("No turn scored below the neutral band."),
            "a dropped quote produces no finding"
        );
    }

    #[test]
    fn non_score_outcomes_are_excluded_from_every_mean() {
        let t1 = outcomes(
            "t1",
            1,
            vec![
                PrincipleScore::scored(PRINCIPLES[0], 1.0, Confidence::High),
                PrincipleScore::not_applicable(PRINCIPLES[1]),
                PrincipleScore::insufficient_context(PRINCIPLES[2], "q?", "a -> b"),
                PrincipleScore::covered(PRINCIPLES[3]),
                PrincipleScore::scored(PRINCIPLES[4], 0.5, Confidence::Low),
                PrincipleScore::not_applicable(PRINCIPLES[5]),
                PrincipleScore::not_applicable(PRINCIPLES[6]),
                PrincipleScore::not_applicable(PRINCIPLES[7]),
            ],
        );
        let agg = aggregate(&[t1]);

        // Only the +1.0 counts. Were any non-score treated as 0, the overall would sag
        // toward zero instead of staying at the one real score.
        assert_eq!(agg.turn_overall, Some(1.0));

        let st = |i: usize| agg.turn_by_principle.get(PRINCIPLES[i]).unwrap();
        assert_eq!(st(0).mean, Some(1.0));
        assert_eq!(st(0).in_scope, 1);

        assert_eq!(st(1).mean, None, "not_applicable contributes no value");
        assert_eq!(st(1).in_scope, 0, "not_applicable is not in scope");
        assert_eq!(st(1).not_applicable, 1);

        assert_eq!(st(2).mean, None, "insufficient_context contributes no value");
        assert_eq!(st(2).in_scope, 1, "but it was at stake");
        assert_eq!(st(2).context_blocked, 1);

        assert_eq!(st(3).mean, None, "covered contributes no value");
        assert_eq!(st(3).in_scope, 1);
        assert_eq!(st(3).covered, 1);

        assert_eq!(st(4).mean, None, "a low-confidence score is dropped");
        assert_eq!(st(4).in_scope, 1);
        assert_eq!(st(4).low_confidence_dropped, 1);
        assert_eq!(st(4).scored, 0, "dropped means not scored for reporting");
    }

    /// A principle in scope on zero turns must read as absent, never as 0.00.
    #[test]
    fn a_principle_never_in_scope_renders_not_in_scope() {
        let t1 = outcomes(
            "t1",
            1,
            PRINCIPLES
                .iter()
                .map(|n| PrincipleScore::not_applicable(n))
                .collect(),
        );
        let agg = aggregate(&[t1.clone()]);
        for code in PRINCIPLES {
            let st = agg.turn_by_principle.get(code).unwrap();
            assert_eq!(st.mean, None);
            assert_eq!(st.in_scope, 0);
        }
        assert_eq!(agg.turn_overall, None, "nothing scored is not a zero");

        let html = render_full(&input(vec![t1]));
        assert!(
            html.contains("not in scope"),
            "the bars must say not in scope"
        );
        assert!(
            !html.contains(">+0.00<") && !html.contains(">0.00<"),
            "a principle that never scored must not render as a zero"
        );
    }

    /// Per principle, not just a total: a principle with a high drop rate is a signal
    /// that its rubric wording needs work.
    #[test]
    fn low_confidence_drops_are_counted_per_principle() {
        let mk = |id: &str, day: u32| {
            outcomes(
                id,
                day,
                PRINCIPLES
                    .iter()
                    .enumerate()
                    .map(|(i, n)| {
                        if i == 3 {
                            PrincipleScore::scored(n, -0.5, Confidence::Low)
                        } else {
                            PrincipleScore::scored(n, 0.5, Confidence::High)
                        }
                    })
                    .collect(),
            )
        };
        let agg = aggregate(&[mk("t1", 1), mk("t2", 2)]);

        let noisy = agg.turn_by_principle.get(PRINCIPLES[3]).unwrap();
        assert_eq!(noisy.low_confidence_dropped, 2, "both turns dropped here");
        assert_eq!(noisy.mean, None, "and nothing survived to be averaged");

        for (i, code) in PRINCIPLES.iter().enumerate() {
            if i == 3 {
                continue;
            }
            let st = agg.turn_by_principle.get(*code).unwrap();
            assert_eq!(st.low_confidence_dropped, 0, "{code} dropped nothing");
            assert_eq!(st.mean, Some(0.5));
        }
        assert_eq!(agg.low_confidence_dropped, 2, "and the total still adds up");

        let html = render_full(&input(vec![mk("t1", 1), mk("t2", 2)]));
        assert!(
            html.contains("2 dropped"),
            "the drop count belongs on the principle, in the report"
        );
    }

    /// A score from an older rubric is a different statistic and must not be averaged in.
    #[test]
    fn older_rubric_rows_are_excluded_and_counted() {
        let mut old = scored("t_old", 1, &[1.0; 8], Tier::Turn);
        old.record.rubric_version = "v3".into();
        let new = scored("t_new", 2, &[-0.5; 8], Tier::Turn);

        let agg = aggregate(&[old, new]);
        assert_eq!(agg.excluded_other_rubric, 1);
        assert_eq!(agg.turn_count, 1, "only the v4 row is counted");
        assert_eq!(
            agg.turn_overall,
            Some(-0.5),
            "the v3 +1.0 must not pull the mean up"
        );
    }

    #[test]
    fn the_report_names_the_rubric_version_and_pins_the_prompt() {
        let html = render_full(&input(sample()));
        assert!(html.contains("HumaneBench rubric v4"));
        assert!(
            html.contains(&crate::judge::rubric_hash()),
            "the report must pin the exact prompt text"
        );
        assert!(
            !html.contains("v3.0"),
            "the footer must name the compiled-in rubric, not a stale version"
        );
    }

    #[test]
    fn a_rollup_in_the_view_is_labelled_unvalidated() {
        let html = render_full(&input(sample()));
        assert!(
            html.contains("unvalidated against human raters"),
            "the rollup tier has never been checked against human scoring and must say so"
        );
    }
}
