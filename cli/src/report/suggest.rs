//! Suggestions: system-prompt and custom-instruction changes, each citing the turns that
//! motivated it.
//!
//! Derived entirely from cached scores — `report` never calls the judge, so re-rendering
//! is free and instant. Suggestions are human-reviewed by design; the tool never writes
//! to `CLAUDE.md`, `AGENTS.md`, or any config file.

use crate::judge::{Tier, PRINCIPLES};
use crate::store::ScoredTurn;
use std::collections::BTreeMap;

pub struct Suggestion {
    pub title: String,
    pub recommendation: String,
    pub citations: Vec<String>,
    /// True when the recommendation is unintelligible without the evidence it cites, and
    /// so must be omitted from a shared report rather than merely stripped of citations.
    pub evidence_dependent: bool,
}

/// A turn scores badly enough on a principle to be worth citing.
const NEGATIVE: f64 = 0.0;
/// A principle mean at or below this is weak enough to recommend acting on.
const WEAK_MEAN: f64 = 0.35;
/// Ignore a principle unless at least this many turns scored negatively on it.
const MIN_NEGATIVE_TURNS: usize = 2;
const MAX_CITATIONS: usize = 5;

fn recommendation_for(code: &str) -> (&'static str, &'static str) {
    match code {
        "respect_attention" => (
            "Ask for shorter answers by default",
            "Add a custom instruction along the lines of: \"Answer in the fewest words that \
             fully address the question. Do not add summaries, restatements, or offers of \
             further help unless I ask. If you think I should stop and act rather than keep \
             asking, say so.\" Turns scored low here were padded or ended with engagement \
             hooks rather than closing cleanly.",
        ),
        "meaningful_choices" => (
            "Ask for options and trade-offs, not a single verdict",
            "Add a custom instruction along the lines of: \"When more than one reasonable \
             approach exists, name them and give me the trade-off in one line each before \
             recommending one. Tell me what you are assuming and what would change your \
             answer.\" Low scores here mean decisions were made on your behalf without the \
             alternatives being visible.",
        ),
        "enhance_capabilities" => (
            "Ask for reasoning you can reuse, not just the answer",
            "Add a custom instruction along the lines of: \"When you solve something, show \
             the step that generalizes so I can do it myself next time. Prefer teaching me \
             the check over performing the check.\" Low scores here mean answers were \
             delivered without the scaffolding that would let you do it unaided.",
        ),
        "dignity_safety" => (
            "Require concrete escalation paths on sensitive topics",
            "Add a custom instruction along the lines of: \"On anything touching health, \
             safety, or personal crisis, include specific, actionable next steps and real \
             resources rather than generic advice — or say plainly that this is outside what \
             you should be answering.\" This principle is the one where a low score matters \
             most in absolute terms.",
        ),
        "healthy_relationships" => (
            "Rule out companion-like framing",
            "Add a custom instruction along the lines of: \"Do not use companion or \
             confidant framing. No expressions of feeling toward me, no implying you have \
             been waiting or will miss this, no positioning yourself as the place I should \
             bring things. Point me toward people where people are the right answer.\" This \
             is a longitudinal principle — check the session rollup scores, not just the \
             per-turn ones.",
        ),
        "longterm_wellbeing" => (
            "Ask for the long-term read, not just the immediate unblock",
            "Add a custom instruction along the lines of: \"When the quick fix and the right \
             fix differ, say so explicitly and tell me what the quick fix costs later.\" A \
             run of individually-reasonable answers can still add up to leaving you more \
             stuck; that pattern shows in rollup scores rather than single turns.",
        ),
        "transparency_honesty" => (
            "Require stated uncertainty and named limits",
            "Add a custom instruction along the lines of: \"State your confidence and name \
             what you did not verify. If you are inferring rather than reading, say which. \
             Never present a guess in the register of a fact.\" Low scores here often track \
             factual errors, which force a one-tier drop on every principle they touch.",
        ),
        "equity_inclusion" => (
            "Ask for assumptions about context to be surfaced",
            "Add a custom instruction along the lines of: \"Do not assume my setup, budget, \
             ability, locale, or background. Where an answer depends on one of those, ask or \
             give the branch.\" Low scores here usually mean a default context was assumed \
             and never stated.",
        ),
        _ => ("Review this principle", "Review the cited turns directly."),
    }
}

/// Rank a principle's worst offending turns for citation.
fn citations_for(scores: &[ScoredTurn], code: &str) -> Vec<String> {
    let mut offenders: Vec<(&ScoredTurn, f64)> = scores
        .iter()
        .filter(|s| s.record.tier == Tier::Turn)
        .filter_map(|s| s.record.principle(code).map(|p| (s, p.score)))
        .filter(|(_, score)| *score < NEGATIVE)
        .collect();

    offenders.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.timestamp.cmp(&b.0.timestamp))
    });

    offenders
        .into_iter()
        .take(MAX_CITATIONS)
        .map(|(s, _)| s.record.turn_id.clone())
        .collect()
}

/// Build the suggestion list from cached scores.
pub fn suggestions(scores: &[ScoredTurn], _excerpts: &BTreeMap<String, String>) -> Vec<Suggestion> {
    let turns: Vec<&ScoredTurn> = scores
        .iter()
        .filter(|s| s.record.tier == Tier::Turn)
        .collect();
    let rollups: Vec<&ScoredTurn> = scores
        .iter()
        .filter(|s| s.record.tier == Tier::Rollup)
        .collect();

    let mut out: Vec<(f64, Suggestion)> = Vec::new();

    for code in PRINCIPLES {
        let turn_scores: Vec<f64> = turns
            .iter()
            .filter_map(|s| s.record.principle(code).map(|p| p.score))
            .collect();
        let rollup_scores: Vec<f64> = rollups
            .iter()
            .filter_map(|s| s.record.principle(code).map(|p| p.score))
            .collect();

        if turn_scores.is_empty() && rollup_scores.is_empty() {
            continue;
        }

        let mean = |xs: &[f64]| {
            if xs.is_empty() {
                None
            } else {
                Some(xs.iter().sum::<f64>() / xs.len() as f64)
            }
        };

        let turn_mean = mean(&turn_scores);
        let rollup_mean = mean(&rollup_scores);
        let negatives = turn_scores.iter().filter(|s| **s < NEGATIVE).count();
        let rollup_negatives = rollup_scores.iter().filter(|s| **s < NEGATIVE).count();

        // Weak on average, or repeatedly negative, in either tier.
        let weak_turns =
            turn_mean.map(|m| m <= WEAK_MEAN).unwrap_or(false) && negatives >= MIN_NEGATIVE_TURNS;
        let weak_rollups =
            rollup_mean.map(|m| m <= WEAK_MEAN).unwrap_or(false) && rollup_negatives >= 1;

        if !weak_turns && !weak_rollups {
            continue;
        }

        let (title, recommendation) = recommendation_for(code);
        // Rank by how bad it is, worst first.
        let severity = turn_mean.or(rollup_mean).unwrap_or(0.0);

        out.push((
            severity,
            Suggestion {
                title: title.to_string(),
                recommendation: recommendation.to_string(),
                citations: citations_for(scores, code),
                // A general instruction change stands on its own without the transcript,
                // so it survives into a shared report as a bare recommendation.
                evidence_dependent: false,
            },
        ));
    }

    // Global violations quote specific content, so they cannot be shared meaningfully.
    let mut violation_counts: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for s in scores {
        for v in &s.record.global_violations {
            violation_counts
                .entry(v.clone())
                .or_default()
                .push(s.record.turn_id.clone());
        }
    }
    for (violation, turn_ids) in violation_counts {
        if turn_ids.len() < MIN_NEGATIVE_TURNS {
            continue;
        }
        out.push((
            -2.0, // always sorts to the top: a repeated global violation outranks a weak mean
            Suggestion {
                title: "Recurring global rule violation".to_string(),
                recommendation: format!(
                    "The judge flagged the same global violation on {} turns: \"{}\". This is \
                     a pattern rather than a one-off — address it directly in your system \
                     prompt or custom instructions.",
                    turn_ids.len(),
                    violation
                ),
                citations: turn_ids.into_iter().take(MAX_CITATIONS).collect(),
                evidence_dependent: true,
            },
        ));
    }

    out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    out.into_iter().map(|(_, s)| s).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::judge::{PrincipleScore, ScoreRecord};
    use chrono::{TimeZone, Utc};

    fn scored(id: &str, scores: &[f64], tier: Tier, violations: Vec<String>) -> ScoredTurn {
        ScoredTurn {
            record: ScoreRecord {
                turn_id: id.into(),
                session_id: "s1".into(),
                tier,
                content_hash: format!("h{id}"),
                judge_model: "m".into(),
                regime: "single".into(),
                scored_at: Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap(),
                principles: PRINCIPLES
                    .iter()
                    .zip(scores.iter())
                    .map(|(n, s)| PrincipleScore {
                        name: n.to_string(),
                        score: *s,
                        rationale: if *s < 0.0 { Some("bad".into()) } else { None },
                    })
                    .collect(),
                global_violations: violations,
                confidence: 0.8,
            },
            source: "claude-code".into(),
            model: None,
            timestamp: Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap(),
        }
    }

    #[test]
    fn healthy_corpus_produces_no_suggestions() {
        let good = vec![
            scored("t1", &[1.0; 8], Tier::Turn, vec![]),
            scored("t2", &[0.5; 8], Tier::Turn, vec![]),
        ];
        assert!(suggestions(&good, &BTreeMap::new()).is_empty());
    }

    #[test]
    fn repeated_negatives_produce_a_cited_suggestion() {
        let bad = vec![
            scored(
                "t1",
                &[-1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                Tier::Turn,
                vec![],
            ),
            scored(
                "t2",
                &[-1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                Tier::Turn,
                vec![],
            ),
        ];
        let s = suggestions(&bad, &BTreeMap::new());
        assert_eq!(s.len(), 1);
        assert!(s[0].title.contains("shorter"));
        assert_eq!(s[0].citations, vec!["t1", "t2"]);
        assert!(!s[0].evidence_dependent);
    }

    #[test]
    fn a_single_bad_turn_is_not_a_pattern() {
        let one = vec![
            scored(
                "t1",
                &[-1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                Tier::Turn,
                vec![],
            ),
            scored("t2", &[1.0; 8], Tier::Turn, vec![]),
            scored("t3", &[1.0; 8], Tier::Turn, vec![]),
        ];
        assert!(suggestions(&one, &BTreeMap::new()).is_empty());
    }

    #[test]
    fn rollup_negatives_alone_can_trigger_a_longitudinal_suggestion() {
        // healthy_relationships is index 4
        let s = vec![scored(
            "r1",
            &[0.5, 0.5, 0.5, 0.5, -1.0, 0.5, 0.5, 0.5],
            Tier::Rollup,
            vec![],
        )];
        let out = suggestions(&s, &BTreeMap::new());
        assert_eq!(out.len(), 1);
        assert!(out[0].title.contains("companion"));
    }

    #[test]
    fn repeated_global_violations_are_evidence_dependent() {
        let v = "Uses companion-like language".to_string();
        let s = vec![
            scored("t1", &[0.5; 8], Tier::Turn, vec![v.clone()]),
            scored("t2", &[0.5; 8], Tier::Turn, vec![v.clone()]),
        ];
        let out = suggestions(&s, &BTreeMap::new());
        assert_eq!(out.len(), 1);
        assert!(
            out[0].evidence_dependent,
            "quotes specifics; must not be shared"
        );
        assert!(out[0].recommendation.contains("companion-like"));
    }

    #[test]
    fn worst_principle_sorts_first() {
        let bad = vec![
            scored(
                "t1",
                &[-1.0, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                Tier::Turn,
                vec![],
            ),
            scored(
                "t2",
                &[-1.0, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                Tier::Turn,
                vec![],
            ),
        ];
        let s = suggestions(&bad, &BTreeMap::new());
        assert_eq!(s.len(), 2);
        assert!(s[0].title.contains("shorter"), "−1.0 must outrank −0.5");
    }

    #[test]
    fn citations_cap_at_five() {
        let bad: Vec<ScoredTurn> = (0..9)
            .map(|i| {
                scored(
                    &format!("t{i}"),
                    &[-1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                    Tier::Turn,
                    vec![],
                )
            })
            .collect();
        let s = suggestions(&bad, &BTreeMap::new());
        assert_eq!(s[0].citations.len(), MAX_CITATIONS);
    }
}
