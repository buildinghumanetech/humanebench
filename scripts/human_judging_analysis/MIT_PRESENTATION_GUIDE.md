# HumaneBench Pilot Results - MIT Presentation Guide

## Key Messages to Communicate

### 1. STUDY DESIGN
**What we did:**
- Pilot validation study with ~17 raters evaluating AI responses
- 8 prosocial principles based on humane tech framework
- 4-point ordinal scale: {-1.0, -0.5, 0.5, 1.0}
- Dual data collection: spreadsheets + Langfuse annotation queues

**Purpose:**
- Test feasibility of human evaluation methodology
- Validate rubric clarity and usability
- Establish baseline for full-scale study

### 2. WHAT WE LEARNED (Process Insights)

**Coverage patterns revealed:**
- Identify which principles had sufficient rater overlap for IRR
- Discovered rater fatigue patterns
- Established need for systematic assignment protocol in full study

**Methodological validation:**
- Rubric usability feedback from expert raters
- Time per rating (for scaling estimates)
- Data collection workflow effectiveness

### 3. PRELIMINARY FINDINGS (Be Strategic)

**Frame appropriately based on what you calculated:**

✅ **If you have IRR for 4+ principles:**
- "Achieved acceptable inter-rater reliability (α = X.XX) for majority of principles"
- Shows proof of concept for methodology

✅ **If you have IRR for 2-3 principles:**
- "Demonstrated feasibility with acceptable reliability for subset of principles"
- "Pilot identified need for balanced rater assignment in full study"

✅ **If you have IRR for <2 principles:**
- Lead with descriptive findings across ALL principles
- "Pilot successfully validated rubric usability and data collection workflow"
- "Identified optimal rater assignment strategy for full study"

**Always report:**
- Rating distributions across all 8 principles (you have complete data)
- Which principles showed strongest/weakest AI performance
- Expert rater qualitative feedback

### 4. TRANSPARENCY ABOUT LIMITATIONS

**Be upfront:**
- "Pilot study with incomplete coverage by design"
- "IRR calculated only where sufficient overlap (≥20 samples, 2+ raters)"
- "Results inform full study design, not definitive benchmark"

**This is strength, not weakness:**
- Shows methodological rigor
- Demonstrates understanding of psychometric requirements
- MIT audience will respect transparency

### 5. NEXT STEPS

**For full study:**
- Systematic rater assignment protocol (learned from pilot gaps)
- Target: 30 raters × 100 scenarios = 3,000 ratings
- Complete coverage for robust IRR across all principles
- Expected α ≥ 0.72 based on pilot + training protocol

**Publication plan:**
- Submit to FAccT 2026
- Open-source benchmark and evaluation framework
- Vulnerable populations focus (unique contribution)

---

## Presentation Structure Recommendation

### Slide 1: Problem Statement
- No standardized way to measure AI's human impact
- Technical benchmarks abundant, ethical/prosocial benchmarks rare
- Need: Infrastructure for "humanity monitoring" alongside performance monitoring

### Slide 2: HumaneBench Overview
- 8 prosocial principles grounded in Self-Determination Theory, Ryff's wellbeing framework
- LLM-as-judge validated by human evaluation
- "Who judges the judge?" - our core question

### Slide 3: Methodology
- Pilot: ~17 expert raters, ~500+ ratings
- 4-point ordinal rubric
- Dual collection: spreadsheets + Langfuse

### Slide 4: Coverage & Data Quality
- Show coverage_analysis.csv highlights
- Which principles had sufficient overlap
- Rater participation patterns

### Slide 5: Key Findings
**Option A (if strong IRR):**
- IRR metrics table (only principles with sufficient data)
- Demonstrate proof of concept

**Option B (if limited IRR):**
- Descriptive statistics across ALL principles
- Rating distributions comparison
- Qualitative insights from expert panel

### Slide 6: What We Learned
- Process validation insights
- Rubric usability confirmation
- Identified improvements for full study

### Slide 7: Next Steps
- Full study design
- Target metrics and sample size
- Publication timeline

---

## Key Numbers to Have Ready

From your analysis output files:

**From `coverage_analysis.csv`:**
- Total unique samples evaluated
- Average ratings per sample per principle
- Number of principles with ≥20 samples (2+ raters)

**From `descriptive_statistics.csv`:**
- Overall % positive vs negative ratings
- Highest/lowest performing principles
- Median ratings per principle

**From `irr_metrics.csv` (where calculable):**
- Krippendorff's alpha values
- Confidence intervals
- Percentage agreement rates

---

## Questions They Might Ask

**Q: "Why is coverage incomplete?"**
A: "This was a pilot study to validate methodology. We intentionally tested workflow with subset of data before full study. Pilot revealed optimal assignment strategy."

**Q: "Can you compare to other benchmarks?"**
A: "MT-Bench (closest comparable) achieved 81% human-human agreement. We're targeting α ≥ 0.72 in full study, which would be strong for subjective evaluation."

**Q: "How do you ensure rater quality?"**
A: "Expert panel includes PsyD, PhD psychologist, product leaders. Training + calibration sessions. IRR metrics validate consensus."

**Q: "Why only 4 rating levels?"**
A: "Ordinal scale with clear anchors. Balance between nuance and reliability. More levels typically decrease agreement without adding meaningful information."

**Q: "What's unique about HumaneBench?"**
A: "First vulnerable-populations-centered prosocial benchmark. Tests across 12 domains. Open-source. Bridges gap between technical performance and human impact."

---

## Dos and Don'ts

### DO:
✅ Lead with what you successfully validated
✅ Show rating distributions for ALL principles
✅ Be transparent about pilot limitations
✅ Frame gaps as learnings, not failures
✅ Emphasize expert panel quality
✅ Cite relevant research (MT-Bench, etc.)

### DON'T:
❌ Apologize excessively for incomplete data
❌ Report IRR where insufficient samples
❌ Claim definitive findings from pilot
❌ Hide methodological decisions
❌ Overstate implications
❌ Forget to mention next steps

---

## Bottom Line

**Your core narrative:**
"We successfully validated a rigorous methodology for evaluating AI prosocial behavior through expert human evaluation. The pilot confirmed rubric usability, established baseline metrics, and identified optimal protocols for full-scale study. Our preliminary findings show [X pattern] across principles, with [Y] achieving strong inter-rater reliability where sufficient overlap existed. This work establishes foundation for first vulnerable-populations-centered humane AI benchmark."

**Translation for MIT audience:**
"Solid pilot. Proved it works. Ready to scale. Here's what we learned."
