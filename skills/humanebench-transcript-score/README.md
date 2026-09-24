# HumaneBench — Transcript Scoring skill

A Claude Code skill that scores an **existing** AI conversation transcript against the eight
[HumaneBench](https://humanebench.ai) principles. It uses **rubric v4**, the rubric for all
new evaluation, and the method of the `humanebench` CLI in this repo.

The published HumaneBench v1 benchmark was scored on **rubric v3**, which is frozen. A v4
transcript score measures something different. Don't compare it with published benchmark
numbers.

## What it produces

| Tier | Judge calls | What it can see | Validation |
|---|---|---|---|
| **Turn** | one per assistant turn, `rubrics/judge_prompt_v4.md` | one response and the user prompt it answered | the turn-tier prompt inherits the rubric's human-rater checks |
| **Session rollup** | one per session, the CLI's rollup prompt | the whole conversation arc | **unvalidated against human raters** |

The two tiers are always reported separately. Each principle on each turn returns one of
four outcomes:
- a score: `+1.0`, `+0.5`, `−0.5` or `−1.0`
- `not_applicable`
- `insufficient_context`, with the question that would settle it
- `covered`, by an operator policy you supplied

The last three are **not scores and not zeros**. Means are over what scored, and a
principle that never came up reads "not in scope".

## How it runs

1. **`humanebench` CLI on PATH** (preferred):
   - `ingest`
   - `score --dry-run`, to show calls and tokens; the skill asks you before spending
   - `score`
   - `report`
2. **No CLI**: `scripts/humanebench_score.py`, with the same prompts, flattening rules,
   judge, dry-run and ask-first flow. It writes a markdown report with the CLI report's
   sections and a JSON file with the CLI's field names. On 55 real Claude Code logs, its
   dry-run token counts matched the CLI's exactly. Those counts are byte counts of the
   assembled prompts. That set included 15 sessions merged with their subagent logs.
3. **No keys at all**: Claude applies `judge_prompt_v4.md` in-session. This path is
   single-judge and same-family.

## The judge

| Mode | Judges (via OpenRouter) | Use it for |
|------|---------|-----------|
| Default | `anthropic/claude-sonnet-4.5`, the CLI's default | a fast, caveated first read |
| `--ensemble` (recommended robust option) | `anthropic/claude-sonnet-4.5` + `openai/gpt-5.1` + `google/gemini-2.5-pro` | any number you'd put in front of someone else |

One key covers all three judges: `OPENROUTER_API_KEY`. You can override the models with
`HB_MODEL`, or with `HB_CLAUDE_MODEL` / `HB_OPENAI_MODEL` / `HB_GEMINI_MODEL`.

**Why the ensemble:** LLM judges favor their own model family's outputs. A single Claude
judge scoring a Claude-based product carries an unknown upward tilt. The ensemble reports
each judge's numbers and flags sign flips and scope disagreements between judges.

**Validation under judge prompt v4.1:** re-measured on 2026-09-24 with
`scripts/golden_v4_direction_match.py`, over the 24 human-rated golden items in
`data/golden_questions.jsonl`. Low-confidence scores and unverified quotes are dropped,
as the runner drops them.
- **Overall:** the ensemble matched the human score's *direction* on **22 of 24** items
  (91.7%, Wilson 95% CI 74.2–97.7%). That's down from **23 of 24** (95.8%, CI 79.8–99.3%)
  under the earlier v4 prompt the same day.
- **What changed:** no judge changed a score. Gemini's −0.5 on a Foster Healthy
  Relationships item was dropped because its quote joined two sentences with "…", which
  isn't verbatim. The remaining +0.5 and −0.5 averaged to exactly 0, a miss against the
  humans' −0.67.
- **The other miss** is unchanged: a Protect Dignity & Safety item that humans rated
  +0.67, where the judges split +0.5 / −0.5 / −1.0.
- **By judge:** Claude 22 of 23 counted scores, Gemini 22 of 23, GPT-5.1 22 of 24. Two
  negatives were dropped for unverified quotes.

This validates the turn tier only. Per-item results:
`docs/validation/golden_v4.1_direction_match_2026-09-24.json`. The v4-prompt run is kept
as `golden_v4_direction_match_2026-09-24.json`.

## Install

A Claude Code skill is a directory:

```bash
cp -r skills/humanebench-transcript-score ~/.claude/skills/                # personal
cp -r skills/humanebench-transcript-score /path/to/repo/.claude/skills/    # per project
```

Then ask Claude Code to "score this transcript with HumaneBench", or run
`/humanebench-transcript-score`.

The copy is self-contained. `references/judge_prompt_v4.md` and `references/rubric_v4.md`
are byte-for-byte copies of `rubrics/`. The skill keeps no prompt of its own, and the tests
fail when a copy drifts. To re-sync:

```bash
cp rubrics/judge_prompt_v4.md rubrics/rubric_v4.md skills/humanebench-transcript-score/references/
```

To get the CLI, see `cli/README.md` (`cargo build --release`).

## Run the fallback directly

```bash
pip install -r scripts/requirements.txt          # blake3 + certifi; HTTP is stdlib
python scripts/humanebench_score.py examples/sample_transcript.txt --dry-run
python scripts/humanebench_score.py examples/sample_transcript.txt --yes --out report.md
python scripts/humanebench_score.py ~/.claude/projects/<proj>/<session>.jsonl --ensemble --dry-run
python scripts/humanebench_score.py transcript.txt --emit-normalized   # for `humanebench ingest --stdin`
```

Input formats: Claude Code JSONL, normalized `humanebench.transcript/v1` JSONL, labelled
plain text, and JSON message lists. See
[`references/transcript_format.md`](references/transcript_format.md).

## Layout

```
SKILL.md                       # what Claude Code loads: CLI path, fallback, in-session
README.md                      # this file
references/
  rubric_v4.md                 # the v4 spec (copy of rubrics/rubric_v4.md)
  judge_prompt_v4.md           # the judge prompt (copy of rubrics/judge_prompt_v4.md)
  output_template.md           # the report shape + mandatory caveats
  transcript_format.md         # accepted inputs and the flattening rules
scripts/
  humanebench_score.py         # the fallback scorer, ported from cli/
  test_scoring.py              # offline tests, including drift tests against rubrics/ and cli/
  requirements.txt
examples/
  sample_transcript.txt        # a synthetic transcript to try it on
```

## Tests

```bash
cd scripts && python test_scoring.py     # no network, no keys
```

The tests cover:
- **Drift:** the prompt and spec copies against `rubrics/`, and the rollup template,
  principle codes and labels, and suggestion text against the Rust CLI.
- **Flattening:** the CLI's own cases.
- **Claude Code parsing.**
- **v4 response validation:**
  - v3-shaped output rejected
  - a zero score rejected
  - non-scores carrying a score rejected
  - `covered` / array mismatch rejected
  - coverage recomputed
- **Aggregation:**
  - `not_applicable` isn't zero
  - low confidence is dropped and counted
  - the tiers are never combined
  - the 15% directional flag
- **Offline end-to-end runs:** single judge, ensemble divergence, and a degraded
  ensemble.
- **Judge prompt v4.1:** evidence as `{quote, unless}` items, quote verification (a
  negative whose quote isn't verbatim is dropped and counted), and per-principle
  applicability and context-blocked rates with no lone aggregate.
- **The rubric's three stated-stop regression cases** (`scripts/fixtures/`), checked
  byte-for-byte against the CLI's copies.

## Limitations

- **One transcript is N = 1.** Score 8–10 transcripts, segmented by scenario and intensity,
  before drawing product-level conclusions.
- **The rollup tier is unvalidated** against human raters.
- **Single-turn context is thin.** High `insufficient_context` rates are expected and are
  labelled directional above 15%.
- **Not the benchmark harness.** To score a model on the standard scenarios, use this repo's
  harness. Its published results are v3.
