# HumaneBench — Transcript Scoring skill

A Claude Code skill that scores an **existing** AI conversation transcript against the
eight [HumaneBench](https://humanebench.ai) principles (rubric v3), producing a
per-principle breakdown (`-1 / -0.5 / +0.5 / +1`) and an overall **HumaneScore**.

It's the transcript-scoring companion to the published benchmark harness at
[`buildinghumanetech/humanebench`](https://github.com/buildinghumanetech/humanebench):
that repo runs standardized scenarios *through a model*; this scores conversations you
**already have** from your product.

## Why the ensemble matters (the whole point)

A single LLM judge inherits its own temperament, and **LLM judges favor their own model
family's outputs**. On one real transcript, two single judges disagreed by 0.37 (0.13 vs
0.50) with three principles flipping sign. So this skill has two modes:

| Mode | Judges | Setup | Use it for |
|------|--------|-------|-----------|
| **Default (Claude-only)** | Claude Sonnet 4.5 | just an Anthropic key | a fast, caveated first read |
| **`--ensemble`** (recommended) | Claude Sonnet 4.5 + GPT-5.1 + Gemini 2.5 Pro | 3 keys | any number you'd put in a deck |

Every result ships with the same-family-tilt warning and an "N=1, one transcript" note.
The default judge is **Sonnet 4.5 on purpose** — it's one of the three judges in the
published ensemble, so single-judge scores stay on the same scale (the published
*methodology* is the full cross-family ensemble, not any single judge; use `--ensemble`
for a number that rests on it).

## Install

A Claude Code skill is a directory. Drop this one where Claude Code looks for skills:

```bash
# Personal (all your projects):
cp -r skills/humanebench-transcript-score ~/.claude/skills/

# Or per-project:
cp -r skills/humanebench-transcript-score /path/to/your/repo/.claude/skills/
```

Then in Claude Code: `/humanebench-transcript-score` (or just ask "score this transcript
with HumaneBench"). Its recommended permanent home is the `humanebench` repo alongside the
rubric and harness — this copy is self-contained (the rubric is embedded) so it's portable.

## Run the scorer directly

```bash
pip install -r scripts/requirements.txt              # Claude-only: just anthropic
# for --ensemble, also:
pip install -r scripts/requirements-ensemble.txt     # adds openai + google-genai

# Claude-only (default)
python scripts/humanebench_score.py examples/sample_transcript.txt

# Cross-family ensemble (recommended)
python scripts/humanebench_score.py examples/sample_transcript.txt --ensemble

# Save a markdown report + raw JSON
python scripts/humanebench_score.py transcript.json --ensemble --out report.md
```

### API keys

| Judge | Env var | SDK |
|-------|---------|-----|
| Claude Sonnet 4.5 | `ANTHROPIC_API_KEY` (or `ANTHROPIC_AUTH_TOKEN`) | `anthropic` |
| GPT-5.1 | `OPENAI_API_KEY` | `openai` |
| Gemini 2.5 Pro | `GEMINI_API_KEY` / `GOOGLE_API_KEY` | `google-genai` |

A judge whose key or package is missing is **skipped with a warning** rather than crashing;
if `--ensemble` can't reach all three, the score is flagged as provisional (not a true
cross-family result). Override any model via `HB_CLAUDE_MODEL` / `HB_OPENAI_MODEL` /
`HB_GEMINI_MODEL`.

## Transcript formats

Plain text with `User:` / `Assistant:` turns, a `.json` list of `{"role","content"}` (or
`{"messages":[...]}`), or piped on stdin (`-`). Details in
[`references/transcript_format.md`](references/transcript_format.md). Only the assistant's
behavior is scored; redact real end-user PII before scoring.

## Layout

```
SKILL.md                       # what Claude Code loads; how to run (script or in-session)
README.md                      # this file
references/
  rubric_v3.md                 # canonical rubric, embedded (self-contained)
  output_template.md           # the report shape + mandatory caveats
  transcript_format.md         # accepted input formats
scripts/
  humanebench_score.py         # the scorer (Claude-only default; --ensemble)
  test_scoring.py              # unit tests for the pure logic (no network)
  requirements.txt             # Claude-only deps (anthropic)
  requirements-ensemble.txt    # extra deps for --ensemble (openai, google-genai)
examples/
  sample_transcript.txt        # a synthetic transcript to try it on
```

## Tests

```bash
cd scripts && python test_scoring.py     # 22 tests, no network / no keys needed
```

Covers transcript parsing (text + JSON), score snapping to the rubric's four values, judge
JSON extraction/validation, ensemble aggregation, judge-divergence detection, and report
rendering (including that the caveats are always present). The judges' network calls are
thin wrappers around the official SDKs and are exercised separately with live keys.

## Limitations

- **One transcript = N of 1.** Scores 8–10 transcripts, segmented by scenario/intensity,
  before drawing product-level conclusions.
- **Multi-turn is an adaptation.** rubric v3 targets single-turn responses; a full session
  is scored holistically across turns.
- **Not the live harness.** To score your product against the *published Sonnet baseline*
  on standard scenarios (system-prompt in place), use the `humanebench` repo's harness.
