# HumaneBench CLI

A single-binary Rust CLI that scores your own conversation history against the HumaneBench
v4 rubric and emits a local HTML report.

The benchmark measures frontier models against synthetic prompts. This turns the same
rubric on real conversations, so you can ask "is this assistant actually treating me
humanely?" instead of trusting a leaderboard.

## Build

```sh
cargo build --release      # -> target/release/humanebench
```

No runtime dependencies. The rubric is compiled into the binary, so a binary cannot drift
from the rubric it claims to implement.

## Use

```sh
humanebench ingest ~/.claude/projects        # detect, normalize, store
humanebench score --dry-run                  # what would this cost? (no API key needed)
humanebench score --since 3d --dry-run       # ...and what would just the last 3 days cost?
humanebench score                            # judge everything not already cached
humanebench report --out report.html         # free, never re-judges
humanebench report --share --out share.html  # excerpt-free artifact
humanebench mcp                              # serve the corpus over stdio
```

Ingest, score, and report are separate commands on purpose. Scoring is the only one that
costs money, so it is the only one you have to consciously invoke — and re-rendering a
report never re-spends.

Scoring judges through OpenRouter by default, which needs one environment variable:
`OPENROUTER_API_KEY`. `--provider vertex` uses Google Vertex AI instead, which needs a
Google Cloud project (`gcloud config set project`) and a login (`gcloud auth login`) —
more setup, and its tokens expire, so the default is the one that works from a single
key.

The first run names the destination, states exactly what
leaves the machine, and requires explicit opt-in — and asks again if the destination
changes.

## How it works

Everything narrows to one documented JSONL contract. Built-in adapters and a converter you
write in an afternoon enter the engine at exactly the same point, which is why a new
harness needs no Rust and no merged PR.

```
sources ──> adapters ──> humanebench.transcript/v1 ──> engine ──> report.html
                          (or your own converter)                 report.share.html
                                                                  MCP over stdio
```

Two scoring tiers: one judge call per assistant turn, plus one session-level rollup. Ten
turns cost eleven calls, not ten. The rollup is not a convenience — engagement loops,
fostered dependency, and sycophancy drift only exist across turns, so a per-turn judge is
structurally blind to four of the eight principles.

Each judge call uses the v4.1 judge prompt. The CLI checks every negative score's quoted
evidence against the response. A negative whose quote is not there verbatim is dropped from
the report and counted. Reports show applicability and context-blocked rates per principle,
and floor applicability separately, never as a single aggregate. See `rubric/README.md`.

Run `humanebench schema` for the field-by-field contract.

### What gets scored, and what doesn't

Most records in an agent log are not conversation. A real Claude Code session in testing
held 3,681 records across 357 sessions but only 2,157 text-bearing assistant turns; scoring
every tool call instead would have cost many times more to mostly ask an ethics rubric what
it thinks of a file read.

- **Trees flatten to the newest-leaf path.** Abandoned regenerations are discarded — score
  what the person actually saw. The report says how many were dropped.
- **Sidechain turns are excluded.** A subagent talking to itself had no human on the other
  end.
- **Tool/MCP/skill calls are context, never the subject.** They fold into the turn's judge
  input as a one-line summary.
- **The user prompt walks backward.** Most assistant turns follow tool results, not a
  person typing, so the judge gets the most recent *kept* user turn — not the immediately
  preceding record.
- **`score --since` narrows what is judged, not what a prompt contains.** A session that
  straddles the cutoff is rolled up whole — an arc judged on half its turns is a different
  claim about a different conversation. The cutoff stays out of the content hash, so
  scoring three days now and thirty days later reuses the overlap instead of paying twice.

## Sources

| Source | Status |
|---|---|
| Claude Code | Verified against real session logs |
| Codex | Verified against real rollout logs |
| ChatGPT export | ⚠ Unverified — field names never confirmed against a real archive |
| Claude app export | ⚠ Unverified — same caveat |
| Hermes | Not built: format unpinned. Use the converter path below. |

Unverified adapters warn on use and are flagged in the report. Confirm them against a real
export before trusting their numbers.

### Bring your own source

Write a converter in any language that emits the documented schema:

```sh
your-converter < logs | humanebench ingest --stdin --source normalized
```

`--source` selects the *adapter* — which input format to parse — so already-normalized JSONL
uses `normalized`. It is an allowlist (`claude-code`, `codex`, `chatgpt`, `claude-app`,
`hermes`, `normalized`); an invented name is rejected. The free-form origin tag is the
`source` **field inside each record**, which is surfaced in reports and filters but never
parsed — that is where `yourtool` belongs.

## Privacy

- Transcripts on disk are read-only and never modified.
- One arrow crosses the trust boundary — turn text plus the rubric, to the judge model —
  and only after explicit opt-in that states exactly that. Opt-in is per destination, so
  consenting to one provider or Google Cloud project is not consent to another.
- A content-hash cache means a given turn crosses at most once, ever. A rubric revision or
  model swap correctly invalidates rather than serving stale scores.
- Reports are files on disk. Nothing is uploaded; sharing one is a deliberate second
  command.
- `--share` drops verbatim excerpts *and* judge reasoning (reasoning restates what the turn
  said), and strips citations from suggestions.
- The MCP server is read-only, cannot trigger scoring, and withholds excerpts unless
  `include_text` is explicitly passed.
- Suggestions are human-reviewed. The tool never writes to `CLAUDE.md`, `AGENTS.md`, or any
  config file.

## Known limits

- **Single judge.** Noisier than the benchmark's validated ensemble. The report says so and
  labels the model and regime, so single-judge numbers are never silently compared against
  ensemble ones.
- **The rollup prompt is net-new** and, unlike the turn tier, has never been validated
  against human raters.
- **No comparison against published benchmark numbers — different rubric *and* different
  statistic.** The CLI runs **v4**. The published HumaneBench v1 results, the whitepaper and
  the leaderboard are **v3**, which is frozen. Never place a v4 score beside a v3 one, and
  never call a v4 score leaderboard-comparable. On top of the version gap, three mechanical
  divergences from the production scorer (`humanebench/scorer.py`):
  - **The denominator.** The benchmark scores each sample on the *one* principle its prompt
    was built to stress, so a principle's mean is taken only over turns that engage it. The
    CLI scores *every* turn on all eight and means them, so each principle's mean is
    dominated by turns where that principle is barely in play.
  - **One judge, not an ensemble.** The benchmark means severities across several judge
    models and yields NaN if any of them marks the item invalid. The CLI calls one model;
    `regime` is the literal `single`.
  - **Opposite missing-data policy, and under v4 it bites constantly.** A principle with no
    usable score is 0 in the benchmark and averaged into the HumaneScore
    (`scorer.py:130,136`). The CLI excludes it. Under v3 this was theoretical, because every
    principle always carried a score; under v4 three of the four outcomes carry no score at
    all, so most turns now have a denominator smaller than eight. `not_applicable` is not a
    zero, and a principle in scope on no turns reports as "not in scope" rather than 0.00.

  Smaller ones in the same direction: the benchmark rounds to 2dp at both aggregation stages
  and the CLI stores full precision (rounding is display-only); the benchmark has no tier
  concept, while the CLI aggregates turn and rollup tiers separately and never combines them.

- **Principle ids still fork the benchmark's.** The CLI and `rubrics/judge_prompt_v4.md` both
  use snake_case codes (`respect_attention`); the benchmark's published slugs are kebab-case
  and descriptive (`respect-user-attention`). No tooling can join a CLI report to leaderboard
  output by principle id without a translation table. Renaming them is a hash-invalidating
  edit, so it is deliberately not bundled here.

  The three prompt divergences this README used to list — an invented 125-character limit on
  sensitive-content responses, `-1.0` anchors gated on intent ("Deliberately"), and three of
  the seven global rules missing — are **gone**. They came from the CLI forking a draft
  template rather than the rubric. v4 is compiled in from `rubrics/judge_prompt_v4.md`
  directly, so there is no fork left to drift.

## Layout

```
src/transcript/  schema types, the flattening rule, idle-gap splitting
src/adapters/    claude_code, codex, chatgpt, claude_app, hermes
src/judge/       prompt assembly, OpenRouter client, response parsing + validation
src/store/       SQLite: turns, scores, content-hash cache
src/report/      HTML rendering, full and --share
src/mcp/         stdio server
rubric/          README.md only; the prompt itself is ../rubrics/judge_prompt_v4.md
```
