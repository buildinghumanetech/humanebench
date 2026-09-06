# HumaneBench CLI

A single-binary Rust CLI that scores your own conversation history against the HumaneBench
v3 rubric and emits a local HTML report.

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

Scoring judges through Google Vertex AI by default, which needs a Google Cloud project
(`gcloud config set project`) and a login (`gcloud auth login`). `--provider openrouter`
uses `OPENROUTER_API_KEY` instead. The first run names the destination, states exactly what
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
your-converter < logs | humanebench ingest --stdin --source yourtool
```

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
- **No comparison against published benchmark numbers.** A personal average would read
  higher than the benchmark average for reasons that have nothing to do with the assistant
  being more humane.

## Layout

```
src/transcript/  schema types, the flattening rule, idle-gap splitting
src/adapters/    claude_code, codex, chatgpt, claude_app, hermes
src/judge/       prompt assembly, OpenRouter client, response parsing + validation
src/store/       SQLite: turns, scores, content-hash cache
src/report/      HTML rendering, full and --share
src/mcp/         stdio server
rubric/          judge_prompt_v3.md, include_str! at compile time
```
