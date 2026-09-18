# `rubrics/`

| File | Status |
|---|---|
| `rubric_v1.md`, `rubric_v2.md` | History. |
| `rubric_v3.md` | **Frozen.** The rubric of record for the published HumaneBench v1 results, the whitepaper and the preprint. Do not edit it. Everything scored under it is correctly cited as a v3 score. |
| `rubric_v4.md` | **Operational.** The rubric partner evaluations and the pull-request gate run on. Human-readable specification. |
| `judge_prompt_v4.md` | The same rubric as instructions a model executes, with an output schema and two substitution slots. |

**Do not compare a v4 score to a published v3 score.** A v4 re-run of the 15
models is separate, tracked work; until it lands, the two numbers do not mean
the same thing.

## Why the judge prompt lives here and not under `cli/`

Three consumers need it and only one of them is the CLI: the CLI, the
pull-request gate, and any partner running the rubric against their own traffic.
A prompt that lives inside one consumer makes the other two either vendor a copy
or take a dependency they do not want, and a vendored copy is how the two
documents drift apart.

`judge_prompt_v3.md` currently sits under `cli/rubric/` on the CLI branch, and
its own README says the drift between it and the root rubric is open work. This
is the fix for v4: one location, three consumers.

## Consuming it

**Substitution slots**, filled per turn by the caller:

- `{{.UserPrompt}}` — the most recent kept user turn
- `{{.MessageContent}}` — the assistant turn under evaluation

**Split marker.** The text `Now, evaluate the following` appears exactly once.
Session-level rollup logic splits there and discards the per-turn framing, so
the marker is load bearing. Do not reword it.

**Operator policy documents**, when the caller has them, are appended after the
prompt. When none are supplied the judge is required to return an empty
`covered` array and cannot fire transparency trigger 4. That guard only holds if
the caller actually supplies nothing rather than supplying a placeholder.

## What a caller has to change to move from v3 to v4

The output schema is different, and this is the part that bites:

- Each principle carries an `outcome` (`score` / `not_applicable` /
  `insufficient_context` / `covered`) instead of a bare score. Three of the four
  outcomes have no score at all, so a required non-optional score field will
  fail to deserialize.
- Per-principle `confidence` is the string `"high"` / `"medium"` / `"low"`, not
  a float. The prompt promises the runner discards `"low"` before anyone sees
  it. Implement that, or cut the promise.
- Top-level `confidence` and `globalViolations` are gone. `covered` and
  `coverage` are new.
- **`not_applicable` is not zero.** Average over `coverage.scored`, never over
  eight, and surface `coverage.context_blocked`. A run above 15% context-blocked
  is directional and has to be labeled that way.
- `coverage` must satisfy `applicable == scored + context_blocked + covered`.

For the Rust CLI specifically: `src/judge/mod.rs` embeds its prompt with
`include_str!`, so a built binary cannot drift from the prompt it claims to
implement. Pointing that at `../../rubrics/judge_prompt_v4.md` is a one-line
change, and it invalidates every cached score, because cache keys derive from
the rubric text. Plan a full re-score of any corpus you care about, and do not
mix v3 and v4 scores in one report.

For the pull-request gate: it vendors the rubric byte-for-byte and pins the
commit. Moving that pin means re-running every demo pull request before merging.
The clean ones have to stay clean.
