# `cli/rubric/`

**There is no prompt in this directory any more, and that is the point.**

The evaluator prompt the CLI compiles into its binary is
[`rubrics/judge_prompt_v4.md`](../../rubrics/judge_prompt_v4.md) at the repository root.
`src/judge/mod.rs` embeds it with `include_str!("../../../rubrics/judge_prompt_v4.md")`, so
a built binary cannot drift from the prompt it claims to implement.

## Why it moved

Three consumers need this prompt and only one of them is the CLI: the CLI, the pull-request
gate, and any partner running the rubric against their own traffic. A prompt that lives
inside one consumer makes the other two either vendor a copy or take a dependency they do
not want — and a vendored copy is how two documents drift apart.

`judge_prompt_v3.md` used to sit here. It was a byte-identical fork of a draft template in
`evaluator/humanebench_evaluator.py`, not of the canonical rubric, and it inherited that
draft's divergences: an invented 125-character limit on sensitive-content responses, `-1.0`
anchors gated on the word "Deliberately", and three of the seven global rules missing. All
three biased scores upward. They are gone, along with the file, because v4 is compiled from
the canonical document rather than from a copy of a copy.

## Mechanics, unchanged

Two substitution slots the CLI fills per turn:

- `{{.UserPrompt}}` — the most recent kept user turn
- `{{.MessageContent}}` — the assistant turn under evaluation, with tool/action context
  appended

**The split marker `Now, evaluate the following` is load bearing.** `src/judge/rollup.rs`
splits the prompt there to reuse the scale, the principles and the output schema without the
per-turn framing. It must appear exactly once; a test guards that directly, because the
fallback when it is missing is silent.

Editing the root prompt changes CLI scores, and invalidates every cached score: the prompt
text is inside the content hash. Plan a full re-score of any corpus you care about.

## What v4 changed for this CLI

- Each principle returns an `outcome` — `score`, `not_applicable`, `insufficient_context` or
  `covered` — and three of the four carry no score. **`not_applicable` is not a zero.**
  Means are taken over what actually scored, and a principle in scope on no turns reports as
  "not in scope".
- Per-principle `confidence` is the string `high` / `medium` / `low`. The prompt promises the
  runner discards `low` before anyone sees it; the CLI keeps the score in the store and
  excludes it from every mean and findings list, and reports the drop count per principle.
- Top-level `confidence` and `globalViolations` are gone. `covered` and `coverage` are new,
  and `coverage` must satisfy `applicable == scored + context_blocked + covered` — the CLI
  recomputes it from the outcomes rather than trusting the judge's own counts.
- Scores are tagged with the rubric version that produced them. v3 rows already in a store
  are kept and excluded from reports rather than averaged in with v4 ones.

## What judge prompt v4.1 changed for this CLI

- **`evidence` is a list of `{quote, unless}` items**, one per independent finding on the
  principle, with one score and one tier per principle. A live response with a string
  `evidence` is rejected as pre-v4.1 output. Rows stored earlier still load.
- **Quotes are verified.** After every judge call, each negative's quotes are checked
  against the response: the assistant turn, or for a rollup, everything the assistant said
  in the arc. Whitespace is the only normalization, and the quote must be inside the
  response. The gate also accepts the reverse; the CLI does not. A negative with no
  verified quote is kept in the store, excluded from every mean and findings list, counted
  per principle, and reported, as the gate does. `score` prints the count.
- **Coverage is per principle.** Every report has a coverage table per tier: applicability
  rate and context-blocked rate per principle, with floor principles marked. The header
  shows floor applicability for Protect Dignity & Safety and Be Transparent & Honest,
  instead of a single blocked-for-context number. The run-level context-blocked rate
  appears only inside the directional warning, next to the principles that exceed 15%.
  The MCP `query_scores` tool reports the same rates per principle.
- **Scores are tagged `v4.1`.** Rows from the earlier v4 prompt are excluded and counted,
  not averaged in. That now applies to the MCP tools too.
- **The stated-stop regression cases** the rubric requires are fixtures in
  `src/judge/fixtures/`, tested offline in `src/judge/regression.rs`.

## The principle-id fork, still open

Zero of the eight principle ids match the benchmark's published slugs: the CLI and
`judge_prompt_v4.md` use `respect_attention`, the benchmark uses `respect-user-attention`.
No tooling can join a CLI report to leaderboard output by principle id. Renaming is
deliberately not done here — the ids are part of the prompt the judge is handed, so changing
them changes every content hash and invalidates every cached score. Do it once, with a
re-score, not twice.
