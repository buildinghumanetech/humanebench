# `judge_prompt_v3.md`

The operational evaluator prompt the CLI compiles into its binary.

**This is not a copy of `rubrics/rubric_v3.md` at the repo root.** That file is the
human-readable rubric specification. This one restates the same 8 principles and the same
4-level scale as instructions a model executes, adds a strict output schema, and carries
two substitution slots the CLI fills per turn:

- `{{.UserPrompt}}` — the most recent kept user turn
- `{{.MessageContent}}` — the assistant turn under evaluation, with tool/action context appended

`src/judge/mod.rs` embeds this file with `include_str!`, so a built binary cannot drift
from the prompt it claims to implement. `src/judge/rollup.rs` reuses the scale and
principle definitions by splitting this text at `"Now, evaluate the following"` and
discarding the per-turn framing.

## Known gap

The two documents are maintained separately. Editing the root rubric does not change CLI
scores; editing this file does. Reconciling them into one source of truth — most likely by
generating this prompt from the root rubric — is open work and was deliberately left out of
the port that introduced this directory, because the slots and the split marker are load
bearing and a naive swap changes every cached score hash.
