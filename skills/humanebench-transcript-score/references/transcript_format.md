# Transcript input formats

Every format is converted to the CLI's normalized schema, `humanebench.transcript/v1`,
before anything is scored. That conversion is what lets the CLI path and the Python
fallback judge identical bytes. `python scripts/humanebench_score.py FILE --emit-normalized`
prints the normalized JSONL. Pipe it into `humanebench ingest --stdin --source normalized`.

## 1. Claude Code session logs (`~/.claude/projects/**/*.jsonl`)

These are detected automatically and parsed the way `cli/src/adapters/claude_code.rs` parses
them:

- **Tree:** each record links to its parent through `parentUuid`. The newest-leaf path of
  each connected component is kept, and abandoned regenerations are dropped and counted.
- **Kept:** only conversation. These are dropped:
  - bookkeeping records (`attachment`, `file-history-*`, `queue-operation`, `system`, …)
  - `isMeta` injections and compaction summaries
  - `tool_result`-only user records
  - harness envelopes such as `<system-reminder>`, `<command-name>` and
    `[Request interrupted`
- **Tool calls** (`tool_use` blocks) become context on the next text-bearing assistant turn:
  `[actions taken before responding: Bash(git status), Read(src/app.ts)]`. They are never
  scored on their own.
- **Sidechain** (`isSidechain: true`) subagent turns are excluded from scoring and from the
  rollup arc. To include a session's subagent logs, concatenate them with the main file. They
  stay excluded, but the tree stays whole.

Codex logs and ChatGPT / Claude-app exports are CLI-only (Path A1). The ChatGPT and
Claude-app adapters are flagged unverified in the CLI.

## 2. Normalized JSONL (`humanebench.transcript/v1`)

One record per line. Run `humanebench schema` for the contract:

```json
{"schema":"humanebench.transcript/v1","source":"myapp","session_id":"s1","turn_id":"s1:0001","role":"user","text":"...","timestamp":"2026-09-01T10:00:00Z"}
```

Optional fields are `parent_id`, `model`, `actions` (`[{name, summary}]`) and `sidechain`.
This is the format to emit from your own logs.

## 3. Plain text

`User:` / `Human:` for the person. `Assistant:` / `AI:` / `Agent:` / `Bot:` for the model.
Labels are case-insensitive, and unlabelled lines continue the previous turn.

```
User: I keep talking to you for hours instead of doing my work.
Assistant: That's a sign it might help to take a break. What's the task you're avoiding?
```

- `System:` turns are dropped, because the schema has no system role.
- Text with no labels at all is an error. Per-turn scoring needs to know which text is an
  assistant turn, so the scorer never guesses.

## 4. JSON message lists

A bare list of `{"role", "content"}`, or `{"messages": [...]}`:

- **Roles:** `user` / `human` map to user. `assistant` / `ai` / `agent` / `model` / `bot`
  map to assistant. `system` and `tool` messages are dropped.
- **Content** may be a string or a list of blocks. `text` blocks are the message, and
  `tool_use` blocks become action context.

## Notes

- **Timestamps.** Plain text and JSON message lists have none. The scorer synthesizes them
  deterministically (2000-01-01 plus one minute per message) so content hashes stay stable.
  The report says they were synthesized. Sessions split at 6-hour idle gaps
  (`--idle-gap-hours`), which never happens with synthesized times.
- **Stdin.** Pass `-` as the path.
- **Privacy.** Only the assistant's behavior is scored, and user turns are context, but both
  are sent to the judge. Redact real end-user PII before scoring.
