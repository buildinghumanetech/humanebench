# Transcript input formats

The scorer accepts any of the following. All are normalized to a `User:` / `Assistant:`
transcript before judging.

## 1. Plain text (simplest)

A `.txt` or `.md` file with speaker-labelled turns. Recognised labels (case-insensitive):
`User:` / `Human:` for the person, `Assistant:` / `AI:` / `Agent:` / `Bot:` for the model.

```
User: I keep talking to you for hours instead of doing my work.
Assistant: That's a sign it might help to take a break. What's the task you're avoiding?
User: A report due tomorrow.
Assistant: ...
```

Lines without a label are treated as a continuation of the previous turn. If the file has
no recognisable labels at all, it is passed to the judges verbatim (they score whatever
text you give them).

## 2. JSON

Either a bare list of message objects, or an object with a `messages` key:

```json
[
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]
```

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

`role` values `user` / `human` map to **User**; `assistant` / `ai` / `agent` / `model` /
`bot` map to **Assistant**. A `system` message, if present, is included as `System:` for
context but is not itself scored. `content` may be a string or a list of
`{"type": "text", "text": ...}` blocks (other block types are ignored).

## 3. Stdin

Pass `-` as the path and pipe text in:

```bash
cat session.txt | python scripts/humanebench_score.py -
```

## Notes

- Only the **assistant's** behavior is scored; user turns are context.
- Redact real end-user PII before scoring and sharing results.
- Keep one session per file. To score many, loop over files and aggregate the HumaneScores
  yourself (segment by scenario type / distress level before averaging — see the caveats).
