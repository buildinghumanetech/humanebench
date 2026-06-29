# HumaneBench evaluation provenance

This document establishes that the reported HumaneBench numbers were produced by
scoring the **finalized dataset** — not by iteratively tuning prompts against eval
results until the numbers looked good. The argument is mechanical and independently
checkable: run `python scripts/verify_provenance.py`.

## TL;DR

- Every reported run scored prompts **byte-identical** to the frozen dataset. We prove
  this by hashing the `(id, input, target)` triples embedded in each `.eval` log and
  showing they equal the frozen prompt hash
  `e1af241db346e0299bb96c893b43746bff7a70d7a50efdd105ba9ded3cfe5d5f`.
- Because each run *is* bound to the frozen prompts, "the prompts were tuned afterward"
  is logically impossible for the reported numbers — there is no later prompt version
  that was scored.
- The prompt content froze at commit
  [`9dc15bd`](https://github.com/buildinghumanetech/humanebench/commit/9dc15bd) (2025-11-16
  23:28). All reported runs executed **2025-11-17 → 2025-11-23**, after the freeze.
- Every later edit to `data/humane_bench.jsonl` (Mar–Apr 2026) preserves that exact
  prompt hash — i.e. it touched only metadata / exclusion flags, never `input` or
  `target`.

## Why content hashes, not "creation timestamps"

The intuitive instinct is to preserve file creation timestamps. We deliberately do **not**
rely on those: git does not store file mtimes, and both filesystem and git commit dates
are trivially forgeable. They prove nothing to a skeptical reader.

The robust evidence is **content-binding**. An Inspect `.eval` log embeds the full
per-sample `input` and `target` it actually scored. We hash those and compare to the
frozen dataset. Identity of content is direction-independent and tamper-evident: you
cannot fake an `.eval` whose 800 embedded prompts hash to the frozen set unless they
genuinely *are* the frozen set. Timestamps and git revisions are recorded as
corroboration only.

## The artifacts

| Artifact | What it is |
| --- | --- |
| `humanebench/provenance.py` | Single source of truth: the freeze commit, the frozen prompt hash, the canonical hashing recipe, and the helpers used by both scripts below. Small and human-auditable. |
| `scripts/build_provenance.py` | Regenerates the manifest from the logs + repo. |
| `provenance/MANIFEST.json` / `MANIFEST.md` | Per-run record: file SHA-256, `eval.created`, `eval.revision.commit`, embedded prompt hash, and pass/fail of each binding check. |
| `scripts/verify_provenance.py` | Independent verifier. Recomputes every claim from disk and exits non-zero on any mismatch. No network needed. |
| Zenodo deposit (DOI: _pending_) | Immutable, third-party-timestamped archive of the 54 raw `.eval` logs + this manifest + the frozen dataset, so anyone can download and re-verify. |

The raw logs (~0.5 GB) are **not** committed to git (`logs/` is gitignored, and the repo
flags possible sensitive content). They live in the Zenodo deposit; the repo carries the
hashes and the verifier.

## The canonical prompt hash

`FROZEN_PROMPT_HASH` is `sha256` over the **sorted** `(id, input, target)` triples, each
record encoded as `id \x1f input \x1f target \x1e` (UTF-8). Sorting by `id` makes the hash
independent of dataset row order and of sample-file order inside an `.eval` zip. Metadata
(`domain`, `vulnerable-population`, `excluded_from_analysis`) is intentionally excluded:
the hash captures exactly what a model is shown and scored on, nothing else.

Reproduce it by hand without any of our code:

```bash
git cat-file -p 9dc15bd:data/humane_bench.jsonl \
  | python3 -c 'import sys,json,hashlib; \
      p=sorted((r["id"],r["input"],r["target"]) for r in map(json.loads,sys.stdin)); \
      h=hashlib.sha256(); [h.update(f"{i}\x1f{x}\x1f{t}\x1e".encode()) for i,x,t in p]; \
      print(h.hexdigest())'
# -> e1af241db346e0299bb96c893b43746bff7a70d7a50efdd105ba9ded3cfe5d5f
```

Unzip any published `.eval`, hash its `samples/*.json` `input`/`target` the same way, and
you get the same digest.

## Chronology, honestly

- **Prompt-content freeze: 2025-11-16 23:28** (commit `9dc15bd`). Note this is *not* the
  Nov 7–8 "final dataset" commits (`822833f`, `1e6fb71`) — those have a **different**
  prompt hash, because prompt text was still being edited until Nov 16. We anchor on the
  hash precisely so this nuance can't be used against us: `9dc15bd` is the earliest commit
  whose prompt hash equals what was actually run.
- **Reported runs: 2025-11-17 → 2025-11-23**, each with `eval.created` after the freeze.
- **Post-run dataset edits** (`b754d34`, `d119c0f`, `79cdbcb`, `ef43b81`, Mar–Apr 2026)
  all preserve the frozen prompt hash → metadata / exclusion-flag changes only.

GitHub records the push/commit dates of `9dc15bd` and the post-freeze commits server-side,
and the Zenodo deposit carries a trusted third-party timestamp on the logs. Combined with
content-binding to the publicly-dated freeze commit, this establishes the ordering without
relying on any single forgeable clock.

## What this proves, and what it does not

**Proven (robust):**
- Each reported run scored the frozen prompt set (content hash identity).
- The dataset's `input`/`target` have not changed since the freeze; later commits are
  metadata-only.
- The runs postdate the freeze.

**Acknowledged limitations (not hidden):**
- **Run-time git commit is local-only.** Each `.eval` records `eval.revision.commit` (e.g.
  `7032d35`), but that working commit was never pushed and does not resolve in the public
  history. We therefore do **not** rely on git ancestry of the run-time SHA; the content
  hash carries the proof, and the manifest flags `revision_resolved_in_repo: false`
  honestly.
- **Judge models are not version-pinned.** The 3-judge ensemble is addressed via OpenRouter
  (`claude-4.5-sonnet`, `gpt-5.1`, `gemini-2.5-pro`), which serves unversioned endpoints.
  Re-running later may not reproduce identical judge outputs. This is a reproducibility
  limitation of the judging step, separate from the dataset-integrity claim above.
- **"Existed-by," not "existed-exactly-at."** Third-party timestamps (Zenodo, GitHub) bound
  when artifacts existed; the dataset-integrity claim does not depend on exact run instants.

## Related rigor already in the repo

- `paper_notes/cut_diff_report__cuts_v2_all_confabulation.md` — the 12 excluded items move
  every headline number by < 0.005 and reclassify zero models; exclusions were flagged in
  place (`metadata.excluded_from_analysis`), never deleted or silently rewritten.
- `tables/golden_set_provenance.md` — provenance of the 24-item human-agreement validation
  set (scored by `src/golden_questions_task.py`; logged at
  `logs/golden_questions_eval/`).
- `humanebench/bootstrap.py` — fixed bootstrap design (seed `20260407`, 1000 replicates)
  shared by every CI in the paper.

## How to verify

```bash
# Against the in-repo logs:
python scripts/verify_provenance.py

# Against a fresh Zenodo download (no repo git history needed for content checks):
python scripts/verify_provenance.py --logs-dir /path/to/extracted/logs
```

A clean tree prints `PROVENANCE VERIFIED: all checks passed.` and exits 0. Flip a single
byte of any prompt in a log or in the dataset and it exits non-zero.
