# HumaneBench evaluation provenance

This document records how to reproduce and independently verify that the reported
HumaneBench numbers were computed against the finalized dataset. Everything here is
factual and checkable — run `python scripts/verify_provenance.py`.

## Summary

- The 800-prompt dataset's content (`input`/`target`) is fingerprinted by a canonical
  hash, `FROZEN_PROMPT_HASH = e1af241db346e0299bb96c893b43746bff7a70d7a50efdd105ba9ded3cfe5d5f`.
- Every reported run's `.eval` log embeds the exact prompts it scored; all of them hash to
  that value, binding each run to the finalized prompt set.
- Prompt content reached its final state at commit `9dc15bd` (2025-11-16). The 45 reported
  runs executed 2025-11-17 → 2025-11-23. Later dataset commits change only metadata /
  exclusion flags and preserve the prompt hash.

## Why a content hash, not file timestamps

Git does not preserve file mtimes, and local/commit dates are easily changed, so this
record does not depend on them. The durable identifier is the **content hash** of the
prompts: each `.eval` embeds the `input`/`target` it scored, which we check against the
dataset's frozen prompt hash. Embedded timestamps and git revisions are recorded as
supplementary metadata only.

## Artifacts

| Artifact | What it is |
| --- | --- |
| `humanebench/provenance.py` | Single source of truth: the freeze commit, the frozen prompt hash, the canonical hashing recipe, and the helpers used by the two scripts below. |
| `scripts/build_provenance.py` | Regenerates the manifest from the logs + repo. |
| `provenance/MANIFEST.json` / `MANIFEST.md` | Per-run record: file SHA-256, `eval.created`, `eval.revision.commit`, embedded prompt hash, and the result of each binding check. |
| `scripts/verify_provenance.py` | Independent verifier. Recomputes every claim from disk and exits non-zero on any mismatch. No network needed. |
| Zenodo deposit ([10.5281/zenodo.21046964](https://doi.org/10.5281/zenodo.21046964)) | Archived copy of the raw `.eval` logs + this manifest + the finalized dataset, so the logs can be downloaded and re-verified. |

The raw logs (~0.5 GB) are not committed to git (`logs/` is gitignored); they live in the
Zenodo deposit, while the repo carries the hashes and the verifier.

## The canonical prompt hash

`FROZEN_PROMPT_HASH` is `sha256` over the **sorted** `(id, input, target)` triples, each
record encoded as `id \x1f input \x1f target \x1e` (UTF-8). Sorting by `id` makes the hash
independent of dataset row order and of sample-file order inside an `.eval` zip. Metadata
(`domain`, `vulnerable-population`, `excluded_from_analysis`) is excluded, so the hash
captures exactly what a model is shown and scored on.

Reproduce it without any of our code:

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

## Dataset timeline

- **Prompt content finalized: commit `9dc15bd` (2025-11-16 23:28).** This is the earliest
  commit whose prompt hash equals the value every reported run scored. (The Nov 7–8 "final
  dataset" commits `822833f` / `1e6fb71` have a *different* prompt hash — prompt text was
  still being edited until Nov 16 — which is why the finalized state is identified by hash
  rather than by date.)
- **Reported runs: 2025-11-17 → 2025-11-23.**
- **Post-freeze dataset commits** (`b754d34`, `d119c0f`, `79cdbcb`, `ef43b81`, Mar–Apr
  2026) preserve the prompt hash; they change only metadata / exclusion flags.

## Reproducibility notes

- **Run-time git commit is local-only.** Each `.eval` records `eval.revision.commit` (e.g.
  `7032d35`), a working commit that was not pushed and does not resolve in public history.
  Verification therefore uses the content hash, not that revision; the manifest records
  `revision_resolved_in_repo: false`.
- **Judge models are not version-pinned.** The 3-judge ensemble is addressed via OpenRouter
  (`claude-4.5-sonnet`, `gpt-5.1`, `gemini-2.5-pro`), which serves unversioned endpoints, so
  re-running may not reproduce identical judge outputs.

## Related materials

- `tables/golden_set_provenance.md` — provenance of the 24-item human-agreement validation
  set (scored by `src/golden_questions_task.py`; logged under `logs/golden_questions_eval/`).
- `paper_notes/cut_diff_report__cuts_v2_all_confabulation.md` — effect of the 12 excluded
  items on the reported numbers.
- `humanebench/bootstrap.py` — bootstrap design (seed `20260407`, 1000 replicates) shared by
  the confidence intervals in the paper.

## How to verify

```bash
# Against the in-repo logs:
python scripts/verify_provenance.py

# Against a fresh Zenodo download (no repo git history needed for content checks):
python scripts/verify_provenance.py --logs-dir /path/to/extracted/logs
```

A clean tree prints `PROVENANCE VERIFIED: all checks passed.` and exits 0. Change a single
byte of any prompt in a log or in the dataset and it exits non-zero.
