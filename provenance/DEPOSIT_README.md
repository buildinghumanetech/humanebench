# HumaneBench evaluation logs — provenance deposit

This archive is the immutable, citable record of the raw evaluation logs behind the
reported HumaneBench numbers, plus everything needed to **verify offline that each run
scored the finalized dataset**. See the project for context:
https://github.com/buildinghumanetech/humanebench

## Contents

```
logs/baseline/<model>/*.eval          15 reported baseline runs
logs/good_persona/<model>/*.eval      15 reported good-persona runs
logs/bad_persona/<model>/*.eval       15 reported bad-persona runs
logs/golden_questions_eval/*.eval     human-agreement validation run (different dataset)
data/humane_bench.jsonl               the finalized 800-prompt dataset
provenance/MANIFEST.json              per-run record + content hashes
provenance/MANIFEST.md                human-readable summary
PROVENANCE.md                         full chain of custody + limitations
scripts/verify_provenance.py          independent verifier
scripts/build_provenance.py           manifest regenerator
humanebench/provenance.py             shared constants + hashing recipe
```

## Verify offline (no network, no git needed)

From the extracted archive root:

```bash
python3 scripts/verify_provenance.py --logs-dir logs
```

Expected: `PROVENANCE VERIFIED: all checks passed.` (exit 0). The git-history checks
report `SKIP` here because this archive has no `.git`; the content-binding checks — every
run's embedded prompts hashing to the frozen set, and the current dataset matching it —
fully pass on their own. Flip one byte of any prompt and the script exits non-zero.

The frozen prompt hash is
`e1af241db346e0299bb96c893b43746bff7a70d7a50efdd105ba9ded3cfe5d5f` (sha256 over the sorted
`id\x1f input\x1f target` triples). The prompt content froze at commit `9dc15bd`
(2025-11-16); all 45 reported runs executed 2025-11-17→23.
