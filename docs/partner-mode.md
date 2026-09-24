# Partner mode: re-judging external prompt/response datasets

How to run the HumaneBench judge panel on a partner's dataset of **stored
prompt/response pairs** (no generation — the stored response is replayed and
judged).

## Ground rules

- **Partner data never enters git.** Keep datasets, `.eval` logs, and results
  in a workspace outside the repo: `~/humanebench-partners/<partner>/`.
  `.eval` logs embed every prompt, response, and judge rationale — treat them
  as sensitive as the raw dataset.
- **Provider egress:** judging sends each prompt + response to the three
  judge-model providers via OpenRouter. Before any run, the OpenRouter
  account must be set to **exclude providers that may train on inputs**, and
  the partner must have signed off on third-party judging.
- Even anonymized messages can contain self-disclosed details (names,
  locations). Re-check verbatim quotes before circulating analysis documents.

## Dataset format

HumaneBench-style JSONL, one sample per (turn, principle):

```json
{"id": "<sample_id>__<principle-slug>",
 "input": "<user message>",
 "target": "<principle-slug>",
 "metadata": {"ai_output": "<the stored assistant response>"}}
```

`target` must be one of the 8 principle slugs in
`humanebench/humane_patterns.py`. `metadata.ai_output` is the contract key
read by the strict replay solver (`src/pregenerated_solver.py`) and must be a
non-empty string. `scripts/convert_partner_results.py` produces this format
from per-turn judged-results files, fanning each turn out across principles.

## Pipeline

1. **Curate** — `scripts/curate_production_pairs.py`: tags rows
   (synthetic/QA traffic, duplicate clusters, trivial labels, language)
   without dropping anything. Tags are analyst-visible only; judges never see
   them.
2. **Select** (optional, for comparison runs) —
   `scripts/select_comparison_subset.py`: deterministic stratified subset
   incl. a repeat slice for panel self-consistency. Requires curated input.
3. **Convert** — `scripts/convert_partner_results.py`: fan out to the format
   above. Turns with empty responses are skipped loudly.
4. **Judge** — run **without `--model`** (nothing can generate; the NoModel
   errors loudly if anything tries), dataset path **absolute**:

   ```
   inspect eval src/partner_rejudge_task.py \
       -T dataset=/absolute/path/to/converted.jsonl \
       --log-dir ~/humanebench-partners/<partner>/logs
   ```

   Pilot with `--limit 10` and verify cost before a full run.

5. **Between-run reliability** (comparison runs) — the selector also writes
   `<subset>_between_run.jsonl` (`__rep3` ids): convert it like the main
   subset and judge it as a **separate invocation on a different day**.
   Within-run repeats (`__rep2`) share one provider load regime and only
   lower-bound nondeterminism; the between-run file gives the honest
   re-run-drift number.

## Notes

- The benchmark tasks (`baseline`, `good_persona`, `bad_persona`) also accept
  `-T dataset=` for ad-hoc runs; there is deliberately **no environment
  variable override** — a stale variable could silently redirect a benchmark
  run.
- A HumaneScore computed over a partner dataset that does not cover all 8
  principles is **not comparable** to benchmark HumaneScores (empty
  principles are zero-filled into the denominator). Report per-principle
  results with their sample counts.
