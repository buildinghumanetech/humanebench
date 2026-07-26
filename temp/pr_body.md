## Summary

- Appends a new **section 7: EXTENDED: POST-EXCLUSION COVERAGE** to `data_generation/analyze_coverage.py`, showing full → filtered counts with deltas for principles, vulnerable populations, and domains.
- Sources excluded IDs from `humanebench.excluded.load_excluded_ids()` (the canonical read path for `metadata.excluded_from_analysis`), so the report stays in sync with the dataset rather than hardcoding the 12 v2 confabulation cuts.
- Existing sections 1–6 + summary are unchanged, so paper-cited numbers remain visible alongside the post-cut view.

Output highlights at a glance for the current dataset:
- 12 items excluded → 788/800 kept (DEI 100→90, BATH 100→98).
- VP impact: people-with-disabilities 30→24 (-6), non-native-speakers 25→22 (-3), neurodivergent-people 8→7 (-1).
- Domain impact concentrated in technology-use (-6) and education (-5).

## Test plan

- [x] `python data_generation/analyze_coverage.py data/humane_bench.jsonl` — sections 1–6 unchanged; section 7 prints with expected deltas matching `paper_notes/cut_lists/cuts_v2_all_confabulation.txt`.
- [ ] Reviewer spot-check: deltas in the principle/VP/domain tables are internally consistent (each table's column-sum delta equals -12).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
