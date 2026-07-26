## Summary

- Re-augment `steerability_comparison.csv` (and the three per-persona score CSVs) with bootstrap CI columns via `scripts/compute_score_cis.py`. The merge from `main` into `bootstrap-confidence-intervals` had reverted these to the pre-CI shape, which silently no-op'd the whisker code in `create_steerability_chart.py`.
- Add small black vertical end-caps to each persona-endpoint CI whisker so the intervals read clearly even at small sizes.
- Add a paper-targeted variant `figures/steerability_candlestick_paper.*` sized for AAAI two-column `figure*` at `\textwidth`. Differences from the website-facing variant: title/subtitle stripped (LaTeX caption owns that text), legend moved below in two columns, baseline CIs omitted from the plot (all widths <0.06; visually clashed with the baseline dot's white halo at the smaller figsize). Persona-endpoint CIs are still drawn since their widths span up to 0.113 and carry the meaningful variance.
- Website-facing `figures/steerability_candlestick.*` and `..._compact.*` keep their existing chrome (title, subtitle, upper-left legend) and now also gain CI whiskers.

Stacked on top of #71 (`bootstrap-confidence-intervals`); merge that first.

## Test plan

- [ ] `python scripts/compute_score_cis.py` runs clean and patches `steerability_comparison.csv` with all six endpoint CI columns + the four delta CI columns
- [ ] `python scripts/create_steerability_chart.py` produces all three variants (default, compact, paper)
- [ ] Visual inspection of `figures/steerability_candlestick_paper.pdf`: every model row shows two whiskers (good + bad) with black caps at lo/hi, no baseline whisker, no in-figure title or subtitle, legend below in 2 columns
- [ ] Visual inspection of `figures/steerability_candlestick.png` (website variant): unchanged chrome; whiskers now visible

🤖 Generated with [Claude Code](https://claude.com/claude-code)
