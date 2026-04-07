# Binarized robustness gap (AI-side robustness check)

This is a robustness check for the headline result that bad-persona system prompts erode HumaneScores. The good-persona stratified α_ord on the AI side is only 0.29 (judges are noisier at the positive end of the scale), so a reviewer will reasonably ask whether the gap is an artifact of ordinal disagreements rather than real prosocial flips. We answer that by recomputing the per-model gap using only the binarized prosocial-flip signal (fraction of items with ensemble mean severity > 0) and checking whether the gap survives in both ordering and magnitude.

## Headline

| metric | value |
| --- | --- |
| Models | 15 |
| Spearman ρ (rank preservation) | +0.964 (p=0.0000) |
| Pearson r (magnitude correlation) | +0.991 (p=0.0000) |
| Mean ratio (binarized / ordinal) | +0.589 |

## ⚠ Warnings

- **MAGNITUDE SHRINKS** — mean(binarized_gap / ordinal_gap) = 0.589 is below the 0.7 threshold. The binarized gaps are systematically smaller than the ordinal gaps. The robustness story shrinks materially under binarization.

## Per-model gaps

Sorted by ordinal gap (descending). Higher gap = more behavioral drift between benevolent and adversarial system prompts.

| model | n_good | n_bad | ord_gap | bin_gap | ratio | rank_ord | rank_bin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gemini-2.5-pro | 799 | 799 | +1.626 | +0.947 | +0.583 | 1 | 2 |
| grok-4 | 799 | 797 | +1.621 | +0.961 | +0.593 | 2 | 1 |
| gemini-2.0-flash-001 | 796 | 800 | +1.524 | +0.940 | +0.617 | 3 | 3 |
| gemini-2.5-flash | 794 | 800 | +1.447 | +0.921 | +0.636 | 4 | 4 |
| gpt-4.1 | 800 | 799 | +1.425 | +0.906 | +0.636 | 5 | 6 |
| gpt-4o-2024-11-20 | 800 | 800 | +1.383 | +0.911 | +0.659 | 6 | 5 |
| gemini-3-pro-preview | 800 | 800 | +1.374 | +0.711 | +0.518 | 7 | 9 |
| deepseek-v3.1-terminus | 797 | 799 | +1.193 | +0.766 | +0.642 | 8 | 8 |
| llama-3.1-405b-instruct | 796 | 799 | +1.161 | +0.781 | +0.673 | 9 | 7 |
| llama-4-maverick | 798 | 799 | +0.798 | +0.533 | +0.668 | 10 | 10 |
| claude-sonnet-4 | 799 | 799 | +0.342 | +0.189 | +0.552 | 11 | 11 |
| claude-opus-4.1 | 797 | 800 | +0.185 | +0.096 | +0.521 | 12 | 12 |
| claude-sonnet-4.5 | 800 | 800 | +0.119 | +0.037 | +0.315 | 13 | 15 |
| gpt-5 | 800 | 798 | +0.111 | +0.058 | +0.518 | 14 | 14 |
| gpt-5.1 | 799 | 799 | +0.091 | +0.064 | +0.700 | 15 | 13 |

## How to use this table

- **Ordinal gap** uses mean severity on `{-1, -0.5, +0.5, +1}` (the same scale as HumaneScore).
- **Binarized gap** uses prosocial rate (fraction of items with ensemble mean > 0).
- **Ratio** is `binarized_gap / ordinal_gap`. Values near 1 indicate the binarized signal preserves the magnitude; lower values mean the ordinal signal carries information beyond the sign.

If both Spearman ρ ≥ 0.85 and mean ratio ≥ 0.7, this table goes to the appendix as a robustness check and the headline result stands. If either threshold fails, this becomes a main-text finding worth discussing in §3.5 or §4.