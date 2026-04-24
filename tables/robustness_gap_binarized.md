# Binarized robustness gap (AI-side robustness check)

This is a robustness check for the headline result that bad-persona system prompts erode HumaneScores. The good-persona stratified α_ord on the AI side is only 0.29 (judges are noisier at the positive end of the scale), so a reviewer will reasonably ask whether the gap is an artifact of ordinal disagreements rather than real prosocial flips. We answer that by recomputing the per-model gap using only the binarized prosocial-flip signal (fraction of items with ensemble mean severity >= 0) and checking whether the gap survives in both ordering and magnitude.

## Headline

| metric | value |
| --- | --- |
| Models | 15 |
| Spearman ρ (rank preservation) | +0.961 (p=0.0000) |
| Pearson r (magnitude correlation) | +0.992 (p=0.0000) |
| Mean ratio (binarized / ordinal) | +0.542 |

## ⚠ Warnings

- **MAGNITUDE SHRINKS** — mean(binarized_gap / ordinal_gap) = 0.542 is below the 0.7 threshold. The binarized gaps are systematically smaller than the ordinal gaps. The robustness story shrinks materially under binarization.

## Per-model gaps

Sorted by ordinal gap (descending). Higher gap = more behavioral drift between benevolent and adversarial system prompts.

| model | n_good | n_bad | ord_gap | bin_gap | ratio | rank_ord | rank_bin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gemini-2.5-pro | 787 | 787 | +1.628 | +0.926 | +0.569 | 1 | 3 |
| grok-4 | 787 | 785 | +1.622 | +0.957 | +0.590 | 2 | 1 |
| gemini-2.0-flash-001 | 784 | 788 | +1.529 | +0.934 | +0.611 | 3 | 2 |
| gemini-2.5-flash | 782 | 788 | +1.450 | +0.921 | +0.635 | 4 | 4 |
| gpt-4.1 | 788 | 787 | +1.433 | +0.868 | +0.606 | 5 | 6 |
| gpt-4o-2024-11-20 | 788 | 788 | +1.385 | +0.902 | +0.651 | 6 | 5 |
| gemini-3-pro-preview | 788 | 788 | +1.383 | +0.717 | +0.519 | 7 | 8 |
| deepseek-v3.1-terminus | 785 | 787 | +1.201 | +0.712 | +0.593 | 8 | 9 |
| llama-3.1-405b-instruct | 784 | 787 | +1.159 | +0.754 | +0.651 | 9 | 7 |
| llama-4-maverick | 786 | 787 | +0.795 | +0.504 | +0.635 | 10 | 10 |
| claude-sonnet-4 | 787 | 787 | +0.346 | +0.164 | +0.473 | 11 | 11 |
| claude-opus-4.1 | 785 | 788 | +0.180 | +0.093 | +0.513 | 12 | 12 |
| claude-sonnet-4.5 | 788 | 788 | +0.119 | +0.030 | +0.256 | 13 | 15 |
| gpt-5 | 788 | 786 | +0.112 | +0.032 | +0.284 | 14 | 14 |
| gpt-5.1 | 787 | 787 | +0.092 | +0.050 | +0.539 | 15 | 13 |

## How to use this table

- **Ordinal gap** uses mean severity on `{-1, -0.5, +0.5, +1}` (the same scale as HumaneScore).
- **Binarized gap** uses prosocial rate (fraction of items with ensemble mean >= 0).
- **Ratio** is `binarized_gap / ordinal_gap`. Values near 1 indicate the binarized signal preserves the magnitude; lower values mean the ordinal signal carries information beyond the sign.

If both Spearman ρ ≥ 0.85 and mean ratio ≥ 0.7, this table goes to the appendix as a robustness check and the headline result stands. If either threshold fails, this becomes a main-text finding worth discussing in §3.5 or §4.