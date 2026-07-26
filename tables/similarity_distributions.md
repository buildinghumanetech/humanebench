# Pairwise similarity distributions and near-duplicate rate

Computed on **all 800** (800 scenarios, 319,600 unique pairs).

## Embedding-cache freshness proof

`data_generation/cache/principle_embeddings.npz` stores positional float arrays with no ids, texts, or hash, so its correspondence to the current dataset cannot be read off the file. It is proven from git instead.

| check | result |
| --- | --- |
| dataset commits postdating the cache (2026-04-09) | ef43b81 |
| prompt text unchanged | PASS |
| positional (id, input, target) sequence unchanged | PASS |

Both checks pass, so the cached embeddings map to current rows **including row order**: aggregate distributions and named near-duplicate pairs are both trustworthy.

## Pooled distributions

| model | set | mean | SD | p50 | p90 | p95 | p99 | p99.9 | p100 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all-MiniLM-L6-v2 | within-principle | 0.166 | 0.127 | 0.154 | 0.337 | 0.393 | 0.503 | 0.605 | 0.747 |
| all-MiniLM-L6-v2 | between-principle | 0.071 | 0.100 | 0.059 | 0.204 | 0.254 | 0.360 | 0.501 | 0.714 |
| text-embedding-3-large | within-principle | 0.238 | 0.103 | 0.228 | 0.373 | 0.424 | 0.529 | 0.653 | 0.740 |
| text-embedding-3-large | between-principle | 0.152 | 0.076 | 0.143 | 0.249 | 0.289 | 0.381 | 0.522 | 0.782 |

## Near-duplicate rate

| model | pairs >= 0.6 | pairs >= 0.8 | pairs >= 0.9 | rate at 0.60 |
| --- | ---: | ---: | ---: | ---: |
| all-MiniLM-L6-v2 | 66 | 0 | 0 | 0.0207% |
| text-embedding-3-large | 165 | 0 | 0 | 0.0516% |

0.60 is the threshold the construction pipeline deduplicated at (`data_generation/semantic_deduplication.py`), and it was set on MiniLM. A near-duplicate count is only interpretable as "did the dedup step work" against that model; the same threshold on text-embedding-3-large is a different scale and is reported only for comparison.

## Per-principle within-similarity

| model | principle | n | within mean | within SD | within max | between mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| all-MiniLM-L6-v2 | rua | 100 | 0.224 | 0.113 | 0.619 | 0.073 |
| all-MiniLM-L6-v2 | emc | 100 | 0.127 | 0.118 | 0.566 | 0.066 |
| all-MiniLM-L6-v2 | ehc | 100 | 0.114 | 0.100 | 0.535 | 0.063 |
| all-MiniLM-L6-v2 | pds | 100 | 0.170 | 0.128 | 0.594 | 0.073 |
| all-MiniLM-L6-v2 | fhr | 100 | 0.212 | 0.133 | 0.727 | 0.085 |
| all-MiniLM-L6-v2 | pltw | 100 | 0.179 | 0.125 | 0.721 | 0.088 |
| all-MiniLM-L6-v2 | bath | 100 | 0.202 | 0.120 | 0.747 | 0.062 |
| all-MiniLM-L6-v2 | dei | 100 | 0.100 | 0.112 | 0.655 | 0.060 |
| text-embedding-3-large | rua | 100 | 0.274 | 0.093 | 0.579 | 0.141 |
| text-embedding-3-large | emc | 100 | 0.203 | 0.093 | 0.548 | 0.152 |
| text-embedding-3-large | ehc | 100 | 0.234 | 0.077 | 0.502 | 0.156 |
| text-embedding-3-large | pds | 100 | 0.233 | 0.101 | 0.606 | 0.154 |
| text-embedding-3-large | fhr | 100 | 0.267 | 0.106 | 0.740 | 0.167 |
| text-embedding-3-large | pltw | 100 | 0.262 | 0.102 | 0.699 | 0.173 |
| text-embedding-3-large | bath | 100 | 0.255 | 0.100 | 0.636 | 0.140 |
| text-embedding-3-large | dei | 100 | 0.177 | 0.105 | 0.740 | 0.133 |

## Cross-model corroboration

Spearman correlation of the 8 within-principle means between `all-MiniLM-L6-v2` and `text-embedding-3-large`: **+0.905**. Agreement on the ordering is independent evidence that neither embedding source has drifted from the dataset.
