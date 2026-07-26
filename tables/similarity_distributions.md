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
| text-embedding-3-large | within-principle | 0.238 | 0.103 | 0.228 | 0.373 | 0.424 | 0.529 | 0.653 | 0.740 |
| text-embedding-3-large | between-principle | 0.152 | 0.076 | 0.143 | 0.249 | 0.289 | 0.381 | 0.522 | 0.782 |

## Near-duplicate rate

| model | pairs >= 0.6 | pairs >= 0.8 | pairs >= 0.9 | rate at 0.60 |
| --- | ---: | ---: | ---: | ---: |
| text-embedding-3-large | 165 | 0 | 0 | 0.0516% |

0.60 is the threshold the construction pipeline deduplicated at (`data_generation/semantic_deduplication.py`), and it was set on MiniLM. A near-duplicate count is only interpretable as "did the dedup step work" against that model; the same threshold on text-embedding-3-large is a different scale and is reported only for comparison.

## Per-principle within-similarity

| model | principle | n | within mean | within SD | within max | between mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| text-embedding-3-large | rua | 100 | 0.274 | 0.093 | 0.579 | 0.141 |
| text-embedding-3-large | emc | 100 | 0.203 | 0.093 | 0.548 | 0.152 |
| text-embedding-3-large | ehc | 100 | 0.234 | 0.077 | 0.502 | 0.156 |
| text-embedding-3-large | pds | 100 | 0.233 | 0.101 | 0.606 | 0.154 |
| text-embedding-3-large | fhr | 100 | 0.267 | 0.106 | 0.740 | 0.167 |
| text-embedding-3-large | pltw | 100 | 0.262 | 0.102 | 0.699 | 0.173 |
| text-embedding-3-large | bath | 100 | 0.255 | 0.100 | 0.636 | 0.140 |
| text-embedding-3-large | dei | 100 | 0.177 | 0.105 | 0.740 | 0.133 |

## Not computed

- **MiniLM**: ModuleNotFoundError: No module named 'huggingface_hub.utils._git_credential'
