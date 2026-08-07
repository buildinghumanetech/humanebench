# Inter-principle cosine similarity matrix

**Embedding model:** `text-embedding-3-large` (OpenAI)
**Source:** `data/humane_bench.jsonl` (100 items per principle, 800 total)
**Methodology:** diagonal = upper-triangle mean cosine similarity within each
principle (self-pairs excluded, matches `qa_analysis.ipynb`); off-diagonal =
mean of the full 100×100 cross-similarity block between two principles.

## 8×8 matrix

| | bath | dei | emc | ehc | fhr | pltw | pds | rua |
|---|---|---|---|---|---|---|---|---|
| **bath** | **0.255** | 0.141 | 0.122 | 0.159 | 0.130 | 0.136 | 0.124 | 0.168 |
| **dei** | 0.141 | **0.177** | 0.120 | 0.157 | 0.137 | 0.130 | 0.129 | 0.121 |
| **emc** | 0.122 | 0.120 | **0.203** | 0.158 | 0.174 | 0.206 | 0.169 | 0.116 |
| **ehc** | 0.159 | 0.157 | 0.158 | **0.234** | 0.155 | 0.176 | 0.136 | 0.154 |
| **fhr** | 0.130 | 0.137 | 0.174 | 0.155 | **0.267** | 0.211 | 0.205 | 0.156 |
| **pltw** | 0.136 | 0.130 | 0.206 | 0.176 | 0.211 | **0.262** | 0.196 | 0.156 |
| **pds** | 0.124 | 0.129 | 0.169 | 0.136 | 0.205 | 0.196 | **0.233** | 0.120 |
| **rua** | 0.168 | 0.121 | 0.116 | 0.154 | 0.156 | 0.156 | 0.120 | **0.274** |

Short codes: `bath` = be-transparent-and-honest, `dei` = design-for-equity-and-inclusion,
`emc` = enable-meaningful-choices, `ehc` = enhance-human-capabilities,
`fhr` = foster-healthy-relationships, `pltw` = prioritize-long-term-wellbeing,
`pds` = protect-dignity-and-safety, `rua` = respect-user-attention.

## Summary stats

- **Overall mean intra** (diagonal): 0.238
- **Overall mean inter** (off-diagonal, 28 unique pairs): 0.152
- **Intra/inter ratio:** 1.57

### Top 3 highest inter-principle pairs
1. fhr ↔ pltw: 0.211
2. emc ↔ pltw: 0.206
3. fhr ↔ pds: 0.205

### Bottom 3 lowest inter-principle pairs
1. emc ↔ rua: 0.116
2. pds ↔ rua: 0.120
3. dei ↔ emc: 0.120

### Nearest other principle (per principle)

- **bath** nearest: rua (0.168)
- **dei** nearest: ehc (0.157)
- **emc** nearest: pltw (0.206)
- **ehc** nearest: pltw (0.176)
- **fhr** nearest: pltw (0.211)
- **pltw** nearest: fhr (0.211)
- **pds** nearest: fhr (0.205)
- **rua** nearest: bath (0.168)

### Farthest other principle (per principle)

- **bath** farthest: emc (0.122)
- **dei** farthest: emc (0.120)
- **emc** farthest: rua (0.116)
- **ehc** farthest: pds (0.136)
- **fhr** farthest: bath (0.130)
- **pltw** farthest: dei (0.130)
- **pds** farthest: rua (0.120)
- **rua** farthest: emc (0.116)

## bath / rua / ehc triangle

- sim(bath, rua) = **0.168**
- sim(bath, ehc) = **0.159**
- sim(rua, ehc) = **0.154**
- Corpus mean inter (all 28 pairs) = 0.152
- Intra: bath = 0.255, rua = 0.274, ehc = 0.234
