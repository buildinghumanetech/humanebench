# Dataset composition: domain x principle

Computed on the **788-scenario analysis set** (all 800 reported separately). Each scenario carries exactly one principle and one domain.

## Principle-domain association

| statistic | value |
| --- | ---: |
| chi-square (77 df) | 1029.2 |
| p | 2.02e-166 |
| Cramer's V | 0.432 |

Cramer's V = **0.432** on a 0-1 scale. Principle and domain are strongly associated, which is by construction: the generation pipeline steered each principle toward the domains where it naturally arises.

## Per-principle domain concentration

| principle | n | domains | top domain | top-1 share | normalized HHI |
| --- | ---: | ---: | --- | ---: | ---: |
| respect-user-attention | 100 | 9 | everyday-decisions (45) | 45.0% | 0.198 |
| enable-meaningful-choices | 100 | 11 | financial-decisions (32) | 32.0% | 0.115 |
| enhance-human-capabilities | 100 | 8 | education (52) | 52.0% | 0.258 |
| protect-dignity-and-safety | 100 | 11 | technology-use (20) | 20.0% | 0.056 |
| foster-healthy-relationships | 100 | 11 | relationships (51) | 51.0% | 0.245 |
| prioritize-long-term-wellbeing | 100 | 10 | health-wellness (20) | 20.0% | 0.059 |
| be-transparent-and-honest | 98 | 5 | technology-use (71) | 72.4% | 0.513 |
| design-for-equity-and-inclusion | 90 | 11 | politics-society (15) | 16.7% | 0.049 |

The four most concentrated principles: be-transparent-and-honest is 72% technology-use; enhance-human-capabilities is 52% education; foster-healthy-relationships is 51% relationships; respect-user-attention is 45% everyday-decisions.

## Thin domains

| domain | n |
| --- | ---: |
| technology-use | 142 |
| relationships | 108 |
| everyday-decisions | 86 |
| education | 82 |
| career | 74 |
| health-wellness | 73 |
| financial-decisions | 63 |
| parenting | 59 |
| mental-health | 46 |
| politics-society | 44 |
| crisis-situations **(thin)** | 7 |
| moral-ambiguity **(thin)** | 4 |

**2 domains carry fewer than 10 scenarios** (crisis-situations = 7, moral-ambiguity = 4) and cannot support a domain-level claim of any kind.

## What this means for the per-principle results

Principle and domain are not separable in this design. Every scenario has exactly one of each, the assignment is far from uniform (Cramer's V = 0.432), and several principles draw the majority of their scenarios from a single domain. There is no cell structure that would let a model estimate a principle effect holding domain fixed.

The consequence for section 4.4 is specific and should be stated rather than hedged: **per-principle scores describe principle-domain bundles, not principles.** A claim that a given principle is the weakest is a claim about that principle *as instantiated in its domain mix*, and a different domain mix could reorder the table. This is a limitation of the construction, not an error in the numbers, and no reweighting available post hoc can remove it -- the confound is structural, not statistical.

## Vulnerable-population tags

| set | VP-tagged | total | groups |
| --- | ---: | ---: | ---: |
| 788 analysis set | **268** | 788 | 17 |
| all 800 | 278 | 800 | 17 |

Section 4.5 states "268 of the 788 scenarios in our dataset carry VP tags spanning 17 population groups". The analysis set actually has **268** VP-tagged scenarios across **17** groups.

The 800-set count is 278; 10 of the 12 excluded scenarios carry a VP tag, giving 278 - 10 = 268 on the analysis set.

Full VP x principle crosstab: `vp_principle_crosstab_788.csv`.

## Crosstab (788 analysis set)

| principle | career | crisis-situations | education | everyday-decisions | financial-decisions | health-wellness | mental-health | moral-ambiguity | parenting | politics-society | relationships | technology-use | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| respect-user-attention | 1 | 1 | 2 | 45 | 0 | 7 | 16 | 0 | 4 | 0 | 10 | 14 | 100 |
| enable-meaningful-choices | 24 | 1 | 6 | 7 | 32 | 10 | 1 | 0 | 6 | 4 | 2 | 7 | 100 |
| enhance-human-capabilities | 7 | 0 | 52 | 3 | 13 | 14 | 0 | 0 | 2 | 1 | 0 | 8 | 100 |
| protect-dignity-and-safety | 5 | 5 | 1 | 3 | 3 | 14 | 7 | 0 | 18 | 9 | 15 | 20 | 100 |
| foster-healthy-relationships | 5 | 0 | 2 | 4 | 3 | 2 | 6 | 1 | 19 | 4 | 51 | 3 | 100 |
| prioritize-long-term-wellbeing | 19 | 0 | 2 | 13 | 7 | 20 | 15 | 0 | 9 | 8 | 3 | 4 | 100 |
| be-transparent-and-honest | 0 | 0 | 4 | 5 | 0 | 0 | 0 | 0 | 0 | 3 | 15 | 71 | 98 |
| design-for-equity-and-inclusion | 13 | 0 | 13 | 6 | 5 | 6 | 1 | 3 | 1 | 15 | 12 | 15 | 90 |
| **total** | 74 | 7 | 82 | 86 | 63 | 73 | 46 | 4 | 59 | 44 | 108 | 142 | 788 |
