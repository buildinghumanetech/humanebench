# Judge-artifact controls

Two separate questions. The harness can be blind while the response still reveals its condition; only the first is a property we control.

## A. Structural blinding of the judge prompt — PASS

- Judge prompts inspected: **360** (8 per run x 45 persona x model cells).

- Message roles sent to the judge: `['system|user']` — the evaluated model's **system message is never included**.

- Roles inside the scored conversation block: `['human|model']`.

- Judge-prompt **scaffolding** containing any persona / condition / task-name probe: **0**.

Probes are applied to the scaffolding only — the rubric, global rules, severity scale and response contract — with the scored conversation block excluded. A probe firing inside the conversation is the model disclosing its own instructions, which is measured separately in (B); counting it here would conflate a harness property with a model behaviour.

The judge sees the principle rubric, the global rules, the severity scale, the user prompt and the model response — and nothing that identifies which system-prompt condition produced it. The paper can state blinding as verified rather than assumed.

## B. Behavioural leakage in the response — REAL, and heterogeneous

| condition | responses | disclose | rate |
| --- | ---: | ---: | ---: |
| baseline | 11,814 | 264 | 2.23% |
| good_persona | 11,794 | 320 | 2.71% |
| bad_persona | 11,808 | 1,810 | 15.33% |

A raw rate conflates two opposite behaviours, so disclosures are split by the item's own ensemble score: a model that names the instruction in order to **refuse** it scores positively, while a model that narrates the objective while **pursuing** it scores negatively.

| condition | disclosures scoring negative (compliance narration) | disclosures scoring positive (pushback) |
| --- | ---: | ---: |
| bad_persona | 962 | 848 |
| baseline | 23 | 241 |
| good_persona | 5 | 315 |

### Bad persona, by model

| model | disclosure rate | share of those scoring positive | mean response chars |
| --- | ---: | ---: | ---: |
| gemini-3-pro-preview | 75.89% | 36% | 4101 |
| gemini-2.5-pro | 50.70% | 5% | 3660 |
| claude-sonnet-4.5 | 40.86% | 95% | 1387 |
| claude-opus-4.1 | 22.72% | 98% | 1259 |
| grok-4 | 13.63% | 2% | 2159 |
| gpt-5 | 7.38% | 98% | 5242 |
| claude-sonnet-4 | 6.35% | 90% | 1197 |
| llama-3.1-405b-instruct | 3.43% | 11% | 1049 |
| gpt-5.1 | 3.05% | 96% | 4349 |
| llama-4-maverick | 2.67% | 24% | 1253 |
| deepseek-v3.1-terminus | 1.02% | 0% | 1122 |
| gpt-4o-2024-11-20 | 0.76% | 0% | 1021 |
| gpt-4.1 | 0.64% | 0% | 1011 |
| gemini-2.0-flash-001 | 0.63% | 0% | 1068 |
| gemini-2.5-flash | 0.13% | 0% | 1009 |

The spread is the finding: disclosure is not a constant property of the condition but a model-specific behaviour. Any persona contrast is therefore partly a disclosure contrast, and a new adversarial condition must have this measured on both arms before its delta is attributed to humaneness alone.

## What this does and does not license

- **Does:** the paper may state that judges were blind to condition by construction, and quantify the residual channel by which condition can still be inferred.
- **Does not:** it does not establish that disclosure *causes* score differences. Testing that needs a disclosure-matched sensitivity analysis (compare scores on disclosing vs non-disclosing responses within model and principle), which is not run here.
