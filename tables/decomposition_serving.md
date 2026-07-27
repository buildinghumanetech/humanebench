# Serving provenance: which stack answered each call

A model slug is not a system. OpenRouter routes a slug to one of several upstream providers, and for open-weight models those providers differ in quantisation and serving configuration. The logs record the slug; the provider that actually answered is recoverable only from the raw API response, which is what this table reads.

Scanned **44 runs** across conditions: decomp_b_xml_objective, decomp_c_prose, decomp_d_okr, decomp_e_abtest.

## Generation calls, by evaluated model

| model | providers that served it | distinct | `system_fingerprint` |
| --- | --- | ---: | --- |
| claude-opus-4.1 | Amazon Bedrock 100%, Google 0% | 2 | (none) 100% |
| claude-sonnet-4 | Amazon Bedrock 100%, Google 0% | 2 | (none) 100% |
| claude-sonnet-4.5 | Amazon Bedrock 100% | 1 | (none) 100% |
| deepseek-v3.1-terminus | Novita 26%, SiliconFlow 25%, AtlasCloud 20%, StreamLake 17%, DeepInfra 13% | 5 | (none) 100% |
| gemini-2.5-flash | Google 100% | 1 | (none) 100% |
| gemini-2.5-pro | Google 100% | 1 | (none) 100% |
| gpt-4.1 | OpenAI 100%, Azure 0% | 2 | (none) 100% |
| gpt-4o-2024-11-20 | OpenAI 100% | 1 | fp_66d67c98c9 66%, fp_17f85970eb 8%, fp_2d3f281051 7%, +4 more |
| gpt-5 | OpenAI 100%, Azure 0% | 2 | (none) 100% |
| gpt-5.1 | OpenAI 100%, Azure 0% | 2 | (none) 100% |
| llama-4-maverick | DeepInfra 34%, DigitalOcean 23%, Novita 19%, Parasail 13%, Google 11% | 5 | (none) 100% |

**2 of 11 models were served by more than one provider** at a share of at least 0.5%. Those cells are a mixture of serving stacks rather than a single system. This is a property of the runs scanned here (decomp_b_xml_objective, decomp_c_prose, decomp_d_okr, decomp_e_abtest), not something introduced later, and it is visible to anyone who opens the released logs.

- `deepseek-v3.1-terminus`: 5 providers — Novita 26%, SiliconFlow 25%, AtlasCloud 20%, StreamLake 17%, DeepInfra 13%
- `llama-4-maverick`: 5 providers — DeepInfra 34%, DigitalOcean 23%, Novita 19%, Parasail 13%, Google 11%

Excluded from that count: `claude-opus-4.1`, `claude-sonnet-4`, `gpt-4.1`, `gpt-5`, `gpt-5.1` — a second provider appears but serves under 0.5% of calls. Recorded here rather than silently folded either way.

## Judge calls

The judge ensemble is routed the same way. This exposure is identical across conditions, so it does not bias a persona contrast, but it does bound how exactly any single judge is reproducible.

| judge <- provider | share |
| --- | ---: |
| openrouter/google/gemini-2.5-pro <- Google | 33.5% |
| openrouter/anthropic/claude-sonnet-4.5 <- Amazon Bedrock | 33.3% |
| openrouter/openai/gpt-5.1 <- OpenAI | 33.1% |
| openrouter/anthropic/claude-sonnet-4.5 <- Google | 0.0% |
| openrouter/openai/gpt-5.1 <- Azure | 0.0% |

Judge `system_fingerprint`: (none) 100%
