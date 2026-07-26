# Serving provenance: which stack answered each call

A model slug is not a system. OpenRouter routes a slug to one of several upstream providers, and for open-weight models those providers differ in quantisation and serving configuration. The logs record the slug; the provider that actually answered is recoverable only from the raw API response, which is what this table reads.

Scanned **45 runs** across conditions: baseline, good_persona, bad_persona.

## Generation calls, by evaluated model

| model | providers that served it | distinct | `system_fingerprint` |
| --- | --- | ---: | --- |
| claude-opus-4.1 | Google 90%, Anthropic 7%, Amazon Bedrock 3% | 3 | (none) 100% |
| claude-sonnet-4 | Google 100% | 1 | (none) 100% |
| claude-sonnet-4.5 | Google 100% | 1 | (none) 100% |
| deepseek-v3.1-terminus | DeepInfra 30%, Novita 28%, SiliconFlow 25%, AtlasCloud 16%, SambaNova 0% | 5 | (none) 75%,  25%, fastcoe 0% |
| gemini-2.0-flash-001 | Google AI Studio 69%, Google 31% | 2 | (none) 100% |
| gemini-2.5-flash | Google 100% | 1 | (none) 100% |
| gemini-2.5-pro | Google 100%, Google AI Studio 0% | 2 | (none) 100% |
| gemini-3-pro-preview | Google AI Studio 64%, Google 36% | 2 | (none) 100% |
| gpt-4.1 | OpenAI 100% | 1 | (none) 100% |
| gpt-4o-2024-11-20 | OpenAI 100% | 1 | fp_c082851c08 88%, fp_b0b25f0bce 12% |
| gpt-5 | OpenAI 100% | 1 | (none) 100% |
| gpt-5.1 | OpenAI 100% | 1 | (none) 100% |
| grok-4 | xAI 100% | 1 | (none) 100% |
| llama-3.1-405b-instruct | Google 48%, Together 34%, Hyperbolic 18% | 3 |  66%, (none) 34% |
| llama-4-maverick | Friendli 24%, DeepInfra 23%, Novita 20%, Google 18%, Together 9%, Groq 2%, +2 more | 8 | (none) 67%,  29%, fastcoe 2%, +2 more |

**7 of 15 models were served by more than one provider.** Those cells are a mixture of serving stacks rather than a single system. This is a property of the reported runs, not something introduced later, and it is visible to anyone who opens the released logs.

- `llama-4-maverick`: 8 providers — Friendli 24%, DeepInfra 23%, Novita 20%, Google 18%, Together 9%, Groq 2%, +2 more
- `deepseek-v3.1-terminus`: 5 providers — DeepInfra 30%, Novita 28%, SiliconFlow 25%, AtlasCloud 16%, SambaNova 0%
- `claude-opus-4.1`: 3 providers — Google 90%, Anthropic 7%, Amazon Bedrock 3%
- `llama-3.1-405b-instruct`: 3 providers — Google 48%, Together 34%, Hyperbolic 18%
- `gemini-2.0-flash-001`: 2 providers — Google AI Studio 69%, Google 31%
- `gemini-2.5-pro`: 2 providers — Google 100%, Google AI Studio 0%
- `gemini-3-pro-preview`: 2 providers — Google AI Studio 64%, Google 36%

## Judge calls

The judge ensemble is routed the same way. This exposure is identical across conditions, so it does not bias a persona contrast, but it does bound how exactly any single judge is reproducible.

| judge <- provider | share |
| --- | ---: |
| openrouter/anthropic/claude-4.5-sonnet <- Google | 35.0% |
| openrouter/google/gemini-2.5-pro <- Google | 32.5% |
| openrouter/openai/gpt-5.1 <- OpenAI | 32.4% |
| openrouter/google/gemini-2.5-pro <- Google AI Studio | 0.1% |
| openrouter/anthropic/claude-4.5-sonnet <- Amazon Bedrock | 0.0% |

Judge `system_fingerprint`: (none) 100%
