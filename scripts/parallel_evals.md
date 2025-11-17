python scripts/run_parallel_evals.py --max-workers 12 --task-types baseline bad_persona good_persona --models openrouter/openai/gpt-4.1 openrouter/openai/gpt-4o-2024-11-20 openrouter/openai/gpt-5 openrouter/anthropic/claude-sonnet-4 openrouter/anthropic/claude-opus-4.1 openrouter/anthropic/claude-sonnet-4.5 openrouter/google/gemini-2.0-flash-001 openrouter/google/gemini-2.5-flash openrouter/google/gemini-2.5-pro openrouter/meta-llama/llama-3.1-405b-instruct openrouter/meta-llama/llama-4-maverick openrouter/deepseek/deepseek-v3.1-terminus openrouter/x-ai/grok-4

python scripts/run_parallel_evals.py --max-workers 12 --task-types test --models openrouter/openai/gpt-4.1 openrouter/openai/gpt-4o-2024-11-20 openrouter/openai/gpt-5 openrouter/anthropic/claude-sonnet-4 openrouter/anthropic/claude-opus-4.1 openrouter/anthropic/claude-sonnet-4.5 openrouter/google/gemini-2.0-flash-001 openrouter/google/gemini-2.5-flash openrouter/google/gemini-2.5-pro openrouter/meta-llama/llama-3.1-405b-instruct openrouter/meta-llama/llama-4-maverick openrouter/deepseek/deepseek-v3.1-terminus openrouter/x-ai/grok-4

python scripts/run_parallel_evals.py --max-workers 4 --task-types baseline bad_persona good_persona --models openrouter/openai/gpt-4o openrouter/google/gemini-2.5-flash

# Upload
python scripts/inspect_logs_to_langfuse.py --principle respect-user-attention --sample-limit 100 --enqueue