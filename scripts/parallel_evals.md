python scripts/run_parallel_evals.py --max-workers 12 --task-types baseline --models openrouter/openai/o3 openrouter/openai/gpt-5 openrouter/openai/gpt-4o-2024-11-20 openrouter/anthropic/claude-sonnet-4.5 openrouter/anthropic/claude-opus-4.1 openrouter/google/gemini-2.5-flash openrouter/google/gemini-2.5-pro openrouter/meta-llama/llama-4-maverick openrouter/nousresearch/hermes-3-llama-3.1-405b openrouter/mistralai/mistral-medium-3.1 openrouter/deepseek/deepseek-chat-v3.1 openrouter/qwen/qwen3-max

python scripts/run_parallel_evals.py --max-workers 4 --task-types baseline bad_persona good_persona --models openrouter/openai/gpt-4o openrouter/google/gemini-2.5-flash

# Upload
python scripts/inspect_logs_to_langfuse.py --principle respect-user-attention --sample-limit 100 --enqueue