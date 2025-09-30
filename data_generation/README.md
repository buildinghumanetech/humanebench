# Humane Tech Benchmark Data Generation Pipeline

This pipeline generates high-quality evaluation scenarios for testing AI assistants' adherence to humane technology principles.

## Features

- **LLM-Assisted Generation**: Uses Claude Sonnet 4 (OpenRouter) for creative scenario generation
- **Intelligent Validation**: Uses Claude Sonnet 4 (OpenRouter) with web search for quality assessment
- **API Fallback**: OpenRouter first, falls back to Cerebras direct API for free tier usage
- **Semantic Deduplication**: Prevents duplicate scenarios using sentence transformers (60% similarity threshold, configurable)
- **Interactive Mode**: Generate in batches with user feedback between iterations
- **Principle-Aware**: Prioritizes existing evaluation categories while maintaining diversity

⚠️ **Important**: Web search for validation is only available through OpenRouter. Cerebras fallback provides a generous free tier but without web research capabilities.

## Setup

1. **Install Dependencies**
   ```bash
   cd data_generation
   python -m venv venv
   source venv/bin/activate # or `. venv\Scripts\activate` on Windows
   pip install -r requirements.txt
   ```

   **Note**: The pipeline requires `cerebras-cloud-sdk` for fallback API support. This is included in the requirements.

2. **Set API Keys**
   ```bash
   # Copy template and add your API keys
   cp .env.template /.env
   # Edit .env file with your OpenRouter and/or Cerebras API keys
   ```

3. **Get OpenRouter API Key**
   - Sign up at [OpenRouter](https://openrouter.ai/)
   - Generate an API key
   - Add it to your `.env` file
   - Optionally add a Cerebras API key in case you run out of OpenRouter credits

## Usage

### Interactive Mode
```bash
cd data_generation
source venv/bin/activate
python pipeline.py
```

This will:
- Generate 75 scenarios per batch (default)
- Validate each scenario for quality
- Filter semantic duplicates
- Append unique scenarios to your CSV
- Wait for your input between batches

### Automated mode
Setting an integer value for `TARGET_ROWS` in `.env` will skip waiting for input until the target number of rows has been added to the dataset.

## Pipeline Flow

```
User Input → Generation (Claude) → Validation (Grok+Web) → Deduplication → CSV Append
     ↑                                                                             ↓
     └──────────────── User feedback/context injection ←─────────────────────────┘
```

## Evaluation Categories

The pipeline prioritizes these existing evaluation categories:

- **Fostering Meaningful Connection**
- **Psychological Safety and Emotional Well-Being**
- **User Autonomy and Freedom from Manipulation**
- **Transparency and Explainability**
- **Trustworthiness and Reliability**
- **Fairness and Inclusivity**
- **Privacy and Data Rights**
- **Accountability and Human Oversight**
- **Ethical Alignment and Moral Integrity**
- **Human Well-Being and Flourishing**
- **Crisis Response and Emergency Situations**

## Configuration

All pipeline settings are configured in `config.py` with sensible defaults:

**Model Configuration:**
- `OPENROUTER_GENERATION_MODEL`: Claude Sonnet 4
- `OPENROUTER_VALIDATION_MODEL`: Claude Sonnet 4
- `CEREBRAS_GENERATION_MODEL`: Qwen-3-235B-A22B-Thinking (fallback)
- `CEREBRAS_VALIDATION_MODEL`: Llama-4-Maverick-17B (fallback)

**Pipeline Settings:**
- `DEFAULT_BATCH_SIZE`: 75 scenarios per batch
- `TEMPERATURE`: 0.88 for generation, 0.3 for validation
- `SIMILARITY_THRESHOLD`: 0.60 for deduplication (configurable per DataManager instance)
- `GENERATION_MAX_TOKENS`: 8000, `VALIDATION_MAX_TOKENS`: 2000
- `VALIDATION_SAMPLE_PERCENTAGE`: 20% for efficiency
- `CONTEXT_ANALYSIS_FREQUENCY`: 1 (analyze context every batch)
- `ENABLE_DATASET_CONTEXT`: True (context-aware generation)
- `ENABLE_DEDUPLICATION_FEEDBACK`: True (improve uniqueness)

**Environment Variables:**
- `TARGET_ROWS` can be set via `.env` for automated runs that don't stop until a set number of rows is generated
- API keys are set in `.env`
- All other settings are edited directly in `config.py`

## Output Format

Each scenario includes:
- `input`: Human's question/situation
- `target`: Ideal humane response approach
- `category`: Situation type (e.g., "Mental Health & Crisis")
- `severity`: Impact level (low/medium/high/critical)
- `principle_to_evaluate`: Evaluation category from list above

## Quality Assurance

The validation engine evaluates scenarios on:
1. **Realism & Relevance** (0-25 points)
2. **Evaluation Utility** (0-25 points)
3. **Principle Alignment** (0-25 points)
4. **Diversity & Coverage** (0-25 points)

Scenarios scoring 65+ are approved for inclusion.

### Semantic Deduplication

The pipeline uses sentence transformers (`all-MiniLM-L6-v2` by default) to prevent duplicate scenarios:
- Default similarity threshold: 60%
- Alternative embedding model: `text-embedding-3-large` (configurable)
- Deduplication runs against all existing scenarios in the dataset

## Commands During Interactive Mode

- **Press Enter**: Continue with current settings
- **Type context**: Provide guidance for next batch (e.g., "more scenarios for teenagers")
- **Type "STOP GENERATION"**: End the session

## Troubleshooting

- **API Errors**: Check your OpenRouter API key and credit balance
- **Import Errors**: Ensure all requirements are installed
- **Validation Failures**: The validator uses web search; network issues may cause failures
- **Memory Issues**: Large batches may require more RAM for embeddings

## Files

- `pipeline.py`: Main controller script
- `generators.py`: Scenario generation engine with web search capabilities
- `validators.py`: Quality validation engine with fallback API support
- `data_manager.py`: CSV handling, statistics, and deduplication integration
- `semantic_deduplication.py`: Similarity-based duplicate detection using transformers
- `llm_client.py`: Unified client with OpenRouter/Cerebras fallback
- `config.py`: Configuration settings (edit directly for customization)
- `.env.template`: API key template (copy to `.env`)

