# Human Friendliness Evaluation

An evaluation framework for testing AI assistant personas on human-friendly behavior using the AISI Inspect framework. The evaluation compares a "good" human-friendly persona against a "bad" engagement-maximizing persona across various scenarios.

## Prerequisites

- Python 3

## Setup

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install requirements

```bash
pip install -r requirements.txt
```

### 3. Install VSCode extensions (optional)

Install the following VSCode extensions:
- [Inspect AI](https://marketplace.visualstudio.com/items?itemName=ukaisi.inspect-ai) - For running and debugging Inspect evaluations
- [Data Wrangler](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.datawrangler) - For viewing and editing the dataset

## Running the Evaluation

To run both the good and bad persona evaluations:

```bash
rm -rf logs/* && inspect eval-set src/good_persona_task.py src/bad_persona_task.py --model openai/gpt-4o
```

### Model Options

You can specify different models using the format from the [Inspect AI providers documentation](https://inspect.aisi.org.uk/providers.html). Examples:

- `openai/gpt-4o`
- `anthropic/claude-3-5-sonnet-20241022`
- `google/gemini-2.5-pro`

## Dataset

The evaluation uses `data/simple_human_friendliness_dataset.jsonl`, which contains test scenarios categorized by the 8 core Humane Technology principles.

### Fixed Category System

All scenarios are categorized using one of the 8 core humane technology principles:

1. **Respect User Attention** - Technology should respect user attention as a finite, precious resource
2. **Enable Meaningful Choices** - Technology should empower users with meaningful choices and control
3. **Enhance Human Capabilities** - Technology should complement and enhance human capabilities, not replace or diminish them
4. **Protect Dignity and Safety** - Technology should protect human dignity, privacy, and safety
5. **Foster Healthy Relationships** - Technology should foster healthy relationships with devices, systems, and other people
6. **Prioritize Long-term Wellbeing** - Technology should prioritize long-term user wellbeing over short-term engagement metrics
7. **Be Transparent and Honest** - Technology should be transparent about its operations and honest about its capabilities
8. **Design for Equity and Inclusion** - Technology should be accessible and beneficial to diverse populations

### Dataset Fields

Each scenario in `data/simple_human_friendliness_dataset.jsonl` includes:

- `input`: The user's question or scenario
- `target`: The expected human-friendly response
- `category`: One of the 8 humane technology principles (enforced for consistency)
- `severity`: The importance level (low, medium, high, critical)
- `principle_to_evaluate`: The core evaluation principle being assessed

Of these fields, `input` and `target` are required. The others serve as metadata that helps the scorer evaluate adherence to the target.

### Generating New Scenarios

To generate additional scenarios with the fixed category system, see [data_generation/README.md](data_generation/README.md). The generation pipeline automatically:
- Enforces use of the 8 fixed principle categories only
- Normalizes category variations to standard categories
- Validates and rejects scenarios with non-standard categories

## Project Structure

```
├── src/
│   ├── good_persona_task.py    # Human-friendly persona evaluation
│   ├── bad_persona_task.py     # Engagement-maximizing persona evaluation
├── data/
│   └── simple_human_friendliness_dataset.jsonl  # Test scenarios
├── logs/                      # Evaluation results
```

## Results

Evaluation results are saved in the `logs/` directory with detailed scoring and analysis of how each persona performs across different human-friendliness principles. Inspect requires this directory to be empty before running again, so if you wish to save a run for comparison, you should copy it somewhere else first.

### Here is an video of Humane Tech member Jack Senechal running this inspect framework against OpenAI's GPT 4o vs. Claude Sonnet 3.5:

[![Inspect LLM Demo](https://p144.p3.n0.cdn.zight.com/items/6qupqLxX/293550a6-cea8-4cc4-bb0a-f7f6f530c577.png)](https://drodio.wistia.com/medias/njfoa1856w)

