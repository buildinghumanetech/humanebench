# HELM Capabilities Integration

This module extracts benchmark data from Stanford CRFM's [HELM Capabilities](https://crfm.stanford.edu/helm/capabilities/latest/) leaderboard and calculates capability percentiles for AI models evaluated in HumaneBench.

## Purpose

Calculate where our 15 evaluated AI models rank in terms of technical capabilities (relative to all models in HELM), enabling analysis of the relationship between capability and humane behavior.

## Prerequisites

### Required

1. **Python 3.8+** with pip
2. **Google Cloud CLI (`gcloud`)** - Required for downloading HELM data

### Install gcloud CLI

**macOS (Homebrew):**
```bash
brew install google-cloud-sdk
```

**Linux/macOS (Official installer):**
```bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
```

**Windows:**
Download from https://cloud.google.com/sdk/docs/install

**Verify installation:**
```bash
gcloud --version
```

### Python Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `scipy>=1.10.0` - For percentile calculations
- `pandas>=2.0.0` - For data manipulation
- `requests>=2.31.0` - For HTTP requests

## Usage

### Step 1: Download HELM Data

Run the scraper to download and parse HELM Capabilities benchmark data from Google Cloud Storage:

```bash
python helm_scraper.py
```

**Options:**
- `--force-download` - Re-download data even if cached locally

**What it does:**
1. Checks if gcloud CLI is installed
2. Downloads HELM data from `gs://crfm-helm-public/capabilities/benchmark_output`
3. Parses benchmark scores for all models
4. Extracts scores for 5 benchmarks:
   - MMLU-Pro (general knowledge)
   - GPQA Diamond (reasoning)
   - IFEval (instruction following)
   - WildBench (dialogue)
   - Omni-MATH (mathematical reasoning)
5. Calculates mean scores (with WildBench rescaled from 1-10 to 0-1)

**Outputs:**
- `data/helm_raw_data.json` - All models with individual benchmark scores
- `data/helm_aggregate_scores.json` - All models with mean scores
- `data/gcs_cache/` - Cached raw HELM data

**Expected runtime:** 2-5 minutes (first run), <10 seconds (cached)

### Step 2: Map Your Models to HELM

Edit `model_mapping.json` to specify which HELM models correspond to your 15 target models:

```bash
# Open in your editor
nano model_mapping.json
# or
code model_mapping.json
```

**Instructions:**
1. Look at `data/helm_raw_data.json` to see exact HELM model names
2. For each of your 15 models, fill in the `helm_model_name` field with the exact name from HELM
3. Leave `helm_model_name` blank (`""`) if the model doesn't exist in HELM

**Example mapping:**
```json
{
  "your_model_name": "GPT-4o",
  "helm_model_name": "GPT-4o (2024-08-06)",
  "notes": "Exact match found"
}
```

**Recommended mappings based on your observation:**
- `GPT-4o` → Check for version like `GPT-4o (2024-XX-XX)`
- `Claude Sonnet 4.5` → Look for `Claude Sonnet 4.5` or similar
- `Gemini 2.5 Pro` → Look for exact match
- `Llama 3.1 405B` → Likely `Llama 3.1 Instruct Turbo 405B`
- `DeepSeek V3` → Likely `DeepSeek v3` (case difference)
- `Llama 4` → Consider `Llama 4 Maverick Instruct FP8` as close match
- `Claude Opus 4.1` → Consider `Claude 4 Opus` as close match

### Step 3: Calculate Percentiles

Run the percentile calculator:

```bash
python calculate_percentiles.py
```

**What it does:**
1. Loads your model mappings from `model_mapping.json`
2. Loads HELM aggregate scores
3. For each mapped model:
   - Finds its mean score in HELM data
   - Calculates percentile rank among ALL HELM models (0-100 scale)
   - Uses `scipy.stats.percentileofscore(kind='rank')` for proper tie handling
4. Generates coverage report and CSV output

**Outputs:**
- `output/model_coverage_report.txt` - Detailed report showing:
  - Which models were found ✓ vs missing ✗
  - Suggested HELM model name matches for missing models
  - Summary statistics
- `output/capability_percentiles.csv` - Final data for plotting with columns:
  - `model_name` - Your model name
  - `helm_model_name` - Corresponding HELM model name
  - `helm_aggregate_score` - Mean score across 5 benchmarks (0-1 scale)
  - `capability_percentile` - Percentile rank (0-100)
  - `in_helm` - Boolean, whether model exists in HELM

**Expected runtime:** <5 seconds

## Directory Structure

```
helm_integration/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── model_mapping.json             # Manual model name mappings (you edit this)
├── helm_scraper.py               # Downloads and parses HELM data
├── calculate_percentiles.py      # Calculates percentiles
├── data/
│   ├── helm_raw_data.json        # All HELM models with benchmark scores
│   ├── helm_aggregate_scores.json # All HELM models with mean scores
│   └── gcs_cache/                # Cached raw HELM data from GCS
└── output/
    ├── model_coverage_report.txt  # Coverage analysis
    └── capability_percentiles.csv # Final output for plotting
```

## File Formats

### model_mapping.json
```json
{
  "mappings": [
    {
      "your_model_name": "GPT-4o",
      "helm_model_name": "GPT-4o (2024-08-06)",
      "notes": "Optional notes about the mapping"
    }
  ],
  "instructions": "..."
}
```

### capability_percentiles.csv
```csv
model_name,helm_model_name,helm_aggregate_score,capability_percentile,in_helm
GPT-4o,GPT-4o (2024-08-06),0.7523,92.3,True
GPT-5.1,,,False
```

## Understanding Percentiles

**Percentile Calculation:**
- Uses `scipy.stats.percentileofscore(kind='rank')` method
- Percentile = what % of HELM models this model outperforms
- Scale: 0-100
  - 100th percentile = best model in HELM
  - 50th percentile = median model
  - 0th percentile = worst model in HELM

**Example:**
If a model scores at the 85th percentile, it means it performs better than 85% of all models evaluated in HELM Capabilities.

## Troubleshooting

### Error: "gcloud CLI is not installed"

**Solution:** Install gcloud CLI (see Prerequisites section)

### Error: "No stats.json files found in cache"

**Possible causes:**
1. GCS download failed
2. HELM data structure changed

**Solution:**
1. Run with `--force-download` to retry
2. Check gcloud authentication: `gcloud auth list`
3. Manually inspect `data/gcs_cache/` directory

### Warning: "Model not found in HELM scores"

**Cause:** The `helm_model_name` in your mapping doesn't exactly match any model in HELM data

**Solution:**
1. Check `data/helm_raw_data.json` for exact model names
2. Verify spelling, capitalization, and version numbers
3. Use fuzzy matches suggested in coverage report

### Low model coverage (few models found)

**Expected:** Based on current HELM Capabilities, expect ~9-13 of 15 models to be found

**Models likely missing:**
- GPT-5.1 (too new)
- Gemini 3 Pro Preview (pre-release)

**Solution:** This is expected. The coverage report will clearly indicate which models are missing.

## Technical Details

### HELM Benchmarks

| Benchmark | Category | Instances | Date | Scale |
|-----------|----------|-----------|------|-------|
| MMLU-Pro | General Knowledge | 1,000 | June 2024 | 0-1 |
| GPQA Diamond | Reasoning | 448 | Nov 2023 | 0-1 |
| IFEval | Instruction Following | 541 | Nov 2023 | 0-1 |
| WildBench | Dialogue | 1,000 | June 2024 | 1-10* |
| Omni-MATH | Math Reasoning | 1,000 | Oct 2024 | 0-1 |

*WildBench is rescaled from 1-10 to 0-1 before calculating mean scores.

### Score Aggregation

HELM calculates a mean score across the 5 benchmarks:

```python
mean_score = (mmlu_pro + gpqa + ifeval + wildbench_rescaled + omni_math) / 5

# Where wildbench_rescaled = (wildbench - 1.0) / 9.0
```

### Percentile Method

Using `kind='rank'` ensures proper handling of tied scores:

```python
from scipy.stats import percentileofscore

percentile = percentileofscore(
    all_helm_scores,  # All model scores in HELM
    model_score,       # Your model's score
    kind='rank'        # Handles ties properly
)
```

## Data Sources

- **HELM Capabilities Leaderboard:** https://crfm.stanford.edu/helm/capabilities/latest/
- **GCS Bucket:** `gs://crfm-helm-public/capabilities/benchmark_output`
- **HELM Documentation:** https://crfm-helm.readthedocs.io/

## Next Steps

After generating `output/capability_percentiles.csv`:

1. **Merge with HumaneBench scores** - Combine capability percentiles with humaneness scores from your evaluations
2. **Plot capability vs. humaneness** - Create scatter plots to visualize the relationship
3. **Statistical analysis** - Test for correlations between capability and humane behavior

Example merge (in parent directory):
```python
import pandas as pd

# Load capability percentiles
capability = pd.read_csv('helm_integration/output/capability_percentiles.csv')

# Load humaneness scores (from your HumaneBench results)
humaneness = pd.read_csv('good_persona_scores.csv')  # Adjust path as needed

# Merge on model_name
combined = capability.merge(humaneness, on='model_name', how='inner')

# Plot
import matplotlib.pyplot as plt
plt.scatter(combined['capability_percentile'], combined['HumaneScore'])
plt.xlabel('Capability Percentile (HELM)')
plt.ylabel('HumaneScore (HumaneBench)')
plt.title('Capability vs. Humaneness')
plt.savefig('figures/capability_vs_humaneness.png')
```

## Notes and Limitations

1. **Model coverage:** Not all models evaluated in HumaneBench exist in HELM, especially newer models (GPT-5.1, Gemini 3, etc.)

2. **Version matching:** HELM may use different version numbers (e.g., "Claude 3.7 Sonnet" vs "Claude Sonnet 4.5"). Manual mapping required.

3. **Data freshness:** HELM Capabilities is periodically updated. Re-run scraper to get latest data.

4. **Benchmark alignment:** HELM measures technical capabilities (knowledge, reasoning), not humaneness. These are complementary dimensions.

5. **Caching:** Raw HELM data is cached locally in `data/gcs_cache/`. Delete this directory or use `--force-download` to refresh.

## Support

For issues with:
- **This integration:** Check Troubleshooting section above
- **HELM data:** https://github.com/stanford-crfm/helm/issues
- **gcloud CLI:** https://cloud.google.com/sdk/docs

## License

This integration tool is part of the HumaneBench project. HELM data is provided by Stanford CRFM under their respective license.
