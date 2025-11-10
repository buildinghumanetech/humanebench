# HumaneBench Data Analysis Pipeline

Two scripts for analyzing your pilot study:
1. **`consolidate_ratings.py`** - Combines spreadsheet + Langfuse data
2. **`analyze_ratings.py`** - Calculates IRR and descriptive statistics

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Create `.env` file** in the same directory as the script:
   ```env
   LANGFUSE_PUBLIC_KEY=pk-your-public-key
   LANGFUSE_SECRET_KEY=sk-your-secret-key
   LANGFUSE_HOST=https://your-langfuse-instance.com
   ```

3. **Edit paths in script:**
   Open `consolidate_ratings.py` and update these lines:
   ```python
   SPREADSHEET_DIR = "./spreadsheets"  # Where your .xlsx files are
   OUTPUT_DIR = "./output"              # Where to save results
   ```

## File Organization

Your spreadsheets should be organized like:
```
spreadsheets/
├── John Smith_annotations.xlsx
├── Jane Doe_ratings.xlsx
└── Alex_template.xlsx
```

**Note:** First name (first word before space) is used as rater identifier.

## Running the Script

```bash
python consolidate_ratings.py
```

## Output

The script creates:
- `output/consolidated_ratings.csv` - Combined dataset with columns:
  - `sample_id`: Unique identifier for each AI response
  - `principle`: Which principle was evaluated
  - `rater_name`: Rater's first name (normalized)
  - `rating`: Score (-1.0, -0.5, 0.5, or 1.0)
  - `user_input`: The prompt given to the AI
  - `ai_output`: The AI's response
  - `source`: 'spreadsheet' or 'langfuse'
  - `comments`: Optional rater comments

## Deduplication Logic

When the same rater rated the same sample in both systems:
- **Keeps:** Langfuse rating (more structured)
- **Drops:** Spreadsheet rating

## Troubleshooting

**"No spreadsheet ratings found"**
- Check that SPREADSHEET_DIR points to correct location
- Ensure .xlsx files contain actual ratings (not all blank)

**"Langfuse connection error"**
- Verify .env file credentials are correct
- Check that LANGFUSE_HOST includes https://

**"Queue name format error"**
- Langfuse queues should be named: `firstname-principle-slug`
- Example: `john-respect-user-attention`

---

## Step 2: Analyze the Data

Once you have `consolidated_ratings.csv`, run the analysis:

```bash
python analyze_ratings.py
```

### What It Calculates

**Tier 1 - Always Calculated:**
- Rating distributions per principle
- Median, mean, IQR per principle
- % positive vs negative ratings
- Coverage statistics

**Tier 2 - Where Sufficient Data (≥20 samples with 2+ raters):**
- Krippendorff's alpha (ordinal) with 95% bootstrap CI
- Percentage agreement
- ICC(2,k) if ≥30 samples

### Output Files

Creates three CSV files in `output/analysis/`:
1. **`coverage_analysis.csv`** - How many samples have multiple raters
2. **`descriptive_statistics.csv`** - Rating distributions per principle
3. **`irr_metrics.csv`** - Inter-rater reliability where calculable

### Interpreting Results

**Krippendorff's Alpha:**
- ≥ 0.80: Strong reliability
- 0.67-0.79: Acceptable reliability  
- < 0.67: Insufficient reliability

**For MIT Presentation:**
- Focus on principles with sufficient overlap
- Report descriptive stats for all principles
- Be transparent about pilot limitations
