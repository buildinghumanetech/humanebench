# Humane Bench QA Framework

Comprehensive quality assurance and data science toolkit for validating the Humane Bench dataset.

## Overview

This QA framework provides both **automated validation** and **manual review** workflows to ensure dataset quality, following peer-reviewed best practices for benchmark dataset validation.

**Key Features:**
- ✅ Automated schema validation and distribution analysis
- 📊 Interactive Jupyter notebook for exploratory data analysis
- 🎯 Stratified sampling for efficient manual review
- 📈 Inter-rater reliability analysis (Fleiss' Kappa)
- 📄 Comprehensive validation reports

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Automated Validation

```bash
python qa_validation.py
```

This generates a `qa_validation_report.json` file with comprehensive validation results.

### 3. Launch Jupyter Notebook for Interactive Analysis

```bash
jupyter notebook qa_analysis.ipynb
```

## Components

### 1. Automated Validation Script (`qa_validation.py`)

**What it does:**
- Validates schema compliance using Pydantic models
- Analyzes distribution balance across principles, domains, and vulnerable populations
- Detects exact duplicates (IDs and inputs)
- Identifies statistical outliers (input length, etc.)
- Checks coverage gaps and underrepresented categories

**Usage:**
```bash
python qa_validation.py
```

**Output:**
- Console report with color-coded findings
- `qa_validation_report.json` with detailed results
- Exit code 1 if errors found, 0 if clean

**Validation Checks:**

| Check | Description | Threshold |
|-------|-------------|-----------|
| Schema Validation | All required fields present and valid | Must pass 100% |
| Principle Balance | Even distribution across 8 principles | Target: 12.5% each (±5% acceptable) |
| Duplicate IDs | No duplicate scenario IDs | 0 duplicates |
| Duplicate Inputs | No exact duplicate scenario texts | 0 duplicates |
| Input Length | Within schema constraints | 10-500 characters |
| Domain Coverage | All standard domains represented | 12 standard domains |

### 2. Interactive Analysis Notebook (`qa_analysis.ipynb`)

**What it includes:**

#### Section 1: Exploratory Data Analysis (EDA)
- Load dataset and compute basic statistics
- Display sample scenarios
- Calculate summary metrics

#### Section 2: Distribution Analysis & Visualization
- Principle distribution (bar charts showing balance)
- Domain distribution (frequency analysis)
- Vulnerable population coverage
- Heatmap: Principle × Domain coverage matrix
- Input length distribution analysis

#### Section 3: Stratified Random Sampling
- Generates stratified sample (default: 120 scenarios, ~11.5%)
- Ensures proportional representation across all principles
- Exports `review_sample.csv` for manual review
- Sample size based on 95% confidence level, ~10% margin of error

#### Section 4: Inter-Rater Reliability Analysis
- Calculates **Fleiss' Kappa** for multiple reviewers
- Interprets agreement using Landis & Koch criteria
- Identifies specific disagreements for discussion
- Visualizes Kappa scores by criterion

**Landis & Koch Interpretation:**
- **< 0.00:** Poor agreement
- **0.00-0.20:** Slight agreement
- **0.21-0.40:** Fair agreement
- **0.41-0.60:** Moderate agreement ✓
- **0.61-0.80:** Substantial agreement ✓✓
- **0.81-1.00:** Almost perfect agreement ✓✓✓

## Manual Review Workflow

### Step 1: Generate Sample

Run the notebook through Section 3 to generate `review_sample.csv`.

### Step 2: Distribute to Reviewers

Share the CSV file with 2-3 reviewers. Each reviewer independently fills in their columns:

| Column | Values | Description |
|--------|--------|-------------|
| `reviewer_N_clear` | yes/no | Is the input clear and understandable? |
| `reviewer_N_realistic` | yes/no | Is the scenario realistic? |
| `reviewer_N_tests_principle` | yes/no | Does it effectively test the target principle? |
| `reviewer_N_quality` | low/medium/high | Overall quality assessment |
| `reviewer_N_notes` | text | Optional notes for discussion |

**Important:** Reviewers should work **independently** without discussing ratings until analysis is complete.

### Step 3: Analyze Inter-Rater Reliability

After all reviews are completed:
1. Open the notebook
2. Run Section 4 with the completed `review_sample.csv`
3. Review Fleiss' Kappa scores for each criterion
4. Discuss disagreements as a team

**Target Thresholds:**
- **Minimum acceptable:** Kappa ≥ 0.41 (Moderate agreement)
- **Good:** Kappa ≥ 0.61 (Substantial agreement)
- **Excellent:** Kappa ≥ 0.81 (Almost perfect agreement)

If Kappa < 0.41: Review guidelines, clarify criteria, and consider additional training or revised rubrics.

### Step 4: Resolve Disagreements

For scenarios where reviewers disagree:
1. Review the specific scenarios flagged in the analysis
2. Discuss reasoning as a team
3. Establish consensus or clarify review criteria
4. Document decisions for consistency

## Interpreting Results

### Distribution Balance

**Goal:** Even distribution across 8 principles (~12.5% each)

**Acceptable Range:** 7.5% - 17.5% per principle (±5% deviation)

**Actions if Imbalanced:**
- Generate additional scenarios for underrepresented principles
- Remove or reallocate scenarios from overrepresented principles

### Coverage Gaps

**Standard Domains (12 total):**
- relationships
- mental-health
- financial-decisions
- career-guidance
- parenting
- health-wellness
- technology-use
- politics-society
- education
- moral-ambiguity
- crisis-situations
- everyday-decisions

**Actions if Gaps Found:**
- Generate scenarios for missing domains
- Verify non-standard domains are intentional
- Consolidate similar non-standard domains into standard ones

### Duplicate Detection

**Types of Duplicates:**
1. **Duplicate IDs:** Critical error - must fix immediately
2. **Exact Duplicate Inputs:** Indicates generation/deduplication failure
3. **Near Duplicates:** May indicate insufficient diversity

**Actions:**
- Investigate root cause (generation process, deduplication threshold)
- Remove or regenerate duplicate scenarios
- Adjust semantic deduplication threshold if needed

## Best Practices

### Sample Size Guidelines

Based on statistical literature for benchmark validation:

| Dataset Size | Sample Size | Coverage | Confidence Level | Margin of Error |
|--------------|-------------|----------|------------------|-----------------|
| 1,000 | 100 | 10% | 95% | ±10% |
| 1,000 | 150 | 15% | 95% | ±8% |
| 1,000 | 278 | 28% | 95% | ±5% |

**Recommended:** 100-150 scenarios (10-15%) for initial QA

### Inter-Rater Reliability Best Practices

1. **Minimum 2 reviewers** (3+ preferred for Fleiss' Kappa)
2. **Independent review** - no discussion until after all ratings
3. **Clear rubrics** - document what constitutes "clear", "realistic", etc.
4. **Blind review** - reviewers shouldn't see each other's ratings
5. **Calibration** - do a small pilot (10-20 scenarios) first to align understanding
6. **Documentation** - record disagreements and resolutions

### Quality Thresholds

**Minimum Acceptable Quality:**
- Schema validation: 100% pass
- Duplicate IDs: 0
- Exact duplicate inputs: 0
- Principle balance: All within ±5% of target
- Inter-rater reliability: Kappa ≥ 0.41 for all criteria

**High Quality:**
- Principle balance: All within ±2% of target
- All standard domains represented with ≥1% coverage
- Inter-rater reliability: Kappa ≥ 0.61 for all criteria
- <1% non-standard domains

## Troubleshooting

### Common Issues

**Issue:** Validation script reports duplicate IDs
- **Cause:** ID generation collision or data corruption
- **Fix:** Review data_manager.py ID counter logic, manually deduplicate

**Issue:** Low inter-rater reliability (Kappa < 0.41)
- **Cause:** Unclear criteria, insufficient training, or genuinely ambiguous scenarios
- **Fix:** Clarify review rubric, conduct calibration session, revise criteria

**Issue:** Severe distribution imbalance (>10% deviation)
- **Cause:** Generation process bias or insufficient balance checking
- **Fix:** Generate targeted scenarios for underrepresented principles

**Issue:** Notebook cells fail with import errors
- **Cause:** Missing dependencies
- **Fix:** Run `pip install -r requirements.txt`

## References

This QA framework is based on peer-reviewed best practices:

1. **Inter-Rater Reliability:** Fleiss, J. L. (1971). "Measuring nominal scale agreement among many raters." Psychological Bulletin, 76(5), 378-382.

2. **Interpretation Guidelines:** Landis, J. R., & Koch, G. G. (1977). "The measurement of observer agreement for categorical data." Biometrics, 33(1), 159-174.

3. **Benchmark Dataset Validation:** Best practices from CVPR, NeurIPS, and ACL conferences for benchmark dataset quality assurance.

4. **Sample Size Methodology:** Cochran, W. G. (1977). "Sampling Techniques" (3rd ed.). John Wiley & Sons.

## Files Generated

| File | Description | Git Track? |
|------|-------------|------------|
| `qa_validation_report.json` | Automated validation results | No (add to .gitignore) |
| `review_sample.csv` | Stratified sample for manual review | Yes (for team collaboration) |
| `qa_validation.py` | Validation script | Yes |
| `qa_analysis.ipynb` | Analysis notebook | Yes |
| `QA_README.md` | This documentation | Yes |

## Next Steps After QA

1. **Address Critical Errors:**
   - Fix duplicate IDs
   - Remove exact duplicate inputs
   - Resolve schema validation failures

2. **Improve Balance:**
   - Generate scenarios for underrepresented principles
   - Review and potentially remove overrepresented scenarios

3. **Enhance Diversity:**
   - Increase coverage of underrepresented domains
   - Ensure vulnerable population representation

4. **Iterate on Quality:**
   - Review scenarios flagged during manual review
   - Regenerate low-quality scenarios
   - Update generation prompts based on findings

5. **Document Findings:**
   - Create a summary report of QA findings
   - Document decisions and rationale
   - Update generation guidelines for future scenarios

---

**Questions or Issues?** Refer to this documentation or review the inline comments in the code files.
