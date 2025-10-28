"""
Automated validation suite for Humane Bench dataset.

Performs comprehensive checks on dataset quality including:
- Schema compliance (all required fields present and valid)
- Distribution balance (principles, domains, vulnerable populations)
- Duplicate detection and outlier identification
- Statistical quality metrics
"""

import json
import pandas as pd
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
from pydantic import BaseModel, Field, field_validator, ValidationError

# Import constants from config
from config import HUMANE_PRINCIPLES, TOPIC_DOMAINS, VULNERABLE_POPULATIONS, DATASET_PATH


# Pydantic model for dataset schema validation
class ScenarioMetadata(BaseModel):
    """Metadata for a scenario."""
    principle: str = Field(..., description="Must match one of the 8 humane principles")
    domain: str = Field(..., description="Topic domain")
    vulnerable_population: str = Field(default="", description="Empty string or vulnerable population type")

    @field_validator('principle')
    @classmethod
    def validate_principle(cls, v: str) -> str:
        """Ensure principle is one of the fixed 8 principles."""
        if v not in HUMANE_PRINCIPLES:
            raise ValueError(f"Principle must be one of {HUMANE_PRINCIPLES}, got: {v}")
        return v

    @field_validator('vulnerable_population')
    @classmethod
    def validate_vulnerable_population(cls, v: str) -> str:
        """Ensure vulnerable population is empty or valid."""
        if v and v not in VULNERABLE_POPULATIONS:
            raise ValueError(f"Vulnerable population must be empty or one of {VULNERABLE_POPULATIONS}, got: {v}")
        return v


class HumaneBenchScenario(BaseModel):
    """Schema for a single scenario in the Humane Bench dataset."""
    id: str = Field(..., description="Unique identifier in format: principle-NNN")
    input: str = Field(..., min_length=10, max_length=500, description="User scenario/question (10-500 chars)")
    target: str = Field(..., description="Target principle (must match metadata.principle)")
    metadata: ScenarioMetadata

    @field_validator('id')
    @classmethod
    def validate_id_format(cls, v: str) -> str:
        """Ensure ID follows the pattern: principle-slug-NNN."""
        pattern = r'^[a-z\-]+-\d{3,}$'
        if not re.match(pattern, v):
            raise ValueError(f"ID must match pattern 'principle-slug-NNN', got: {v}")
        return v

    @field_validator('target')
    @classmethod
    def validate_target(cls, v: str) -> str:
        """Ensure target is one of the fixed principles."""
        if v not in HUMANE_PRINCIPLES:
            raise ValueError(f"Target must be one of {HUMANE_PRINCIPLES}, got: {v}")
        return v

    def model_post_init(self, __context) -> None:
        """Additional validation after model initialization."""
        # Ensure target matches metadata.principle
        if self.target != self.metadata.principle:
            raise ValueError(f"Target '{self.target}' does not match metadata.principle '{self.metadata.principle}'")


class ValidationReport:
    """Comprehensive validation report for the dataset."""

    def __init__(self, dataset_path: str = DATASET_PATH):
        self.dataset_path = Path(dataset_path)
        self.scenarios: List[Dict] = []
        self.errors: List[Dict] = []
        self.warnings: List[Dict] = []
        self.stats: Dict = {}

    def load_dataset(self) -> bool:
        """Load the dataset from JSONL file."""
        if not self.dataset_path.exists():
            self.errors.append({"type": "file_not_found", "message": f"Dataset file not found: {self.dataset_path}"})
            return False

        try:
            with open(self.dataset_path, 'r') as f:
                self.scenarios = [json.loads(line) for line in f]
            print(f"✓ Loaded {len(self.scenarios)} scenarios from {self.dataset_path}")
            return True
        except Exception as e:
            self.errors.append({"type": "load_error", "message": f"Failed to load dataset: {e}"})
            return False

    def validate_schema(self) -> Tuple[int, int]:
        """
        Validate each scenario against the Pydantic schema.

        Returns:
            Tuple of (valid_count, invalid_count)
        """
        print("\n=== Schema Validation ===")
        valid_count = 0
        invalid_count = 0

        for idx, scenario in enumerate(self.scenarios):
            try:
                HumaneBenchScenario(**scenario)
                valid_count += 1
            except ValidationError as e:
                invalid_count += 1
                error_details = []
                for error in e.errors():
                    field = '.'.join(str(x) for x in error['loc'])
                    error_details.append(f"{field}: {error['msg']}")

                self.errors.append({
                    "type": "schema_error",
                    "row": idx + 1,
                    "id": scenario.get('id', 'unknown'),
                    "errors": error_details
                })

        print(f"✓ Valid scenarios: {valid_count}/{len(self.scenarios)}")
        if invalid_count > 0:
            print(f"✗ Invalid scenarios: {invalid_count}/{len(self.scenarios)}")

        return valid_count, invalid_count

    def analyze_distributions(self) -> Dict:
        """Analyze distribution of principles, domains, and vulnerable populations."""
        print("\n=== Distribution Analysis ===")

        df = pd.DataFrame(self.scenarios)

        # Extract metadata fields
        df['principle'] = df['metadata'].apply(lambda x: x.get('principle', ''))
        df['domain'] = df['metadata'].apply(lambda x: x.get('domain', ''))
        df['vulnerable_pop'] = df['metadata'].apply(lambda x: x.get('vulnerable-population', ''))

        # Principle distribution
        principle_dist = df['principle'].value_counts().to_dict()
        principle_pct = df['principle'].value_counts(normalize=True).mul(100).round(2).to_dict()

        # Domain distribution
        domain_dist = df['domain'].value_counts().to_dict()

        # Vulnerable population distribution
        vuln_pop_dist = df['vulnerable_pop'].value_counts().to_dict()

        # Check balance (target: 12.5% per principle)
        target_pct = 100.0 / len(HUMANE_PRINCIPLES)
        imbalanced_principles = []

        print(f"\nPrinciple Distribution (target: {target_pct:.1f}% each):")
        for principle in HUMANE_PRINCIPLES:
            count = principle_dist.get(principle, 0)
            pct = principle_pct.get(principle, 0.0)
            status = "✓" if abs(pct - target_pct) < 5.0 else "⚠"
            print(f"  {status} {principle}: {count} ({pct:.1f}%)")

            if abs(pct - target_pct) >= 5.0:
                imbalanced_principles.append({
                    "principle": principle,
                    "count": count,
                    "percentage": pct,
                    "target": target_pct,
                    "deviation": abs(pct - target_pct)
                })

        if imbalanced_principles:
            self.warnings.append({
                "type": "distribution_imbalance",
                "message": f"{len(imbalanced_principles)} principles deviate by >5% from target",
                "details": imbalanced_principles
            })

        print(f"\nTop 10 Domains:")
        for domain, count in sorted(domain_dist.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  • {domain}: {count}")

        print(f"\nVulnerable Populations:")
        vuln_with_pop = sum(1 for v in df['vulnerable_pop'] if v)
        vuln_without = len(df) - vuln_with_pop
        print(f"  • With vulnerable population: {vuln_with_pop} ({vuln_with_pop/len(df)*100:.1f}%)")
        print(f"  • Without: {vuln_without} ({vuln_without/len(df)*100:.1f}%)")
        for pop, count in sorted(vuln_pop_dist.items(), key=lambda x: x[1], reverse=True):
            if pop:  # Skip empty string
                print(f"    - {pop}: {count}")

        return {
            "principle_distribution": principle_dist,
            "principle_percentages": principle_pct,
            "domain_distribution": domain_dist,
            "vulnerable_population_distribution": vuln_pop_dist,
            "imbalanced_principles": imbalanced_principles
        }

    def detect_duplicates(self) -> List[Tuple[int, int, float]]:
        """
        Detect potential exact and near-duplicate inputs.

        Returns:
            List of tuples: (index1, index2, similarity_score)
        """
        print("\n=== Duplicate Detection ===")

        df = pd.DataFrame(self.scenarios)

        # Exact duplicates
        exact_dupes = df[df.duplicated(subset=['input'], keep=False)]

        if len(exact_dupes) > 0:
            print(f"✗ Found {len(exact_dupes)} exact duplicate inputs")
            for input_text in exact_dupes['input'].unique():
                ids = df[df['input'] == input_text]['id'].tolist()
                self.errors.append({
                    "type": "exact_duplicate",
                    "message": f"Exact duplicate input found in {len(ids)} scenarios",
                    "ids": ids,
                    "input": input_text[:100] + "..." if len(input_text) > 100 else input_text
                })
        else:
            print("✓ No exact duplicate inputs found")

        # Check for duplicate IDs
        id_dupes = df[df.duplicated(subset=['id'], keep=False)]
        if len(id_dupes) > 0:
            print(f"✗ Found {len(id_dupes)} duplicate IDs")
            for dup_id in id_dupes['id'].unique():
                self.errors.append({
                    "type": "duplicate_id",
                    "message": f"Duplicate ID found: {dup_id}",
                    "count": len(df[df['id'] == dup_id])
                })
        else:
            print("✓ No duplicate IDs found")

        return []

    def detect_outliers(self) -> Dict:
        """Detect statistical outliers in input length and other metrics."""
        print("\n=== Outlier Detection ===")

        df = pd.DataFrame(self.scenarios)
        df['input_length'] = df['input'].str.len()

        # Input length statistics
        length_stats = df['input_length'].describe()

        print(f"Input Length Statistics:")
        print(f"  • Mean: {length_stats['mean']:.1f} characters")
        print(f"  • Median: {length_stats['50%']:.1f} characters")
        print(f"  • Min: {length_stats['min']:.0f}, Max: {length_stats['max']:.0f}")
        print(f"  • Std Dev: {length_stats['std']:.1f}")

        # Flag outliers (< 10 or > 500 characters - per schema)
        too_short = df[df['input_length'] < 10]
        too_long = df[df['input_length'] > 500]

        if len(too_short) > 0:
            print(f"⚠ {len(too_short)} inputs are too short (< 10 chars)")
            for idx, row in too_short.iterrows():
                self.errors.append({
                    "type": "input_too_short",
                    "id": row['id'],
                    "length": row['input_length'],
                    "input": row['input']
                })

        if len(too_long) > 0:
            print(f"⚠ {len(too_long)} inputs are too long (> 500 chars)")
            for idx, row in too_long.iterrows():
                self.errors.append({
                    "type": "input_too_long",
                    "id": row['id'],
                    "length": row['input_length'],
                    "input": row['input'][:100] + "..."
                })

        # Check for very short inputs (even if >= 10, may be low quality)
        suspicious_short = df[(df['input_length'] >= 10) & (df['input_length'] < 30)]
        if len(suspicious_short) > 0:
            print(f"⚠ {len(suspicious_short)} inputs are suspiciously short (10-29 chars)")
            self.warnings.append({
                "type": "suspicious_short_inputs",
                "count": len(suspicious_short),
                "message": "These inputs may lack detail or context",
                "examples": suspicious_short[['id', 'input']].head(5).to_dict('records')
            })

        return {
            "length_stats": length_stats.to_dict(),
            "too_short": len(too_short),
            "too_long": len(too_long),
            "suspicious_short": len(suspicious_short)
        }

    def check_coverage_gaps(self) -> Dict:
        """Identify missing or underrepresented categories."""
        print("\n=== Coverage Gap Analysis ===")

        df = pd.DataFrame(self.scenarios)
        df['principle'] = df['metadata'].apply(lambda x: x.get('principle', ''))
        df['domain'] = df['metadata'].apply(lambda x: x.get('domain', ''))

        gaps = []

        # Check for missing principles
        present_principles = set(df['principle'].unique())
        missing_principles = set(HUMANE_PRINCIPLES) - present_principles

        if missing_principles:
            print(f"✗ Missing principles: {', '.join(missing_principles)}")
            gaps.append({
                "type": "missing_principles",
                "items": list(missing_principles)
            })
        else:
            print("✓ All 8 principles are represented")

        # Check for underrepresented principles (< 5% of dataset)
        principle_counts = df['principle'].value_counts()
        underrep_threshold = len(df) * 0.05
        underrep_principles = principle_counts[principle_counts < underrep_threshold]

        if len(underrep_principles) > 0:
            print(f"⚠ Underrepresented principles (< 5%):")
            for principle, count in underrep_principles.items():
                pct = count / len(df) * 100
                print(f"    - {principle}: {count} ({pct:.1f}%)")
            gaps.append({
                "type": "underrepresented_principles",
                "items": underrep_principles.to_dict()
            })

        # Check domain coverage
        present_domains = set(df['domain'].unique())
        standard_domains_present = present_domains.intersection(set(TOPIC_DOMAINS))

        print(f"\n✓ Coverage: {len(standard_domains_present)}/{len(TOPIC_DOMAINS)} standard domains represented")

        # Non-standard domains
        non_standard_domains = present_domains - set(TOPIC_DOMAINS)
        if non_standard_domains:
            print(f"⚠ Non-standard domains found: {len(non_standard_domains)}")
            for domain in sorted(non_standard_domains):
                count = len(df[df['domain'] == domain])
                print(f"    - {domain}: {count}")
            self.warnings.append({
                "type": "non_standard_domains",
                "count": len(non_standard_domains),
                "domains": list(non_standard_domains)
            })

        return {
            "missing_principles": list(missing_principles),
            "underrepresented_principles": underrep_principles.to_dict() if len(underrep_principles) > 0 else {},
            "non_standard_domains": list(non_standard_domains)
        }

    def generate_report(self, output_path: Optional[str] = None) -> Dict:
        """Generate comprehensive validation report."""
        print("\n" + "="*60)
        print("HUMANE BENCH DATASET - QUALITY ASSURANCE REPORT")
        print("="*60)

        # Load dataset
        if not self.load_dataset():
            print("✗ Failed to load dataset. Aborting validation.")
            return {"success": False, "errors": self.errors}

        # Run all checks
        valid_count, invalid_count = self.validate_schema()
        distributions = self.analyze_distributions()
        self.detect_duplicates()
        outliers = self.detect_outliers()
        gaps = self.check_coverage_gaps()

        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Total scenarios: {len(self.scenarios)}")
        print(f"Valid scenarios: {valid_count}")
        print(f"Invalid scenarios: {invalid_count}")
        print(f"Errors found: {len(self.errors)}")
        print(f"Warnings: {len(self.warnings)}")

        if len(self.errors) == 0 and len(self.warnings) == 0:
            print("\n🎉 Dataset passed all validation checks!")
        elif len(self.errors) == 0:
            print(f"\n✓ Dataset is valid but has {len(self.warnings)} warnings to review")
        else:
            print(f"\n⚠ Dataset has {len(self.errors)} errors that need attention")

        # Compile report
        report = {
            "success": len(self.errors) == 0,
            "total_scenarios": len(self.scenarios),
            "valid_scenarios": valid_count,
            "invalid_scenarios": invalid_count,
            "distributions": distributions,
            "outliers": outliers,
            "coverage_gaps": gaps,
            "errors": self.errors,
            "warnings": self.warnings
        }

        # Save report if output path provided
        if output_path:
            output_file = Path(output_path)
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n📄 Report saved to: {output_file}")

        return report


def main():
    """Run validation suite."""
    validator = ValidationReport()
    report = validator.generate_report(output_path="qa_validation_report.json")

    # Exit with error code if validation failed
    if not report["success"]:
        exit(1)


if __name__ == "__main__":
    main()
