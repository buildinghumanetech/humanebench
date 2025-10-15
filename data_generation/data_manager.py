"""
Data management utilities for the pipeline with semantic deduplication.
"""

import json
import pandas as pd
import shutil
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from config import DATASET_PATH, BACKUP_PATH, HUMANE_PRINCIPLES, TOPIC_DOMAINS
from semantic_deduplication import SemanticDeduplicator


class DataManager:
    def __init__(self, similarity_threshold: float = 0.60):
        self.dataset_path = Path(DATASET_PATH)
        self.backup_path = Path(BACKUP_PATH)
        self.deduplicator = SemanticDeduplicator(similarity_threshold=similarity_threshold)

        # ID validation state
        self.principle_counters: Dict[str, int] = {}
        self.existing_ids: set = set()

        # Initialize deduplicator with existing scenarios
        self._initialize_deduplicator()

        # Initialize ID validation with existing IDs
        self._initialize_id_validation()

    def _initialize_deduplicator(self):
        """Initialize the deduplicator with existing dataset scenarios."""
        if self.dataset_path.exists():
            try:
                if self.dataset_path.stat().st_size == 0:
                    print("No existing scenarios found in dataset")
                    return

                df = pd.read_json(self.dataset_path, lines=True)
                if 'input' in df and not df.empty:
                    existing_inputs = df['input'].dropna().astype(str).tolist()
                    if existing_inputs:
                        print(f"Initializing deduplicator with {len(existing_inputs)} existing scenarios...")
                        self.deduplicator.update_existing_texts(existing_inputs)
                    else:
                        print("No existing scenarios found in dataset")
                else:
                    print("Dataset has no 'input' column or is empty")
            except Exception as e:
                print(f"Warning: Could not load existing scenarios for deduplication: {e}")

    def _initialize_id_validation(self):
        """Initialize ID validation by loading existing IDs and building counters."""
        # Initialize counters for all principles to 0
        for principle in HUMANE_PRINCIPLES:
            self.principle_counters[principle] = 0

        if not self.dataset_path.exists() or self.dataset_path.stat().st_size == 0:
            print("ID validation initialized with zero counters (no existing dataset)")
            return

        try:
            df = pd.read_json(self.dataset_path, lines=True)

            if 'id' not in df or len(df) == 0:
                print("ID validation initialized with zero counters (no IDs in dataset)")
                return

            # Collect all existing IDs
            self.existing_ids = set(df['id'].dropna().astype(str).tolist())

            # Parse IDs to find the highest number for each principle
            for principle in HUMANE_PRINCIPLES:
                self.principle_counters[principle] = self._find_highest_id_number(principle)

            print(f"Loaded {len(self.existing_ids)} existing IDs from dataset")
            print(f"Current ID counters per principle:")
            for principle, count in sorted(self.principle_counters.items()):
                if count > 0:
                    print(f"   - {principle}: {count} entries (next: {principle}-{count+1:03d})")

        except Exception as e:
            print(f"WARNING: Error loading existing IDs: {e}")
            # Counters remain at 0 on error

    def _find_highest_id_number(self, principle: str) -> int:
        """Find the highest ID number for a given principle."""
        pattern = re.compile(rf"^{re.escape(principle)}-(\d{{3,}})$")
        max_number = 0

        for id_str in self.existing_ids:
            match = pattern.match(id_str)
            if match:
                number = int(match.group(1))
                max_number = max(max_number, number)

        return max_number

    def _validate_id_format(self, id_str: str) -> Tuple[bool, Optional[str]]:
        """
        Validate that an ID follows the correct format: principle-###

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not id_str or not isinstance(id_str, str):
            return False, "ID is empty or not a string"

        # Check format: principle-###
        pattern = re.compile(r"^([a-z\-]+)-(\d{3,})$")
        match = pattern.match(id_str)

        if not match:
            return False, f"ID '{id_str}' doesn't match format 'principle-###'"

        principle, number_str = match.groups()

        # Validate principle exists
        if principle not in HUMANE_PRINCIPLES:
            return False, f"Invalid principle '{principle}' in ID '{id_str}'"

        return True, None

    def _validate_scenario_id(self, scenario: Dict, temp_counters: Dict[str, int], temp_existing_ids: set) -> Tuple[bool, Optional[str]]:
        """
        Validate a scenario's ID against expected sequential numbering.

        Args:
            scenario: Scenario dict to validate
            temp_counters: Temporary counters for batch validation
            temp_existing_ids: Temporary set of existing IDs for batch validation

        Returns:
            Tuple of (is_valid, error_message)
        """
        scenario_id = scenario.get('id', '')
        metadata = scenario.get('metadata', {})
        principle = metadata.get('principle', '') if isinstance(metadata, dict) else ''

        # Validate format first
        is_valid_format, format_error = self._validate_id_format(scenario_id)
        if not is_valid_format:
            return False, format_error

        # Parse the ID
        pattern = re.compile(r"^([a-z\-]+)-(\d{3,})$")
        match = pattern.match(scenario_id)
        id_principle, number_str = match.groups()
        actual_number = int(number_str)

        # Check that principle from ID matches metadata principle
        if id_principle != principle:
            return False, f"ID principle '{id_principle}' doesn't match metadata principle '{principle}'"

        # Check sequential numbering
        expected_number = temp_counters.get(principle, 0) + 1
        if actual_number != expected_number:
            expected_id = f"{principle}-{expected_number:03d}"
            return False, f"Expected ID '{expected_id}' but got '{scenario_id}'"

        # Check collision with existing IDs
        if scenario_id in temp_existing_ids:
            return False, f"ID '{scenario_id}' already exists (collision detected)"

        return True, None

    def _correct_scenario_ids(self, scenarios: List[Dict]) -> Tuple[List[Dict], int]:
        """
        Correct all IDs in a batch to match expected sequential numbering.

        Returns:
            Tuple of (corrected_scenarios, count_corrected)
        """
        corrected_scenarios = []
        temp_counters = self.principle_counters.copy()
        corrections_made = 0

        for scenario in scenarios:
            principle = scenario.get('metadata', {}).get('principle', '')
            if not principle or principle not in HUMANE_PRINCIPLES:
                print(f"WARNING: Skipping scenario with invalid principle: {principle}")
                continue

            # Generate correct ID based on temp counter
            next_number = temp_counters[principle] + 1
            correct_id = f"{principle}-{next_number:03d}"

            # Update scenario if ID is wrong
            old_id = scenario.get('id', 'NONE')
            if old_id != correct_id:
                print(f"Correcting ID: '{old_id}' -> '{correct_id}'")
                scenario['id'] = correct_id
                corrections_made += 1

            corrected_scenarios.append(scenario)

            # Update temp counter
            temp_counters[principle] = next_number

        return corrected_scenarios, corrections_made

    def _register_new_ids(self, scenarios: List[Dict]):
        """
        Register new IDs after successful append to dataset.
        Updates internal counters and existing IDs set.
        """
        for scenario in scenarios:
            scenario_id = scenario['id']
            principle = scenario.get('metadata', {}).get('principle', '')

            # Update existing IDs set
            self.existing_ids.add(scenario_id)

            # Parse number and update counter
            pattern = re.compile(r"^([a-z\-]+)-(\d{3,})$")
            match = pattern.match(scenario_id)
            if match:
                _, number_str = match.groups()
                number = int(number_str)

                # Update counter to highest seen
                if principle in self.principle_counters:
                    self.principle_counters[principle] = max(
                        self.principle_counters[principle],
                        number
                    )

    def create_backup(self):
        """Create a backup of the current dataset."""
        if self.dataset_path.exists():
            shutil.copy2(self.dataset_path, self.backup_path)
            print(f"Backup created at {self.backup_path}")

    def append_rows(self, new_rows: List[Dict[str, str]]) -> int:
        """Append new rows to the dataset, filtering semantic duplicates and validating/correcting IDs."""
        if not new_rows:
            return 0

        # Step 1: Validate and correct IDs
        print(f"\nValidating IDs for {len(new_rows)} scenarios...")
        corrected_rows, corrections_made = self._correct_scenario_ids(new_rows)

        if corrections_made > 0:
            print(f"Corrected {corrections_made} IDs to match expected sequential numbering")
        else:
            print(f"All IDs are valid and sequential")

        # Step 2: Extract input texts for deduplication
        new_inputs = [row['input'] for row in corrected_rows]

        # Step 3: Filter out semantic duplicates
        unique_inputs, unique_rows = self.deduplicator.filter_duplicates(new_inputs, corrected_rows)

        if not unique_rows:
            print("No unique rows to add (all were semantic duplicates)")
            return 0

        # Step 4: Register the new IDs before appending
        self._register_new_ids(unique_rows)

        # Step 5: Append to JSONL
        # Ensure file ends with newline before appending
        if self.dataset_path.exists() and self.dataset_path.stat().st_size > 0:
            with open(self.dataset_path, 'rb') as f:
                f.seek(-1, 2)  # Go to last byte
                last_char = f.read(1)
                needs_newline = last_char != b'\n'
        else:
            needs_newline = False

        with open(self.dataset_path, 'a', encoding='utf-8', newline='') as f:
            if needs_newline:
                f.write("\n")
            for row in unique_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"Added {len(unique_rows)} unique rows to dataset")
        if len(unique_rows) < len(corrected_rows):
            print(f"Filtered out {len(corrected_rows) - len(unique_rows)} semantic duplicates")

        return len(unique_rows)

    def get_dataset_stats(self) -> Dict:
        """Get statistics about the current dataset."""
        if not self.dataset_path.exists() or self.dataset_path.stat().st_size == 0: # In case of empty existing file
            return {
                "total_rows": 0,
                "principle_distribution": {},
                "domain_distribution": {},
                "vulnerable_population_distribution": {},
                "deduplication_stats": self.deduplicator.get_statistics()
            }

        try:
            df = pd.read_json(self.dataset_path, lines=True)

            # Validate required columns exist
            required_columns = ['id', 'input', 'target', 'metadata']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                raise ValueError(
                    f"Dataset is missing required columns: {', '.join(missing_columns)}. "
                    f"Required columns are: {', '.join(required_columns)}"
                )

            # Extract metadata fields
            df['principle'] = df['metadata'].apply(lambda x: x.get('principle', '') if isinstance(x, dict) else '')
            df['domain'] = df['metadata'].apply(lambda x: x.get('domain', '') if isinstance(x, dict) else '')
            df['vulnerable_population'] = df['metadata'].apply(lambda x: x.get('vulnerable-population', '') if isinstance(x, dict) else '')

            # Count by fields
            total_rows = len(df)
            principle_dist = df['principle'].value_counts().to_dict()
            domain_dist = df['domain'].value_counts().to_dict()
            vuln_pop_dist = df['vulnerable_population'].value_counts().to_dict()

            return {
                "total_rows": total_rows,
                "principle_distribution": principle_dist,
                "domain_distribution": domain_dist,
                "vulnerable_population_distribution": vuln_pop_dist,
                "deduplication_stats": self.deduplicator.get_statistics()
            }
        except Exception as e:
            print(f"Error getting dataset stats: {e}")
            return {
                "total_rows": 0,
                "principle_distribution": {},
                "domain_distribution": {},
                "vulnerable_population_distribution": {},
                "deduplication_stats": self.deduplicator.get_statistics()
            }

    def get_sample_rows(self, n: int = 3) -> List[Dict]:
        """Get a sample of recent rows for display."""
        if not self.dataset_path.exists() or self.dataset_path.stat().st_size == 0:
            return []

        try:
            df = pd.read_json(self.dataset_path, lines=True)
            if len(df) == 0:
                return []

            # Get last n rows
            sample_df = df.tail(n)
            return sample_df.to_dict('records')
        except Exception as e:
            print(f"Error getting sample rows: {e}")
            return []

    def adjust_similarity_threshold(self, new_threshold: float):
        """Adjust the semantic similarity threshold."""
        self.deduplicator.adjust_threshold(new_threshold)
        print(f"Semantic similarity threshold adjusted to {new_threshold}")

    def clear_deduplication_cache(self):
        """Clear the semantic deduplication cache."""
        self.deduplicator.clear_cache()
        print("Semantic deduplication cache cleared")

    def get_principle_balance(self) -> Dict[str, float]:
        """Get the balance of core humane principles in the dataset."""
        stats = self.get_dataset_stats()
        total = stats['total_rows']

        if total == 0:
            return {}

        # Get principle counts directly from distribution
        principle_counts = stats['principle_distribution']

        # Convert to ratios
        return {principle: count/total for principle, count in principle_counts.items()}

    def suggest_needed_principles(self, target_balance: float = 0.125) -> List[str]:
        """Suggest which core principles need more scenarios (target is ~1/8 each for 8 principles)."""
        balance = self.get_principle_balance()

        if not balance:
            return list(HUMANE_PRINCIPLES)  # All principles needed if no data

        underrepresented = []
        for principle in HUMANE_PRINCIPLES:
            current_ratio = balance.get(principle, 0)
            if current_ratio < target_balance:
                underrepresented.append(principle)

        return underrepresented

    def get_diversity_analysis(self) -> Dict:
        """Analyze diversity patterns in existing dataset to guide generation."""
        if not self.dataset_path.exists() or self.dataset_path.stat().st_size == 0:
            return {
                "total_scenarios": 0,
                "guidance": "No existing dataset found. Generate diverse scenarios across all categories and principles.",
                "coverage_gaps": {"categories": [], "principles": []},
                "common_patterns": {}
            }

        try:
            df = pd.read_json(self.dataset_path, lines=True)

            if len(df) == 0:
                return {
                    "total_scenarios": 0,
                    "guidance": "Empty dataset. Generate diverse scenarios across all categories and principles.",
                    "coverage_gaps": {"categories": [], "principles": []},
                    "common_patterns": {}
                }

            # Analyze input patterns for uniqueness guidance
            # Handle missing input field
            if 'input' not in df:
                return {
                    "total_scenarios": len(df),
                    "guidance": "Dataset missing 'input' field; cannot analyze patterns.",
                    "coverage_gaps": {"categories": [], "principles": []},
                    "common_patterns": {}
                }

            inputs = df['input'].dropna().astype(str).tolist()

            # Extract key patterns
            common_starters = {}
            common_topics = {}

            for inp in inputs:
                # Common question starters
                starter = inp.split()[:3] if inp.split() else []
                starter_key = " ".join(starter).lower()
                common_starters[starter_key] = common_starters.get(starter_key, 0) + 1

                # Common topic keywords
                words = inp.lower().split()
                for word in words:
                    if len(word) > 4:  # Focus on meaningful words
                        common_topics[word] = common_topics.get(word, 0) + 1

            # Identify overrepresented patterns (top 20%)
            sorted_starters = sorted(common_starters.items(), key=lambda x: x[1], reverse=True)
            sorted_topics = sorted(common_topics.items(), key=lambda x: x[1], reverse=True)

            overused_starters = [starter for starter, count in sorted_starters[:max(1, len(sorted_starters)//5)] if count > 2]
            overused_topics = [topic for topic, count in sorted_topics[:max(1, len(sorted_topics)//10)] if count > 3]

            # Analyze distribution gaps
            stats = self.get_dataset_stats()

            # Find underrepresented domains
            total_rows = stats['total_rows'] or 1 # Defensive approach to avoid division by 0
            domain_gaps = []
            for domain, count in stats['domain_distribution'].items():
                if count / total_rows < 0.08:  # Less than ~8% representation
                    domain_gaps.append(domain)

            # Find underrepresented core principles
            principle_gaps = []
            core_principle_balance = self.get_principle_balance()
            for principle in HUMANE_PRINCIPLES:
                current_ratio = core_principle_balance.get(principle, 0)
                if current_ratio < 0.10:  # Less than ~10% representation
                    principle_gaps.append(principle)

            # Generate guidance
            guidance_parts = []
            guidance_parts.append(f"Existing dataset has {len(df)} scenarios.")

            if overused_starters:
                guidance_parts.append(f"AVOID these overused question starters: {', '.join(overused_starters[:5])}")

            if overused_topics:
                guidance_parts.append(f"AVOID these overused topics: {', '.join(overused_topics[:8])}")

            if domain_gaps:
                guidance_parts.append(f"FOCUS on underrepresented domains: {', '.join(domain_gaps)}")

            if principle_gaps:
                guidance_parts.append(f"FOCUS on underrepresented principles: {', '.join(principle_gaps[:3])}")

            return {
                "total_scenarios": len(df),
                "guidance": " ".join(guidance_parts),
                "coverage_gaps": {
                    "domains": domain_gaps,
                    "principles": principle_gaps
                },
                "common_patterns": {
                    "overused_starters": overused_starters[:5],
                    "overused_topics": overused_topics[:8]
                },
                "distribution": {
                    "domains": stats['domain_distribution'],
                    "principles": stats['principle_distribution']
                }
            }

        except Exception as e:
            print(f"Error analyzing diversity: {e}")
            return {
                "total_scenarios": 0,
                "guidance": "Error analyzing existing dataset. Generate diverse scenarios.",
                "coverage_gaps": {"domains": [], "principles": []},
                "common_patterns": {}
            }

    def get_deduplication_feedback(self) -> Dict:
        """Get feedback about recent deduplication patterns."""
        dedupe_stats = self.deduplicator.get_statistics()
        session_stats = dedupe_stats.get('session_stats', {})

        # Get detailed deduplication feedback
        detailed_feedback = self.deduplicator.get_deduplication_feedback()

        feedback = {
            "total_cached_scenarios": dedupe_stats.get('total_cached_texts', 0),
            "similarity_threshold": dedupe_stats.get('similarity_threshold', 0.60),
            "session_processed": session_stats.get('total_processed', 0),
            "session_duplicates": session_stats.get('total_duplicates', 0),
            "duplicate_rate": 0,
            "guidance": ""
        }

        # Calculate duplicate rate
        if feedback["session_processed"] > 0:
            feedback["duplicate_rate"] = (feedback["session_duplicates"] / feedback["session_processed"]) * 100

        # Generate comprehensive guidance
        guidance_parts = []

        # Base guidance
        if feedback["total_cached_scenarios"] > 100:
            guidance_parts.append(f"Large dataset ({feedback['total_cached_scenarios']} scenarios) requires highly unique generation.")
        elif feedback["total_cached_scenarios"] > 0:
            guidance_parts.append(f"Growing dataset ({feedback['total_cached_scenarios']} scenarios) needs diverse scenarios.")

        # Add detailed deduplication feedback
        if detailed_feedback:
            guidance_parts.append(detailed_feedback)

        # General uniqueness guidance
        if feedback["duplicate_rate"] > 60:
            guidance_parts.append("URGENT: Drastically vary vocabulary, contexts, and question structures.")
        elif feedback["duplicate_rate"] > 30:
            guidance_parts.append("Focus on novel scenarios with different angles and phrasings.")

        feedback["guidance"] = " ".join(guidance_parts) if guidance_parts else "Generate diverse scenarios."

        return feedback


