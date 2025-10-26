"""
Data management utilities for the pipeline with semantic deduplication.
"""

import json
import pandas as pd
import shutil
from pathlib import Path
from typing import List, Dict
from config import DATASET_PATH, BACKUP_PATH, HUMANE_PRINCIPLES, TOPIC_DOMAINS, VULNERABLE_POPULATIONS
from semantic_deduplication import SemanticDeduplicator


class DataManager:
    def __init__(self, similarity_threshold: float = 0.60):
        self.dataset_path = Path(DATASET_PATH)
        self.backup_path = Path(BACKUP_PATH)
        self.deduplicator = SemanticDeduplicator(similarity_threshold=similarity_threshold)

        # ID generation state
        self.principle_counters: Dict[str, int] = {}

        # Initialize deduplicator with existing scenarios
        self._initialize_deduplicator()

        # Initialize ID counters with existing IDs
        self._initialize_id_counters()

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

    def _initialize_id_counters(self):
        """Initialize ID counters by finding the highest ID number for each principle."""
        import re

        # Initialize counters for all principles to 0
        for principle in HUMANE_PRINCIPLES:
            self.principle_counters[principle] = 0

        if not self.dataset_path.exists() or self.dataset_path.stat().st_size == 0:
            print("ID counters initialized with zero (no existing dataset)")
            return

        try:
            df = pd.read_json(self.dataset_path, lines=True)

            if 'id' not in df or len(df) == 0:
                print("ID counters initialized with zero (no IDs in dataset)")
                return

            # Parse IDs to find the highest number for each principle
            pattern = re.compile(r"^([a-z\-]+)-(\d{3,})$")
            for id_str in df['id'].dropna().astype(str):
                match = pattern.match(id_str)
                if match:
                    principle, number_str = match.groups()
                    if principle in self.principle_counters:
                        number = int(number_str)
                        self.principle_counters[principle] = max(
                            self.principle_counters[principle],
                            number
                        )

            print(f"Loaded existing ID counters from {len(df)} scenarios")
            for principle, count in sorted(self.principle_counters.items()):
                if count > 0:
                    print(f"   - {principle}: next ID will be {principle}-{count+1:03d}")

        except Exception as e:
            print(f"WARNING: Error loading existing IDs: {e}")
            # Counters remain at 0 on error

    def _generate_ids_for_scenarios(self, scenarios: List[Dict]) -> List[Dict]:
        """
        Generate sequential IDs for scenarios based on their principle.
        This is called AFTER deduplication to avoid skipped numbers.
        Includes collision detection to prevent duplicate IDs.

        Args:
            scenarios: List of scenario dicts (may or may not have 'id' field)

        Returns:
            List of scenarios with properly generated sequential IDs
        """

        # Load existing IDs from dataset to detect collisions
        existing_ids = set()
        if self.dataset_path.exists() and self.dataset_path.stat().st_size > 0:
            try:
                df = pd.read_json(self.dataset_path, lines=True)
                if 'id' in df:
                    existing_ids = set(df['id'].dropna().astype(str).tolist())
            except Exception as e:
                print(f"WARNING: Could not load existing IDs for collision detection: {e}")

        temp_counters = self.principle_counters.copy()
        generated_ids = set()  # Track IDs generated in this batch

        for scenario in scenarios:
            principle = scenario.get('metadata', {}).get('principle', '')
            if not principle or principle not in HUMANE_PRINCIPLES:
                print(f"WARNING: Skipping scenario with invalid principle: {principle}")
                continue

            # Generate ID and check for collisions
            max_attempts = 1000  # Safety limit to prevent infinite loops. Increase if dataset grows beyond this.
            collision_detected = False

            for attempt in range(max_attempts):
                next_number = temp_counters[principle] + 1
                candidate_id = f"{principle}-{next_number:03d}"

                # Check for collision with existing IDs or IDs generated in this batch
                if candidate_id in existing_ids or candidate_id in generated_ids:
                    collision_detected = True
                    print(f"WARNING: ID collision detected: {candidate_id} already exists, incrementing...")
                    temp_counters[principle] = next_number  # Increment counter and try again
                else:
                    # No collision, assign the ID
                    scenario['id'] = candidate_id
                    generated_ids.add(candidate_id)
                    temp_counters[principle] = next_number
                    break
            else:
                # Failed to generate unique ID after max_attempts
                raise RuntimeError(
                    f"Failed to generate unique ID for principle '{principle}' after {max_attempts} attempts. "
                    f"Dataset may be corrupted or counter is out of sync."
                )

        # Update the real counters after processing all scenarios
        self.principle_counters = temp_counters

        return scenarios

    def create_backup(self):
        """Create a backup of the current dataset."""
        if self.dataset_path.exists():
            shutil.copy2(self.dataset_path, self.backup_path)
            print(f"Backup created at {self.backup_path}")

    def append_rows(self, new_rows: List[Dict[str, str]]) -> int:
        """Append new rows to the dataset, filtering semantic duplicates and generating IDs."""
        if not new_rows:
            return 0

        # Step 1: Extract input texts for deduplication
        new_inputs = [row['input'] for row in new_rows]

        # Step 2: Filter out semantic duplicates
        unique_inputs, unique_rows = self.deduplicator.filter_duplicates(new_inputs, new_rows)

        if not unique_rows:
            print("No unique rows to add (all were semantic duplicates)")
            return 0

        # Step 3: Generate sequential IDs AFTER deduplication (avoids skipped numbers)
        print(f"Generating sequential IDs for {len(unique_rows)} unique scenarios...")
        unique_rows = self._generate_ids_for_scenarios(unique_rows)

        # Step 4: Append to JSONL
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
                # Ensure field order: id, input, target, metadata
                ordered_row = {
                    "id": row["id"],
                    "input": row["input"],
                    "target": row["target"],
                    "metadata": row["metadata"]
                }
                f.write(json.dumps(ordered_row, ensure_ascii=False) + "\n")

        print(f"Added {len(unique_rows)} unique rows to dataset")
        if len(unique_rows) < len(new_rows):
            print(f"Filtered out {len(new_rows) - len(unique_rows)} semantic duplicates")

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
            df['vulnerable_populations'] = df['metadata'].apply(lambda x: x.get('vulnerable-populations', []) if isinstance(x, dict) else [])

            # Count by fields
            total_rows = len(df)
            principle_dist = df['principle'].value_counts().to_dict()
            domain_dist = df['domain'].value_counts().to_dict()

            # Track vulnerable populations in two ways:
            # 1. Individual counts - each population increments its counter
            vuln_pop_individual_counts = {}
            for vuln_pops in df['vulnerable_populations']:
                if isinstance(vuln_pops, list):
                    for pop in vuln_pops:
                        vuln_pop_individual_counts[pop] = vuln_pop_individual_counts.get(pop, 0) + 1

            # 2. Combination counts - unique combinations as comma-separated strings
            df['vulnerable_populations_combo'] = df['vulnerable_populations'].apply(
                lambda x: ', '.join(x) if isinstance(x, list) and x else 'none'
            )
            vuln_pop_combo_counts = df['vulnerable_populations_combo'].value_counts().to_dict()

            # Calculate vulnerable population coverage (% of scenarios with at least one VP)
            scenarios_with_vp = sum(1 for vuln_pops in df['vulnerable_populations'] if isinstance(vuln_pops, list) and vuln_pops)
            vuln_pop_coverage = scenarios_with_vp / total_rows if total_rows > 0 else 0

            return {
                "total_rows": total_rows,
                "principle_distribution": principle_dist,
                "domain_distribution": domain_dist,
                "vulnerable_populations_individual": vuln_pop_individual_counts,
                "vulnerable_populations_combinations": vuln_pop_combo_counts,
                "vulnerable_populations_coverage": vuln_pop_coverage,
                "vulnerable_populations_count_with_vp": scenarios_with_vp,
                "deduplication_stats": self.deduplicator.get_statistics()
            }
        except Exception as e:
            print(f"Error getting dataset stats: {e}")
            return {
                "total_rows": 0,
                "principle_distribution": {},
                "domain_distribution": {},
                "vulnerable_populations_individual": {},
                "vulnerable_populations_combinations": {},
                "vulnerable_populations_coverage": 0,
                "vulnerable_populations_count_with_vp": 0,
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
                "coverage_gaps": {"domains": [], "principles": [], "vulnerable_populations": [], "vulnerable_populations_coverage_status": None},
                "common_patterns": {}
            }

        try:
            df = pd.read_json(self.dataset_path, lines=True)

            if len(df) == 0:
                return {
                    "total_scenarios": 0,
                    "guidance": "Empty dataset. Generate diverse scenarios across all categories and principles.",
                    "coverage_gaps": {"domains": [], "principles": [], "vulnerable_populations": [], "vulnerable_populations_coverage_status": None},
                    "common_patterns": {}
                }

            # Analyze input patterns for uniqueness guidance
            # Handle missing input field
            if 'input' not in df:
                return {
                    "total_scenarios": len(df),
                    "guidance": "Dataset missing 'input' field; cannot analyze patterns.",
                    "coverage_gaps": {"domains": [], "principles": [], "vulnerable_populations": [], "vulnerable_populations_coverage_status": None},
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

            # Find underrepresented vulnerable populations (individual balancing)
            from config import VULNERABLE_POPULATION_COVERAGE_TARGET, VULNERABLE_POPULATION_COVERAGE_TOLERANCE

            vuln_pop_gaps = []
            vuln_pop_individual_counts = stats.get('vulnerable_populations_individual', {})

            # Calculate dynamic threshold
            ideal_per_VP = VULNERABLE_POPULATION_COVERAGE_TARGET / len(VULNERABLE_POPULATIONS)
            tolerance_factor = VULNERABLE_POPULATION_COVERAGE_TOLERANCE / VULNERABLE_POPULATION_COVERAGE_TARGET
            vp_threshold = ideal_per_VP * (1 - tolerance_factor)

            # Check all VPs (including those with zero representation)
            for vp in VULNERABLE_POPULATIONS:
                current_count = vuln_pop_individual_counts.get(vp, 0)
                current_ratio = current_count / total_rows
                if current_ratio < vp_threshold:
                    vuln_pop_gaps.append(vp)

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

            if vuln_pop_gaps:
                guidance_parts.append(f"FOCUS on underrepresented vulnerable populations: {', '.join(vuln_pop_gaps)}")

            # Check vulnerable population coverage against target (bidirectional)
            from config import VULNERABLE_POPULATION_COVERAGE_TARGET, VULNERABLE_POPULATION_COVERAGE_TOLERANCE

            vuln_pop_coverage = stats.get('vulnerable_populations_coverage', 0)
            target_min = VULNERABLE_POPULATION_COVERAGE_TARGET - VULNERABLE_POPULATION_COVERAGE_TOLERANCE
            target_max = VULNERABLE_POPULATION_COVERAGE_TARGET + VULNERABLE_POPULATION_COVERAGE_TOLERANCE

            coverage_status = None  # "below", "above", or None (within tolerance)

            if vuln_pop_coverage < target_min:
                coverage_status = "below"
                guidance_parts.append(
                    f"INCREASE vulnerable population scenarios (currently {vuln_pop_coverage:.1%}, target {VULNERABLE_POPULATION_COVERAGE_TARGET:.0%}±{VULNERABLE_POPULATION_COVERAGE_TOLERANCE:.0%})"
                )
            elif vuln_pop_coverage > target_max:
                coverage_status = "above"
                guidance_parts.append(
                    f"DECREASE vulnerable population scenarios (currently {vuln_pop_coverage:.1%}, target {VULNERABLE_POPULATION_COVERAGE_TARGET:.0%}±{VULNERABLE_POPULATION_COVERAGE_TOLERANCE:.0%})"
                )

            return {
                "total_scenarios": len(df),
                "guidance": " ".join(guidance_parts),
                "coverage_gaps": {
                    "domains": domain_gaps,
                    "principles": principle_gaps,
                    "vulnerable_populations": vuln_pop_gaps,
                    "vulnerable_populations_coverage_status": coverage_status
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
                "coverage_gaps": {"domains": [], "principles": [], "vulnerable_populations": [], "vulnerable_populations_coverage_status": None},
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


