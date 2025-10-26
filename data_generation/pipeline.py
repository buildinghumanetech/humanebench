"""
Main pipeline controller for data generation.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the current directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

from generators import ScenarioGenerator
from validators import ScenarioValidator
from data_manager import DataManager
from config import (DEFAULT_BATCH_SIZE, SIMILARITY_THRESHOLD, TARGET_ROWS,
                    ENABLE_DATASET_CONTEXT, ENABLE_DEDUPLICATION_FEEDBACK, CONTEXT_ANALYSIS_FREQUENCY)


class DataGenerationPipeline:
    def __init__(self, api_key: Optional[str] = None, similarity_threshold: Optional[float] = None):
        """Initialize the data generation pipeline."""
        # Use config value if not explicitly provided
        if similarity_threshold is None:
            similarity_threshold = SIMILARITY_THRESHOLD

        self.generator = ScenarioGenerator(api_key)
        self.validator = ScenarioValidator(api_key)
        self.data_manager = DataManager(similarity_threshold)

        # Initialize context tracking
        self.batch_counter = 0

        print("🚀 Data Generation Pipeline Initialized")
        print(f"📊 Current dataset: {self.data_manager.get_dataset_stats()['total_rows']} rows")
        if ENABLE_DATASET_CONTEXT or ENABLE_DEDUPLICATION_FEEDBACK:
            print("🎯 Context-aware generation: ENABLED")
        else:
            print("🎯 Context-aware generation: DISABLED")

    def run_interactive(self):
        """Run the pipeline in interactive mode."""
        print("\n" + "="*60)
        print("🎯 HUMANE TECH BENCHMARK DATA GENERATION PIPELINE")
        print("="*60)

        # Check if automated mode is enabled
        automated_mode = TARGET_ROWS is not None
        if automated_mode:
            print(f"🤖 AUTOMATED MODE: Target {TARGET_ROWS} additional rows")
            starting_rows = self.data_manager.get_dataset_stats()['total_rows']
            session_rows_added = 0
            target_reached = False
        else:
            print("👤 MANUAL MODE: Press Enter to continue, or type context/commands")

        # Show current dataset statistics
        self._show_dataset_stats()

        batch_size = DEFAULT_BATCH_SIZE
        user_context = ""

        while True:
            # Show progress in automated mode
            if automated_mode:
                progress_percent = (session_rows_added / TARGET_ROWS) * 100 if TARGET_ROWS > 0 else 0
                print(f"\n🤖 AUTOMATED MODE PROGRESS: {session_rows_added}/{TARGET_ROWS} rows ({progress_percent:.1f}%)")

                # Check if target reached
                if session_rows_added >= TARGET_ROWS:
                    print(f"\n🎉 TARGET REACHED! Added {session_rows_added} rows this session")
                    break

                print(f"📝 Generating batch {batch_size} scenarios")
            else:
                print(f"\n📝 Ready to generate {batch_size} scenarios")
                print("💡 Current context:", user_context if user_context else "None")

                # Check for stop command
                user_input = input("\nPress Enter to continue, or provide context/commands: ").strip()

                if "STOP GENERATION" in user_input.upper():
                    print("\n🛑 Generation stopped by user")
                    break

                # Update context if provided
                if user_input and "STOP GENERATION" not in user_input.upper():
                    user_context = user_input
                    print(f"📝 Context updated: {user_context}")

            # Conditionally gather context based on settings and frequency
            dataset_context = None
            deduplication_feedback = None

            if (ENABLE_DATASET_CONTEXT or ENABLE_DEDUPLICATION_FEEDBACK) and (self.batch_counter % CONTEXT_ANALYSIS_FREQUENCY == 0):
                print(f"\n📊 Analyzing existing dataset for context-aware generation...")

                if ENABLE_DATASET_CONTEXT:
                    dataset_context = self.data_manager.get_diversity_analysis()

                if ENABLE_DEDUPLICATION_FEEDBACK:
                    deduplication_feedback = self.data_manager.get_deduplication_feedback()

            # Generate scenarios
            if dataset_context or deduplication_feedback:
                print(f"\n🔄 Generating {batch_size} context-aware scenarios...")
            else:
                print(f"\n🔄 Generating {batch_size} scenarios...")

            # Check if we should use web search for inspiration
            use_web_search = False
            if deduplication_feedback and deduplication_feedback.get('duplicate_rate', 0) > 50:
                print(f"🔍 High duplicate rate ({deduplication_feedback['duplicate_rate']:.1f}%) - using web search for fresh inspiration")
                use_web_search = True

            scenarios = self.generator.generate_batch(
                batch_size=batch_size,
                context=user_context,
                dataset_context=dataset_context,
                deduplication_feedback=deduplication_feedback,
                search_for_inspiration=use_web_search
            )

            # Increment batch counter
            self.batch_counter += 1

            if not scenarios:
                print("❌ No scenarios generated. Check your API key and model availability.")
                continue

            # Validate scenarios
            print(f"\n🔍 Validating {len(scenarios)} scenarios...")
            approved_scenarios, validation_reports, feedback = self.validator.validate_batch(scenarios)

            # Show validation summary
            validation_summary = self.validator.get_validation_summary(validation_reports)
            self._show_validation_summary(validation_summary)

            if not approved_scenarios:
                if feedback:
                    print("\n📋 VALIDATION FEEDBACK FOR IMPROVEMENT:")
                    print(feedback)
                    print("\n🔄 Regenerating batch with feedback and web search for inspiration...")

                    # Regenerate with feedback (use fresh context analysis)
                    enhanced_context = f"{user_context}\n\nPREVIOUS BATCH FEEDBACK:\n{feedback}" if user_context else f"PREVIOUS BATCH FEEDBACK:\n{feedback}"
                    scenarios = self.generator.generate_batch(
                        batch_size=batch_size,
                        context=enhanced_context,
                        dataset_context=dataset_context,
                        deduplication_feedback=deduplication_feedback,
                        search_for_inspiration=True  # Use web search when struggling
                    )

                    if scenarios:
                        print(f"\n🔍 Re-validating {len(scenarios)} improved scenarios...")
                        approved_scenarios, validation_reports, _ = self.validator.validate_batch(scenarios)

                        if approved_scenarios:
                            print(f"✅ Feedback integration successful: {len(approved_scenarios)} scenarios approved")
                        else:
                            print("❌ Re-generation also failed validation. Continuing to next batch...")
                            continue
                    else:
                        print("❌ Re-generation failed. Continuing to next batch...")
                        continue
                else:
                    print("❌ No scenarios passed validation. Continuing to next batch...")
                    continue

            # Add to dataset (with semantic deduplication)
            print(f"\n💾 Adding {len(approved_scenarios)} approved scenarios to dataset...")
            added_count = self.data_manager.append_rows(approved_scenarios)

            # Update session tracking in automated mode
            if automated_mode:
                session_rows_added += added_count

            # Show results
            print(f"\n✅ Added {added_count} unique scenarios to dataset")

            # Show sample of what was added
            if added_count > 0:
                self._show_sample_scenarios(approved_scenarios)

            # Show updated dataset stats
            self._show_dataset_stats()

            print("\n" + "-"*60)

        print("\n🎉 Pipeline completed!")
        self._show_final_summary()

    def run_batch(self,
                  total_scenarios: int,
                  batch_size: int = None,
                  context: str = "",
                  auto_approve: bool = False) -> Dict:
        """
        Run the pipeline in batch mode for a specific number of scenarios.

        Args:
            total_scenarios: Total number of scenarios to generate
            batch_size: Size of each batch (default from config)
            context: Generation context
            auto_approve: If True, skip validation (use with caution)

        Returns:
            Summary statistics
        """
        if batch_size is None:
            batch_size = DEFAULT_BATCH_SIZE

        print(f"🚀 Running batch mode: {total_scenarios} scenarios in batches of {batch_size}")

        total_generated = 0
        total_added = 0
        all_validation_reports = []

        batches = (total_scenarios + batch_size - 1) // batch_size  # Ceiling division

        for batch_num in range(batches):
            current_batch_size = min(batch_size, total_scenarios - total_generated)

            print(f"\n📦 Batch {batch_num + 1}/{batches}: Generating {current_batch_size} scenarios")

            # Generate
            scenarios = self.generator.generate_batch(
                batch_size=current_batch_size,
                context=context
            )

            if not scenarios:
                print(f"❌ Batch {batch_num + 1} failed to generate scenarios")
                continue

            total_generated += len(scenarios)

            # Validate (unless auto-approve is enabled)
            if auto_approve:
                approved_scenarios = scenarios
                validation_reports = []
            else:
                approved_scenarios, validation_reports, feedback = self.validator.validate_batch(scenarios)
                all_validation_reports.extend(validation_reports)

                # Handle feedback in batch mode (simpler than interactive)
                if feedback and not approved_scenarios:
                    print(f"⚠️ Batch {batch_num + 1} failed validation with feedback - skipping")
                    continue

            # Add to dataset
            if approved_scenarios:
                added_count = self.data_manager.append_rows(approved_scenarios)
                total_added += added_count
                print(f"✅ Batch {batch_num + 1}: Added {added_count} scenarios")

        # Final summary
        final_stats = {
            "total_generated": total_generated,
            "total_added": total_added,
            "validation_summary": self.validator.get_validation_summary(all_validation_reports) if all_validation_reports else {},
            "final_dataset_stats": self.data_manager.get_dataset_stats()
        }

        print(f"\n🎉 Batch mode completed!")
        print(f"📊 Generated: {total_generated}, Added: {total_added}")

        return final_stats

    def run_automated(self,
                     target_additional_rows: int,
                     batch_size: int = None,
                     context: str = "",
                     max_attempts: int = None) -> Dict:
        """
        Run the pipeline in automated mode until N additional rows are added to the dataset.

        Args:
            target_additional_rows: Number of additional rows to add to dataset
            batch_size: Size of each batch (default from config)
            context: Generation context
            max_attempts: Maximum batches to attempt (prevents infinite loops)

        Returns:
            Summary statistics
        """
        if batch_size is None:
            batch_size = DEFAULT_BATCH_SIZE

        if max_attempts is None:
            # Default: Allow up to 3x target rows in generation attempts
            max_attempts = (target_additional_rows * 3 + batch_size - 1) // batch_size

        print(f"🤖 AUTOMATED MODE: Adding {target_additional_rows} rows to dataset")
        print(f"   Batch size: {batch_size}")
        print(f"   Max attempts: {max_attempts} batches")
        print(f"   Context: {context if context else 'None'}")

        starting_row_count = self.data_manager.get_dataset_stats()['total_rows']
        target_total_rows = starting_row_count + target_additional_rows

        total_generated = 0
        total_added = 0
        batch_count = 0
        all_validation_reports = []

        print(f"\n📊 Starting with {starting_row_count} rows, targeting {target_total_rows} total rows")

        while total_added < target_additional_rows and batch_count < max_attempts:
            batch_count += 1
            current_total = self.data_manager.get_dataset_stats()['total_rows']
            remaining_needed = target_total_rows - current_total

            print(f"\n🔄 BATCH {batch_count}/{max_attempts}")
            print(f"   Progress: {total_added}/{target_additional_rows} rows added")
            print(f"   Need {remaining_needed} more rows")

            # Generate scenarios
            scenarios = self.generator.generate_batch(
                batch_size=batch_size,
                context=context
            )

            if not scenarios:
                print(f"❌ Batch {batch_count} failed to generate scenarios")
                continue

            total_generated += len(scenarios)

            # Validate scenarios
            approved_scenarios, validation_reports = self.validator.validate_batch(scenarios)
            all_validation_reports.extend(validation_reports)

            # Add to dataset
            if approved_scenarios:
                batch_added = self.data_manager.append_rows(approved_scenarios)
                total_added += batch_added
                print(f"✅ Batch {batch_count}: Added {batch_added} scenarios")

                # Show sample if any were added
                if batch_added > 0:
                    self._show_sample_scenarios(approved_scenarios)

                # Check if we've reached our target
                if total_added >= target_additional_rows:
                    print(f"\n🎉 TARGET REACHED! Added {total_added} rows in {batch_count} batches")
                    break
            else:
                print(f"❌ Batch {batch_count}: No scenarios added")

        # Final summary
        final_row_count = self.data_manager.get_dataset_stats()['total_rows']
        actual_added = final_row_count - starting_row_count

        final_stats = {
            "target_additional_rows": target_additional_rows,
            "actual_rows_added": actual_added,
            "total_generated": total_generated,
            "batches_processed": batch_count,
            "success_rate": actual_added / target_additional_rows if target_additional_rows > 0 else 0,
            "validation_summary": self.validator.get_validation_summary(all_validation_reports) if all_validation_reports else {},
            "final_dataset_stats": self.data_manager.get_dataset_stats()
        }

        print(f"\n🏁 AUTOMATED MODE COMPLETED!")
        print(f"   Target: {target_additional_rows} rows")
        print(f"   Actual: {actual_added} rows added")
        print(f"   Success rate: {final_stats['success_rate']:.1%}")
        print(f"   Batches processed: {batch_count}")

        if actual_added < target_additional_rows:
            shortfall = target_additional_rows - actual_added
            print(f"⚠️  Shortfall: {shortfall} rows (reached max attempts or generation issues)")

        return final_stats

    def _show_dataset_stats(self):
        """Display current dataset statistics."""
        stats = self.data_manager.get_dataset_stats()
        print(f"\n📊 CURRENT DATASET STATS:")
        print(f"   Total rows: {stats['total_rows']}")

        if stats['principle_distribution']:
            print(f"   Principle distribution:")
            for principle, count in stats['principle_distribution'].items():
                percentage = (count / stats['total_rows']) * 100 if stats['total_rows'] > 0 else 0
                print(f"     • {principle}: {count} ({percentage:.1f}%)")

    def _show_validation_summary(self, summary: Dict):
        """Display validation summary."""
        if not summary:
            return

        print(f"\n🔍 VALIDATION SUMMARY:")
        print(f"   Approved: {summary['approved_count']}/{summary['total_scenarios']} ({summary['approval_rate']:.1%})")
        print(f"   Average score: {summary['average_score']:.1f}/100")

        if summary['rejection_reasons']:
            print(f"   Top rejection reasons:")
            for reason, count in list(summary['rejection_reasons'].items())[:3]:
                print(f"     • {reason}: {count}")

    def _show_sample_scenarios(self, scenarios: List[Dict]):
        """Display random sample of scenarios (10% of batch)."""
        import random

        if not scenarios:
            return

        # Calculate 10% sample size (minimum 1, maximum 10)
        sample_size = max(1, min(10, len(scenarios) // 10))

        # Get random sample
        sample = random.sample(scenarios, sample_size)

        print(f"\n📝 SAMPLE SCENARIOS ADDED ({sample_size} of {len(scenarios)}):")
        for i, scenario in enumerate(sample, 1):
            metadata = scenario.get('metadata', {})
            vuln_pops = metadata.get('vulnerable-populations', [])
            vuln_pops_display = ', '.join(vuln_pops) if vuln_pops else 'None'
            print(f"\n   {i}. ID: {scenario.get('id', 'N/A')}")
            print(f"      INPUT: {scenario['input']}")
            print(f"      TARGET: {scenario['target']}")
            print(f"      PRINCIPLE: {metadata.get('principle', 'N/A')}")
            print(f"      DOMAIN: {metadata.get('domain', 'N/A')}")
            print(f"      VULNERABLE POPULATIONS: {vuln_pops_display}")

    def _show_final_summary(self):
        """Display final pipeline summary."""
        stats = self.data_manager.get_dataset_stats()
        dedup_stats = stats.get('deduplication_stats', {})

        print(f"\n📈 FINAL SUMMARY:")
        print(f"   Final dataset size: {stats['total_rows']} rows")
        print(f"   Semantic similarity threshold: {dedup_stats.get('similarity_threshold', 'N/A')}")
        print(f"   Cached embeddings: {dedup_stats.get('total_cached_texts', 'N/A')}")


def main():
    """Main entry point for the pipeline."""
    # Check for environment variables
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ Error: OPENROUTER_API_KEY environment variable not set")
        print("   Please set your OpenRouter API key:")
        print("   export OPENROUTER_API_KEY='your-api-key-here'")
        return

    try:
        pipeline = DataGenerationPipeline()
        pipeline.run_interactive()

    except KeyboardInterrupt:
        print("\n\n🛑 Pipeline interrupted by user")
    except Exception as e:
        print(f"\n❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()