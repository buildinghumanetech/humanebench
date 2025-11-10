"""
HumaneBench Data Consolidation Script
Combines ratings from Excel spreadsheets and Langfuse annotation queues
"""

import os
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.api.resources.annotation_queues.types import AnnotationQueueStatus
import re
from datetime import datetime

# ============================================================================
# CONFIGURATION - Edit these paths for your system
# ============================================================================

SPREADSHEET_DIR = "/Users/asamandari1/Downloads/human-eval-spreadsheets"  # Directory containing rater .xlsx files
OUTPUT_DIR = "./output"              # Where to save consolidated CSV
CACHE_DIR = "./cache"                # Where to cache API responses
ENV_FILE = "../../.env"              # Path to .env file with Langfuse credentials
USE_CACHE = True                     # Set to False to force refresh from API

# List of principle first words (to detect principle-only queues without rater names)
PRINCIPLE_KEYWORDS = {
    'design', 'be', 'prioritize', 'foster', 'protect',
    'enhance', 'enable', 'respect'
}

# Rating conversion map: categorical strings to numeric values
RATING_MAP = {
    'HELL YES': 1.0,
    'Soft yes': 0.5,
    'Soft no': -0.5,
    'HELL NO': -1.0
}

# ============================================================================
# Helper Functions
# ============================================================================

def extract_rater_name(filename):
    """Extract first name from filename (first word before whitespace)"""
    basename = os.path.basename(filename)
    # Remove .xlsx extension
    name_part = basename.replace('.xlsx', '').replace('.xls', '')
    # Get first word
    first_word = name_part.split()[0] if ' ' in name_part else name_part
    return first_word.lower()  # Normalize to lowercase


def normalize_principle(principle_name):
    """Normalize principle names to consistent format"""
    # Already in correct format: 'respect-user-attention' etc.
    return principle_name.lower().strip()


def convert_rating(rating_value):
    """Convert categorical rating string to numeric value"""
    # If already numeric, return as-is
    if isinstance(rating_value, (int, float)):
        return float(rating_value)

    # If string, try to map it
    if isinstance(rating_value, str):
        rating_str = rating_value.strip()
        if rating_str in RATING_MAP:
            return RATING_MAP[rating_str]
        else:
            # Log warning for unmapped values
            print(f"    ⚠ Warning: Unmapped rating value '{rating_str}', skipping")
            return None

    return None


# ============================================================================
# Cache Helper Functions
# ============================================================================

def get_cache_path(category, identifier):
    """Generate cache file path for a given category and identifier"""
    cache_subdir = Path(CACHE_DIR) / category
    cache_subdir.mkdir(parents=True, exist_ok=True)
    # Sanitize identifier to be safe for filenames
    safe_id = str(identifier).replace('/', '_').replace('\\', '_')
    return cache_subdir / f"{safe_id}.json"


def load_from_cache(cache_file):
    """Load data from cache file if it exists and USE_CACHE is True"""
    if not USE_CACHE:
        return None

    if not cache_file.exists():
        return None

    try:
        with open(cache_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"    ⚠ Error loading cache {cache_file}: {e}")
        return None


def save_to_cache(cache_file, data):
    """Save data to cache file with timestamp"""
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_data = {
            'cached_at': datetime.now().isoformat(),
            'data': data
        }
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2, default=str)
    except Exception as e:
        print(f"    ⚠ Error saving cache {cache_file}: {e}")


def langfuse_obj_to_dict(obj):
    """Convert Langfuse API object to JSON-serializable dict"""
    if obj is None:
        return None

    # If it's already a dict, return it
    if isinstance(obj, dict):
        return obj

    # Try to convert using __dict__
    if hasattr(obj, '__dict__'):
        result = {}
        for key, value in obj.__dict__.items():
            if key.startswith('_'):
                continue
            # Recursively convert nested objects
            if isinstance(value, list):
                result[key] = [langfuse_obj_to_dict(item) for item in value]
            elif hasattr(value, '__dict__'):
                result[key] = langfuse_obj_to_dict(value)
            else:
                result[key] = value
        return result

    # Fallback: convert to string
    return str(obj)


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_spreadsheet_ratings(spreadsheet_dir):
    """Load all ratings from Excel spreadsheets"""
    print("\n" + "="*80)
    print("LOADING SPREADSHEET RATINGS")
    print("="*80)
    
    all_ratings = []
    spreadsheet_files = list(Path(spreadsheet_dir).glob("*.xlsx"))
    
    print(f"\nFound {len(spreadsheet_files)} spreadsheet files")
    
    for filepath in spreadsheet_files:
        rater_name = extract_rater_name(filepath.name)
        print(f"\nProcessing: {filepath.name}")
        print(f"  Rater: {rater_name}")
        
        try:
            # Get all sheet names (one per principle)
            xl = pd.ExcelFile(filepath)
            
            for sheet_name in xl.sheet_names:
                principle = normalize_principle(sheet_name)
                
                # Read the sheet
                df = pd.read_excel(filepath, sheet_name=sheet_name)

                # Filter out rows without ratings
                df_rated = df[df['rating'].notna()].copy()

                if len(df_rated) > 0:
                    # Convert categorical ratings to numeric values
                    df_rated['rating'] = df_rated['rating'].apply(convert_rating)

                    # Remove rows where rating conversion failed (returned None)
                    df_rated = df_rated[df_rated['rating'].notna()].copy()

                    if len(df_rated) == 0:
                        continue

                    # Add metadata
                    df_rated['rater_name'] = rater_name
                    df_rated['principle'] = principle
                    df_rated['source'] = 'spreadsheet'
                    
                    # Select relevant columns
                    columns = ['sample_id', 'principle', 'rater_name', 'rating', 
                              'user_input', 'ai_output', 'source']
                    
                    # Add comments if available
                    if 'comments' in df_rated.columns:
                        columns.append('comments')
                    
                    df_rated = df_rated[columns]
                    all_ratings.append(df_rated)
                    
                    print(f"    {principle}: {len(df_rated)} ratings")
        
        except Exception as e:
            print(f"  ERROR processing {filepath.name}: {e}")
            continue
    
    if all_ratings:
        combined_df = pd.concat(all_ratings, ignore_index=True)
        print(f"\n✓ Total spreadsheet ratings loaded: {len(combined_df)}")
        return combined_df
    else:
        print("\n⚠ No spreadsheet ratings found")
        return pd.DataFrame()


def load_langfuse_ratings():
    """Load all ratings from Langfuse annotation queues"""
    print("\n" + "="*80)
    print("LOADING LANGFUSE RATINGS")
    print("="*80)

    # Cache statistics
    cache_stats = {
        'queue_items_hits': 0,
        'queue_items_misses': 0,
        'observations_hits': 0,
        'observations_misses': 0,
        'traces_hits': 0,
        'traces_misses': 0
    }

    # Load environment variables
    load_dotenv(ENV_FILE)

    # Initialize Langfuse client
    langfuse = Langfuse(
        public_key=os.getenv('LANGFUSE_PUBLIC_KEY'),
        secret_key=os.getenv('LANGFUSE_SECRET_KEY'),
        host=os.getenv('LANGFUSE_HOST')
    )

    print("\n✓ Connected to Langfuse")

    # Get all annotation queues with pagination
    all_queues = []
    page = 1
    limit = 100

    print("\nFetching annotation queues...")
    while True:
        queues_response = langfuse.api.annotation_queues.list_queues(page=page, limit=limit)

        if not queues_response.data:
            break

        all_queues.extend(queues_response.data)

        # Check if there are more pages
        if hasattr(queues_response, 'meta') and hasattr(queues_response.meta, 'total_pages'):
            if page >= queues_response.meta.total_pages:
                break
        elif len(queues_response.data) < limit:
            # If we got fewer items than the limit, we're on the last page
            break

        page += 1

    print(f"Found {len(all_queues)} annotation queues")

    all_ratings = []

    for queue in all_queues:
        queue_name = queue.name

        # Parse queue name: {firstname}-{principle-first-word}[-optional-rest]
        # Examples: john-respect, john-respect-user-attention
        parts = queue_name.lower().split('-', 1)  # Split on first dash only

        if len(parts) < 2:
            print(f"\n⚠ Skipping queue with unexpected name format: {queue_name}")
            continue

        rater_name = parts[0]
        principle_full = parts[1]

        # Skip principle-only queues (no rater name)
        if rater_name in PRINCIPLE_KEYWORDS:
            print(f"\n⚠ Skipping principle-only queue (no rater name): {queue_name}")
            continue

        # Extract just the first word of the principle for matching
        # "respect-user-attention" -> "respect"
        # "respect" -> "respect"
        principle = principle_full.split('-')[0] if '-' in principle_full else principle_full

        print(f"\nProcessing queue: {queue_name}")
        print(f"  Rater: {rater_name}")
        print(f"  Principle: {principle} (from '{principle_full}')")

        try:
            # Check cache for queue items
            queue_items_cache = get_cache_path('queue_items', queue.id)
            cached_data = load_from_cache(queue_items_cache)

            if cached_data and 'data' in cached_data:
                print(f"  ✓ Loaded {len(cached_data['data'])} items from cache")
                all_queue_items = cached_data['data']
                cache_stats['queue_items_hits'] += 1
            else:
                # Get all items from this queue with pagination
                print(f"  Fetching items from API...")
                cache_stats['queue_items_misses'] += 1
                all_queue_items = []
                item_page = 1
                item_limit = 100

                while True:
                    queue_items_response = langfuse.api.annotation_queues.list_queue_items(
                        queue_id=queue.id,
                        status=AnnotationQueueStatus.COMPLETED,  # Only fetch completed items
                        page=item_page,
                        limit=item_limit
                    )

                    if not queue_items_response.data:
                        break

                    # Convert API objects to dicts for caching
                    items_as_dicts = [langfuse_obj_to_dict(item) for item in queue_items_response.data]
                    all_queue_items.extend(items_as_dicts)

                    # Check if there are more pages
                    if hasattr(queue_items_response, 'meta') and hasattr(queue_items_response.meta, 'total_pages'):
                        if item_page >= queue_items_response.meta.total_pages:
                            break
                    elif len(queue_items_response.data) < item_limit:
                        # If we got fewer items than the limit, we're on the last page
                        break

                    item_page += 1

                # Save to cache
                save_to_cache(queue_items_cache, all_queue_items)
                print(f"  ✓ Fetched and cached {len(all_queue_items)} items")

            # Process all items from this queue (already filtered to COMPLETED by API)
            queue_ratings = 0
            for item in all_queue_items:
                # Queue items are dicts with: object_id, object_type, status
                # Handle both dict (from cache) and object (direct API) formats
                if isinstance(item, dict):
                    object_id = item.get('object_id')
                    object_type = item.get('object_type')
                else:
                    object_id = getattr(item, 'object_id', None)
                    object_type = getattr(item, 'object_type', None)

                if not object_id or not object_type:
                    continue

                # Skip non-observation items (status is already COMPLETED from API filter)
                if object_type != 'OBSERVATION':
                    continue

                try:
                    # Check cache for observation
                    obs_cache = get_cache_path('observations', object_id)
                    cached_obs = load_from_cache(obs_cache)

                    if cached_obs and 'data' in cached_obs:
                        observation = cached_obs['data']
                        cache_stats['observations_hits'] += 1
                    else:
                        # Step 1: Fetch the observation to get metadata and trace_id
                        observation = langfuse.api.observations.get(object_id)
                        observation = langfuse_obj_to_dict(observation)
                        save_to_cache(obs_cache, observation)
                        cache_stats['observations_misses'] += 1

                    # Get metadata from observation (handle both dict and object)
                    if isinstance(observation, dict):
                        metadata = observation.get('metadata', {})
                        sample_id = metadata.get('sample_id') if isinstance(metadata, dict) else None
                        user_input = observation.get('input')
                        ai_output = observation.get('output')
                        trace_id = observation.get('trace_id')
                    else:
                        metadata = observation.metadata if hasattr(observation, 'metadata') else {}
                        sample_id = metadata.get('sample_id') if isinstance(metadata, dict) else None
                        user_input = observation.input if hasattr(observation, 'input') else None
                        ai_output = observation.output if hasattr(observation, 'output') else None
                        trace_id = observation.trace_id if hasattr(observation, 'trace_id') else None

                    if not trace_id:
                        continue

                    # Check cache for trace
                    trace_cache = get_cache_path('traces', trace_id)
                    cached_trace = load_from_cache(trace_cache)

                    if cached_trace and 'data' in cached_trace:
                        trace = cached_trace['data']
                        cache_stats['traces_hits'] += 1
                    else:
                        # Step 2: Fetch the trace to get scores
                        trace = langfuse.api.trace.get(trace_id)
                        trace = langfuse_obj_to_dict(trace)
                        save_to_cache(trace_cache, trace)
                        cache_stats['traces_misses'] += 1

                    # Step 3: Extract scores from trace that match this observation
                    if isinstance(trace, dict):
                        scores = trace.get('scores', [])
                    else:
                        scores = trace.scores if hasattr(trace, 'scores') else []

                    if not scores:
                        continue

                    for score in scores:
                        # Handle both dict and object formats for score
                        if isinstance(score, dict):
                            score_obs_id = score.get('observation_id')
                            rating_value = score.get('value')
                            rating_comment = score.get('comment')
                        else:
                            score_obs_id = score.observation_id if hasattr(score, 'observation_id') else None
                            rating_value = score.value if hasattr(score, 'value') else None
                            rating_comment = score.comment if hasattr(score, 'comment') else None

                        # Check if this score is for our observation
                        if score_obs_id == object_id:

                            if rating_value is not None:
                                # Convert rating to numeric value
                                numeric_rating = convert_rating(rating_value)

                                # Only add if conversion succeeded
                                if numeric_rating is not None:
                                    rating_record = {
                                        'sample_id': sample_id,
                                        'principle': principle,
                                        'rater_name': rater_name,
                                        'rating': numeric_rating,
                                        'user_input': user_input,
                                        'ai_output': ai_output,
                                        'source': 'langfuse'
                                    }

                                    if rating_comment:
                                        rating_record['comments'] = rating_comment

                                    all_ratings.append(rating_record)
                                    queue_ratings += 1

                except Exception as item_error:
                    print(f"    ⚠ Error processing item {object_id}: {item_error}")
                    continue

            print(f"    Loaded: {queue_ratings} ratings")
        
        except Exception as e:
            print(f"  ERROR processing queue {queue_name}: {e}")
            continue
    
    # Print cache statistics
    print("\n" + "-"*80)
    print("CACHE STATISTICS")
    print("-"*80)
    if USE_CACHE:
        total_hits = sum([v for k, v in cache_stats.items() if 'hits' in k])
        total_misses = sum([v for k, v in cache_stats.items() if 'misses' in k])
        total_requests = total_hits + total_misses

        print(f"  Queue Items:   {cache_stats['queue_items_hits']:3d} hits, {cache_stats['queue_items_misses']:3d} misses")
        print(f"  Observations:  {cache_stats['observations_hits']:3d} hits, {cache_stats['observations_misses']:3d} misses")
        print(f"  Traces:        {cache_stats['traces_hits']:3d} hits, {cache_stats['traces_misses']:3d} misses")
        print(f"  Total:         {total_hits:3d} hits, {total_misses:3d} misses")

        if total_requests > 0:
            hit_rate = (total_hits / total_requests) * 100
            print(f"  Hit Rate:      {hit_rate:.1f}%")
            print(f"  API Calls Saved: {total_hits}")
    else:
        print("  Cache disabled (USE_CACHE=False)")

    if all_ratings:
        df = pd.DataFrame(all_ratings)
        print(f"\n✓ Total Langfuse ratings loaded: {len(df)}")
        return df
    else:
        print("\n⚠ No Langfuse ratings found")
        return pd.DataFrame()


def deduplicate_ratings(spreadsheet_df, langfuse_df):
    """Deduplicate ratings, keeping Langfuse over spreadsheet"""
    print("\n" + "="*80)
    print("DEDUPLICATING RATINGS")
    print("="*80)
    
    # Combine both dataframes
    combined_df = pd.concat([spreadsheet_df, langfuse_df], ignore_index=True)
    
    print(f"\nTotal ratings before deduplication: {len(combined_df)}")
    print(f"  From spreadsheets: {len(spreadsheet_df)}")
    print(f"  From Langfuse: {len(langfuse_df)}")
    
    # Create deduplication key
    combined_df['dedup_key'] = (
        combined_df['rater_name'].str.lower() + '|' + 
        combined_df['sample_id'].astype(str) + '|' + 
        combined_df['principle'].str.lower()
    )
    
    # Count duplicates
    duplicate_mask = combined_df.duplicated(subset=['dedup_key'], keep=False)
    n_duplicates = duplicate_mask.sum()
    
    if n_duplicates > 0:
        print(f"\nFound {n_duplicates} duplicate ratings")

        # Sort by source (langfuse first) and drop duplicates
        combined_df['source_priority'] = combined_df['source'].map({
            'langfuse': 0,      # Keep langfuse (lower number = higher priority)
            'spreadsheet': 1    # Drop spreadsheet
        })

        combined_df = combined_df.sort_values('source_priority')
        deduplicated_df = combined_df.drop_duplicates(subset=['dedup_key'], keep='first')

        n_removed = len(combined_df) - len(deduplicated_df)
        print(f"Removed {n_removed} duplicate spreadsheet ratings (kept Langfuse)")

        # Clean up temporary columns (both dedup_key and source_priority)
        deduplicated_df = deduplicated_df.drop(columns=['dedup_key', 'source_priority'])
    else:
        print("\nNo duplicates found")
        deduplicated_df = combined_df

        # Clean up temporary columns (only dedup_key, source_priority wasn't created)
        deduplicated_df = deduplicated_df.drop(columns=['dedup_key'])
    
    print(f"\n✓ Final dataset size: {len(deduplicated_df)} ratings")
    
    return deduplicated_df


def print_coverage_summary(df):
    """Print summary statistics of coverage"""
    print("\n" + "="*80)
    print("COVERAGE SUMMARY")
    print("="*80)
    
    print(f"\nTotal ratings: {len(df)}")
    print(f"Unique raters: {df['rater_name'].nunique()}")
    print(f"Unique samples: {df['sample_id'].nunique()}")
    
    print("\n" + "-"*80)
    print("Ratings by Principle:")
    print("-"*80)
    principle_counts = df.groupby('principle').size().sort_values(ascending=False)
    for principle, count in principle_counts.items():
        print(f"  {principle:40s} {count:4d} ratings")
    
    print("\n" + "-"*80)
    print("Ratings by Rater:")
    print("-"*80)
    rater_counts = df.groupby('rater_name').size().sort_values(ascending=False)
    for rater, count in rater_counts.items():
        print(f"  {rater:20s} {count:4d} ratings")
    
    print("\n" + "-"*80)
    print("Ratings by Source:")
    print("-"*80)
    source_counts = df.groupby('source').size()
    for source, count in source_counts.items():
        print(f"  {source:20s} {count:4d} ratings")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    print("\n" + "="*80)
    print("HUMANEBENCH DATA CONSOLIDATION")
    print("="*80)

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Clean existing consolidated CSV if it exists
    output_path = os.path.join(OUTPUT_DIR, 'consolidated_ratings.csv')
    if os.path.exists(output_path):
        print("\n" + "="*80)
        print("CLEANING EXISTING CONSOLIDATED RATINGS")
        print("="*80)

        try:
            existing_df = pd.read_csv(output_path)
            original_count = len(existing_df)

            # Remove entries where rater_name is actually a principle keyword
            cleaned_df = existing_df[~existing_df['rater_name'].isin(PRINCIPLE_KEYWORDS)]
            cleaned_count = len(cleaned_df)
            removed_count = original_count - cleaned_count

            if removed_count > 0:
                print(f"\nFound {removed_count} invalid entries with principle-only queue names:")
                invalid_raters = existing_df[existing_df['rater_name'].isin(PRINCIPLE_KEYWORDS)]['rater_name'].value_counts()
                for rater, count in invalid_raters.items():
                    print(f"  {rater}: {count} entries")

                # Save cleaned version
                cleaned_df.to_csv(output_path, index=False)
                print(f"\n✓ Cleaned CSV saved: {cleaned_count} valid entries remaining")
            else:
                print("\n✓ No invalid entries found in existing CSV")

        except Exception as e:
            print(f"\n⚠ Error cleaning existing CSV: {e}")

    # Load spreadsheet ratings
    spreadsheet_df = load_spreadsheet_ratings(SPREADSHEET_DIR)
    
    # Load Langfuse ratings
    langfuse_df = load_langfuse_ratings()
    
    # Deduplicate
    if not spreadsheet_df.empty or not langfuse_df.empty:
        final_df = deduplicate_ratings(spreadsheet_df, langfuse_df)
        
        # Print summary
        print_coverage_summary(final_df)
        
        # Save to CSV
        output_path = os.path.join(OUTPUT_DIR, 'consolidated_ratings.csv')
        final_df.to_csv(output_path, index=False)
        
        print("\n" + "="*80)
        print(f"✓ CONSOLIDATED DATA SAVED TO: {output_path}")
        print("="*80)
    else:
        print("\n⚠ No data loaded from either source!")


if __name__ == "__main__":
    main()
