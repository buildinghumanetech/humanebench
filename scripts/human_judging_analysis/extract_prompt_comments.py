"""
Extract Prompt Comments from Langfuse
Aggregates rater comments about input prompts (sample_id level)
"""

import os
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.api.resources.annotation_queues.types import AnnotationQueueStatus
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = "./output"
CACHE_DIR = "./cache"
ENV_FILE = "../../.env"
USE_CACHE = True

# List of principle first words (to detect principle-only queues without rater names)
PRINCIPLE_KEYWORDS = {
    'design', 'be', 'prioritize', 'foster', 'protect',
    'enhance', 'enable', 'respect'
}

# Rating conversion map
RATING_MAP = {
    'HELL YES': 1.0,
    'Soft yes': 0.5,
    'Soft no': -0.5,
    'HELL NO': -1.0
}

# ============================================================================
# Helper Functions
# ============================================================================

def convert_rating(rating_value):
    """Convert categorical rating string to numeric value"""
    if isinstance(rating_value, (int, float)):
        return float(rating_value)
    if isinstance(rating_value, str):
        rating_str = rating_value.strip()
        if rating_str in RATING_MAP:
            return RATING_MAP[rating_str]
    return None


def get_cache_path(category, identifier):
    """Generate cache file path for a given category and identifier"""
    cache_subdir = Path(CACHE_DIR) / category
    cache_subdir.mkdir(parents=True, exist_ok=True)
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
    if isinstance(obj, dict):
        return obj

    # Handle enum types specifically (they have __members__ on their class)
    if hasattr(obj, '__class__') and hasattr(obj.__class__, '__members__'):
        if hasattr(obj, 'value'):
            return obj.value
        elif hasattr(obj, 'name'):
            return obj.name

    # Try to convert using __dict__ (for regular objects)
    if hasattr(obj, '__dict__'):
        result = {}
        for key, value in obj.__dict__.items():
            if key.startswith('_'):
                continue
            if isinstance(value, list):
                result[key] = [langfuse_obj_to_dict(item) for item in value]
            elif hasattr(value, '__dict__') or (hasattr(value, '__class__') and hasattr(value.__class__, '__members__')):
                result[key] = langfuse_obj_to_dict(value)
            else:
                result[key] = value
        return result

    return str(obj)


# ============================================================================
# Main Extraction Logic
# ============================================================================

def extract_prompt_comments():
    """Extract all comments about prompts from Langfuse"""
    print("\n" + "="*80)
    print("EXTRACTING PROMPT COMMENTS FROM LANGFUSE")
    print("="*80)

    # Load environment variables
    load_dotenv(ENV_FILE)

    # Initialize Langfuse client
    langfuse = Langfuse(
        public_key=os.getenv('LANGFUSE_PUBLIC_KEY'),
        secret_key=os.getenv('LANGFUSE_SECRET_KEY'),
        host=os.getenv('LANGFUSE_HOST')
    )

    print("\n✓ Connected to Langfuse")

    # Cache statistics
    cache_stats = {
        'queue_items_hits': 0,
        'queue_items_misses': 0,
        'observations_hits': 0,
        'observations_misses': 0,
        'traces_hits': 0,
        'traces_misses': 0
    }

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
        if len(queues_response.data) < limit:
            break
        page += 1

    print(f"Found {len(all_queues)} annotation queues")

    all_comments = []

    for queue in all_queues:
        queue_name = queue.name

        # Parse queue name: {firstname}-{principle} OR just {principle}
        parts = queue_name.lower().split('-', 1)

        if len(parts) < 2:
            print(f"\n⚠ Skipping queue with unexpected name format: {queue_name}")
            continue

        first_part = parts[0]

        # Check if this is a principle-only queue (no rater name prefix)
        if first_part in PRINCIPLE_KEYWORDS:
            # Principle-only queue (e.g., "respect-user-attention")
            rater_name = "principle-only"
            principle_full = queue_name.lower()
            principle = first_part
        else:
            # Regular queue with rater name (e.g., "mark-l-respect")
            rater_name = first_part
            principle_full = parts[1]
            principle = principle_full.split('-')[0] if '-' in principle_full else principle_full

        print(f"\nProcessing queue: {queue_name}")
        print(f"  Rater: {rater_name}")
        print(f"  Principle: {principle}")

        try:
            # Check cache for queue items
            queue_items_cache = get_cache_path('queue_items', queue.id)
            cached_data = load_from_cache(queue_items_cache)

            if cached_data and 'data' in cached_data:
                all_queue_items = cached_data['data']
                cache_stats['queue_items_hits'] += 1
            else:
                # Get all items from this queue with pagination
                cache_stats['queue_items_misses'] += 1
                all_queue_items = []
                item_page = 1
                item_limit = 100

                while True:
                    queue_items_response = langfuse.api.annotation_queues.list_queue_items(
                        queue_id=queue.id,
                        status=AnnotationQueueStatus.COMPLETED,
                        page=item_page,
                        limit=item_limit
                    )

                    if not queue_items_response.data:
                        break

                    items_as_dicts = [langfuse_obj_to_dict(item) for item in queue_items_response.data]
                    all_queue_items.extend(items_as_dicts)

                    if len(queue_items_response.data) < item_limit:
                        break

                    item_page += 1

                save_to_cache(queue_items_cache, all_queue_items)

            # Process all items from this queue
            queue_comments = 0
            for item in all_queue_items:
                # Handle both dict and object formats
                if isinstance(item, dict):
                    object_id = item.get('object_id')
                    object_type = item.get('object_type')
                else:
                    object_id = getattr(item, 'object_id', None)
                    object_type = getattr(item, 'object_type', None)

                if not object_id or object_type != 'OBSERVATION':
                    continue

                try:
                    # Check cache for observation
                    obs_cache = get_cache_path('observations', object_id)
                    cached_obs = load_from_cache(obs_cache)

                    if cached_obs and 'data' in cached_obs:
                        observation = cached_obs['data']
                        cache_stats['observations_hits'] += 1
                    else:
                        observation = langfuse.api.observations.get(object_id)
                        observation = langfuse_obj_to_dict(observation)
                        save_to_cache(obs_cache, observation)
                        cache_stats['observations_misses'] += 1

                    # Get metadata from observation
                    if isinstance(observation, dict):
                        metadata = observation.get('metadata', {})
                        sample_id = metadata.get('sample_id') if isinstance(metadata, dict) else None
                        user_input = observation.get('input')
                        trace_id = observation.get('trace_id')
                    else:
                        metadata = observation.metadata if hasattr(observation, 'metadata') else {}
                        sample_id = metadata.get('sample_id') if isinstance(metadata, dict) else None
                        user_input = observation.input if hasattr(observation, 'input') else None
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
                        trace = langfuse.api.trace.get(trace_id)
                        trace = langfuse_obj_to_dict(trace)
                        save_to_cache(trace_cache, trace)
                        cache_stats['traces_misses'] += 1

                    # Extract scores from trace
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

                        # Check if this score is for our observation AND has a comment
                        if score_obs_id == object_id and rating_comment:
                            # Convert rating to numeric value
                            numeric_rating = convert_rating(rating_value)

                            comment_record = {
                                'sample_id': sample_id,
                                'user_input': user_input,
                                'rater_name': rater_name,
                                'principle': principle,
                                'rating': numeric_rating,
                                'comment': rating_comment
                            }

                            all_comments.append(comment_record)
                            queue_comments += 1

                except Exception as item_error:
                    print(f"    ⚠ Error processing item {object_id}: {item_error}")
                    continue

            print(f"    Extracted: {queue_comments} comments")

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
    else:
        print("  Cache disabled (USE_CACHE=False)")

    if all_comments:
        df = pd.DataFrame(all_comments)

        # Sort by sample_id for easy review
        df = df.sort_values('sample_id')

        print(f"\n✓ Total comments extracted: {len(df)}")
        print(f"  Unique prompts with comments: {df['sample_id'].nunique()}")
        print(f"  Raters who commented: {df['rater_name'].nunique()}")

        return df
    else:
        print("\n⚠ No comments found")
        return pd.DataFrame()


# ============================================================================
# Main Execution
# ============================================================================

def main():
    print("\n" + "="*80)
    print("PROMPT COMMENTS EXTRACTION")
    print("="*80)

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Extract comments
    comments_df = extract_prompt_comments()

    if not comments_df.empty:
        # Save to CSV
        output_path = os.path.join(OUTPUT_DIR, 'prompt_comments.csv')
        comments_df.to_csv(output_path, index=False)

        print("\n" + "="*80)
        print(f"✓ COMMENTS SAVED TO: {output_path}")
        print("="*80)

        # Print sample
        print("\nSample of extracted comments:")
        print("-"*80)
        for idx, row in comments_df.head(5).iterrows():
            print(f"\nPrompt: {row['sample_id']}")
            print(f"  Input: {row['user_input'][:80]}...")
            print(f"  Rater: {row['rater_name']} ({row['principle']})")
            print(f"  Rating: {row['rating']}")
            print(f"  Comment: {row['comment']}")
    else:
        print("\n⚠ No comments to save!")


if __name__ == "__main__":
    main()
