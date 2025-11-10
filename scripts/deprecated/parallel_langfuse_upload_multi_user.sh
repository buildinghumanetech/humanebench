#!/bin/bash
# Parallel upload to all users' Langfuse annotation queues for all principles
#
# Usage:
#   bash scripts/parallel_langfuse_upload_multi_user.sh queue_ids.csv [--sample-limit 100]
#
# Arguments:
#   queue_ids.csv: Path to CSV file with user queue IDs
#   --sample-limit N: Optional sample limit per principle (default: 100)

# Check if queue CSV path is provided
if [ -z "$1" ]; then
    echo "Error: Queue CSV file path required"
    echo "Usage: bash scripts/parallel_langfuse_upload_multi_user.sh queue_ids.csv [--sample-limit 100]"
    exit 1
fi

QUEUE_CSV="$1"
shift  # Remove first argument

# Check if file exists
if [ ! -f "$QUEUE_CSV" ]; then
    echo "Error: Queue CSV file not found: $QUEUE_CSV"
    exit 1
fi

# Default sample limit
SAMPLE_LIMIT="100"

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --sample-limit)
            SAMPLE_LIMIT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Starting parallel upload for all principles..."
echo "Queue CSV: $QUEUE_CSV"
echo "Sample limit: $SAMPLE_LIMIT"
echo ""

# Run uploads for all principles in parallel
for principle in be-transparent-and-honest design-for-equity-and-inclusion enable-meaningful-choices enhance-human-capabilities foster-healthy-relationships prioritize-long-term-wellbeing protect-dignity-and-safety respect-user-attention;
do
    python scripts/inspect_logs_to_langfuse_multi_user.py \
        --principle $principle \
        --queue-csv "$QUEUE_CSV" \
        --sample-limit $SAMPLE_LIMIT \
        --enqueue &
done

# Wait for all background jobs to complete
wait

echo ""
echo "All parallel uploads complete!"
