#!/bin/bash
# Parallel upload for late-arriving evaluators - uses reference user's exact samples
#
# This script populates all queues for late-arriving users by copying the exact
# samples from a reference user who already completed their uploads.
#
# Usage:
#   bash scripts/parallel_langfuse_upload_late_arrivals.sh erika.csv "rater-1"
#
# Arguments:
#   queue_ids.csv: Path to CSV file with user queue IDs (must include reference user)
#   reference_user: Name of user whose samples to copy (e.g., "rater-1")

# Check arguments
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Error: Both CSV file and reference user name required"
    echo "Usage: bash scripts/parallel_langfuse_upload_late_arrivals.sh queue_ids.csv \"Reference User Name\""
    exit 1
fi

QUEUE_CSV="$1"
REFERENCE_USER="$2"

# Check if file exists
if [ ! -f "$QUEUE_CSV" ]; then
    echo "Error: Queue CSV file not found: $QUEUE_CSV"
    exit 1
fi

echo "Starting parallel upload for late arrivals..."
echo "Queue CSV: $QUEUE_CSV"
echo "Reference user: $REFERENCE_USER"
echo ""
echo "This will populate all users' queues with the exact samples from $REFERENCE_USER's queues."
echo ""

# Run uploads for all principles in parallel
for principle in be-transparent-and-honest design-for-equity-and-inclusion enable-meaningful-choices enhance-human-capabilities foster-healthy-relationships prioritize-long-term-wellbeing protect-dignity-and-safety respect-user-attention;
do
    python scripts/inspect_logs_to_langfuse_multi_user.py \
        --principle $principle \
        --queue-csv "$QUEUE_CSV" \
        --reference-user "$REFERENCE_USER" \
        --enqueue &
done

# Wait for all background jobs to complete
wait

echo ""
echo "All parallel uploads complete!"
echo "All users in $QUEUE_CSV now have the same samples as $REFERENCE_USER"
