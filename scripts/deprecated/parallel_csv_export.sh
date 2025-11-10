#!/bin/bash
# Parallel CSV export for all principles

for principle in be-transparent-and-honest design-for-equity-and-inclusion enable-meaningful-choices enhance-human-capabilities foster-healthy-relationships prioritize-long-term-wellbeing protect-dignity-and-safety respect-user-attention;
    do python scripts/inspect_logs_to_langfuse.py --principle $principle --sample-limit 100 --export-csv &
done
wait
