# Security policy

## Report a vulnerability

Email info@buildinghumanetech.com with "SECURITY" in the subject. Please do not open a public issue for anything that exposes API keys, user data or a way to tamper with published results.

We aim to acknowledge a report within 5 business days and tell you what we plan to do within 14 days. We do not run a bounty program.

## What is in scope

- Code in this repository (the evaluation harness, scorers, scripts).
- Anything that could leak credentials or evaluation transcripts, such as logs that contain a system prompt.
- Ways to alter published scores or rubric files without detection.

## What is not in scope

- Models producing harmful output under test. That is what the benchmark measures, not a vulnerability in it.
- Disagreements with a rubric or a score. Open an issue instead.

## Supported versions

Only the `main` branch. Rubric versions are frozen once published; see [rubrics/README.md](rubrics/README.md).
