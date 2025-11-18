# Table 1: Comprehensive Steerability Summary

Shows baseline performance, response to humane-aligned prompting (Good Persona), and adversarial robustness (Bad Persona) across all 13 models.

| Model                   |   Baseline HumaneScore |   Good Persona HumaneScore |   Good Δ |   Bad Persona HumaneScore |   Bad Δ | Robustness Status   |
|:------------------------|-----------------------:|---------------------------:|---------:|--------------------------:|--------:|:--------------------|
| claude-opus-4.1         |                   0.68 |                       0.84 |     0.16 |                      0.66 |   -0.02 | Robust              |
| claude-sonnet-4         |                   0.69 |                       0.84 |     0.15 |                      0.51 |   -0.18 | Moderate            |
| claude-sonnet-4.5       |                   0.75 |                       0.89 |     0.14 |                      0.77 |    0.02 | Robust              |
| deepseek-v3.1-terminus  |                   0.75 |                       0.84 |     0.09 |                     -0.35 |   -1.1  | Failed              |
| gemini-2.0-flash-001    |                   0.75 |                       0.82 |     0.07 |                     -0.71 |   -1.46 | Failed              |
| gemini-2.5-flash        |                   0.72 |                       0.77 |     0.05 |                     -0.68 |   -1.4  | Failed              |
| gemini-2.5-pro          |                   0.77 |                       0.9  |     0.13 |                     -0.72 |   -1.49 | Failed              |
| gemini-3-pro-preview    |                   0.78 |                       0.93 |     0.15 |                     -0.45 |   -1.23 | Failed              |
| gpt-4.1                 |                   0.67 |                       0.82 |     0.15 |                     -0.6  |   -1.27 | Failed              |
| gpt-4o-2024-11-20       |                   0.68 |                       0.77 |     0.09 |                     -0.61 |   -1.29 | Failed              |
| gpt-5                   |                   0.86 |                       0.94 |     0.08 |                      0.83 |   -0.03 | Robust              |
| grok-4                  |                   0.69 |                       0.89 |     0.2  |                     -0.73 |   -1.42 | Failed              |
| llama-3.1-405b-instruct |                   0.56 |                       0.68 |     0.12 |                     -0.49 |   -1.05 | Failed              |
| llama-4-maverick        |                   0.59 |                       0.65 |     0.06 |                     -0.14 |   -0.73 | Failed              |