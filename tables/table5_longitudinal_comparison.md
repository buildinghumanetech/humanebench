# Table 5: Longitudinal Comparison Across Labs

Tracking humaneness and adversarial robustness across model generations from Anthropic, OpenAI, Google, and Meta.

| Lab       | Model                   |   Generation |   Baseline HumaneScore |   Good Persona HumaneScore |   Bad Persona HumaneScore |   Good Δ |   Bad Δ | Robustness Status   |
|:----------|:------------------------|-------------:|-----------------------:|---------------------------:|--------------------------:|---------:|--------:|:--------------------|
| Anthropic | claude-sonnet-4         |            1 |                   0.69 |                       0.84 |                      0.51 |     0.15 |   -0.18 | Moderate            |
| Anthropic | claude-sonnet-4.5       |            2 |                   0.75 |                       0.89 |                      0.77 |     0.14 |    0.02 | Robust              |
| Anthropic | claude-opus-4.1         |            3 |                   0.68 |                       0.84 |                      0.66 |     0.16 |   -0.02 | Robust              |
| OpenAI    | gpt-4o-2024-11-20       |            1 |                   0.68 |                       0.77 |                     -0.61 |     0.09 |   -1.29 | Failed              |
| OpenAI    | gpt-4.1                 |            2 |                   0.67 |                       0.82 |                     -0.6  |     0.15 |   -1.27 | Failed              |
| OpenAI    | gpt-5                   |            3 |                   0.86 |                       0.94 |                      0.83 |     0.08 |   -0.03 | Robust              |
| OpenAI    | gpt-5.1                 |            4 |                   0.86 |                       0.92 |                      0.82 |     0.06 |   -0.04 | Robust              |
| Google    | gemini-2.0-flash-001    |            1 |                   0.75 |                       0.82 |                     -0.71 |     0.07 |   -1.46 | Failed              |
| Google    | gemini-2.5-flash        |            2 |                   0.72 |                       0.77 |                     -0.68 |     0.05 |   -1.4  | Failed              |
| Google    | gemini-2.5-pro          |            3 |                   0.77 |                       0.9  |                     -0.72 |     0.13 |   -1.49 | Failed              |
| Google    | gemini-3-pro-preview    |            4 |                   0.78 |                       0.93 |                     -0.45 |     0.15 |   -1.23 | Failed              |
| Meta      | llama-3.1-405b-instruct |            1 |                   0.56 |                       0.68 |                     -0.49 |     0.12 |   -1.05 | Failed              |
| Meta      | llama-4-maverick        |            2 |                   0.59 |                       0.65 |                     -0.14 |     0.06 |   -0.73 | Failed              |