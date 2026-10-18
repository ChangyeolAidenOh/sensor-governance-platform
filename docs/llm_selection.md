# 0-F LLM selection (spike result)

Runs: 120 · rule: zero format errors and zero exceptions -> min cost per request -> min latency per request

| model | runs | pass_rate | format_error_rate | expected_tools_rate | forbidden_tool_runs | interrupt_resume_rate | oos_refusal_rate | exceptions | cost_per_request_usd | cost_per_run_usd | requests_per_run | output_tokens_per_request | latency_per_request_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| claude-haiku-5-5 | 30 | 0.8 | 0.0 | 0.8 | 0 | 0.333 | 1.0 | 0 | 0.000311 | 0.0008 | 2.5 | 232.9 | 1.6967 |
| claude-sonnet-5-5 | 30 | 0.8 | 0.0 | 0.8 | 0 | 0.333 | 1.0 | 0 | 0.005787 | 0.0145 | 2.5 | 194.5 | 2.2888 |
| gemini-4-argon | 30 | 1.0 | 0.0 | 1.0 | 0 | 1.0 | 1.0 | 0 | 0.003102 | 0.0084 | 2.7 | 92.2 | 13.2359 |
| gpt-6.1-sol | 30 | 0.833 | 0.0 | 0.833 | 0 | 0.444 | 1.0 | 0 | 0.002506 | 0.0064 | 2.5667 | 81.1 | 3.4885 |

**Eligible (0 format errors, 0 exceptions):** ['claude-haiku-5-5', 'claude-sonnet-5-5', 'gemini-4-argon', 'gpt-6.1-sol']
**Ranking:** ['claude-haiku-5-5', 'gpt-6.1-sol', 'gemini-4-argon', 'claude-sonnet-5-5']
**Selected:** claude-haiku-5-5
**Judge (different vendor, cheapest):** gpt-6.1-sol

