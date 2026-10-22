-- S4a-05 (ST4-21): the LLM ledger settles actual usage into three more agent_steps columns.
-- Column-level, additive to 006 (status, reason_code, output, latency_ms, violation, created_at); input/request_id/seq stay read-only.
GRANT UPDATE (input_tokens, output_tokens, cost_usd) ON mart.agent_steps TO sgp_agent;
