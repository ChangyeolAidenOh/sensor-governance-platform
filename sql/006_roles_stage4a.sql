-- sql/006_roles_stage4a.sql — grants for the Stage 4a tables (ST4-11). Run as superuser after 002_roles.sql.
-- sgp_agent (Skill servers / graph in deployment, handoff §7): read the window grid view, write execution rows.
-- Stage 7 item: the deployed DB gets ONLY this role for the app; the superuser never goes into app secrets.
GRANT SELECT, INSERT ON mart.episode_windows, mart.gate_checks, mart.proposal_drafts, mart.policy_decisions, mart.threads TO sgp_agent;
GRANT UPDATE (q_used, query_request_id, expanded_set) ON mart.episode_windows TO sgp_agent;
GRANT UPDATE (current_version, request_form) ON mart.proposals TO sgp_agent;
GRANT UPDATE (status, reason_code, output, latency_ms, violation, created_at) ON mart.agent_steps TO sgp_agent;   -- request ownership (ST4-13)
GRANT USAGE ON ALL SEQUENCES IN SCHEMA mart TO sgp_agent;
-- The detect server needs the episode→engine key map to read the episode's parquet rows. It is a trusted backend
-- (no LLM can call SQL); in development it connects as postgres. For deployment the demo episodes are pre-computed
-- (S4a-03) and the chat path reads stored rows only, so this grant is NOT given to sgp_agent.
