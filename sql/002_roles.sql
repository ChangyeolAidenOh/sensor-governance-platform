-- sql/002_roles.sql — agent DB role (ST1-02). Run as superuser with:  psql -v agent_password='...' -f sql/002_roles.sql
-- The agent / policy / MCP skill server connects as sgp_agent and can see ONLY the allowed view + execution tables.
-- No access to staging (engine_id, run_id, fault), mart.window_labels, mart.load_log, raw, or checkpointer tables
-- (the LangGraph checkpointer connects as a separate role in Stage 4).
SELECT format('CREATE ROLE sgp_agent LOGIN PASSWORD %L', :'agent_password')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'sgp_agent') \gexec

REVOKE ALL ON SCHEMA raw, staging, mart FROM sgp_agent;
REVOKE ALL ON ALL TABLES IN SCHEMA raw, staging, mart FROM sgp_agent;
REVOKE ALL ON DATABASE sgp_v2 FROM PUBLIC;               -- default PUBLIC grants would otherwise leak CONNECT
GRANT CONNECT ON DATABASE sgp_v2 TO sgp_agent;
GRANT USAGE ON SCHEMA mart TO sgp_agent;
GRANT SELECT ON mart.v_windows_agent TO sgp_agent;       -- view runs with owner rights; staging stays invisible
GRANT SELECT, INSERT ON mart.agent_runs, mart.agent_steps, mart.alarms, mart.diagnoses, mart.evidence,
                        mart.proposals, mart.approvals, mart.work_orders, mart.autonomy_log TO sgp_agent;
GRANT UPDATE (finished_at) ON mart.agent_runs TO sgp_agent;
GRANT UPDATE (token_used) ON mart.approvals TO sgp_agent;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA mart TO sgp_agent;
-- replay tables are written by the replayer (loader role), read by the evaluator; agent gets nothing there.
