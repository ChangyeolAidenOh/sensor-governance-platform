-- sql/011_roles_public_skill.sql: second login role for the PUBLIC demo (demo-public branch). Run AFTER 002 with:
--   psql -v skill_password='...' -f sql/011_roles_public_skill.sql
-- The deployed app connects as sgp_agent (SGP_AGENT_PG_DSN) and the skills_detect server as sgp_skill (SGP_PG_DSN).
-- runtime.check_dsns (ST4-30) requires the two users to differ, and ST4-11 requires the deployed database to expose
-- nothing beyond the agent privileges: sgp_skill is a member of sgp_agent and inherits exactly its grants (no staging,
-- no window_labels, no load_log, no raw), nothing of its own. The admin role never goes into app secrets.
SELECT format('CREATE ROLE sgp_skill LOGIN PASSWORD %L', :'skill_password')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'sgp_skill') \gexec

GRANT sgp_agent TO sgp_skill;
