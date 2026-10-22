-- sql/009_roles_checkpointer.sql — sgp_agent grants on the LangGraph PostgresSaver tables (ST4-24, S4a-06).
-- Run as superuser AFTER PostgresSaver.setup() has created the tables (setup() is an admin step; the demo never calls it)
-- and after 006/008 (reapply those first if a test run recreated mart tables: scripts/apply_role_grants.py).
-- Verbs follow the installed package's SQL for the regular PostgresSaver (not the shallow saver):
--   checkpoints / checkpoint_blobs / checkpoint_writes : SELECT + INSERT (+ UPDATE for the ON CONFLICT ... DO UPDATE upserts)
--   checkpoint_migrations                              : SELECT only (setup() is the only writer)
-- DELETE is NOT granted here: delete_thread is never called by the demo. If the base.py SQL used by put()/put_writes()
-- turns out to DELETE superseded rows (to be read from the dump / role write-probe), add it per table with the function cited.
GRANT USAGE ON SCHEMA public TO sgp_agent;
GRANT SELECT ON public.checkpoint_migrations TO sgp_agent;
GRANT SELECT, INSERT, UPDATE ON public.checkpoints TO sgp_agent;
GRANT SELECT, INSERT, UPDATE ON public.checkpoint_blobs TO sgp_agent;
GRANT SELECT, INSERT, UPDATE ON public.checkpoint_writes TO sgp_agent;
