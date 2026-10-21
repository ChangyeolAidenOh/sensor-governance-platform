-- sql/007_graph_state.sql — Stage 4a S4a-03/04 (schema v7, ST4-14..ST4-19). Idempotent. Run AFTER 005_gate_checks.sql
-- (and 006_roles_stage4a.sql on DBs that have the sgp_agent role). Graph-side state only; no Skill contract change.
--   * policy_decisions.decision_key / decision_sha256 — one decision row per (thread, turn, phase); a repeated
--     record with the same key and the same semantic input returns the existing row, a different input is refused
--     (DECISION_MISMATCH). Plain nullable UNIQUE: rows written without a key (goldens, H0 tests) keep NULL.
--   * threads.owner_secret_sha256 / preset — session secret issued by start_session (local demo scope, ST4-16).
--   * graph_turns — turn registry + execution lease (ST4-16): same turn_id + same message = resend, never a second
--     execution; one executor per thread at a time; per-turn result/checkpoint link for recovery.
--   * query_plans — the work planned when a window's single query was spent (ST4-18): what was planned and what
--     completed/failed, so the query state can be derived exactly (unused / in_progress / complete / partial).
INSERT INTO mart.schema_version (version, note) VALUES (7, 'S4a-03/04: decision_key, thread secrets, graph_turns, query_plans')
ON CONFLICT (version) DO NOTHING;

ALTER TABLE mart.policy_decisions ADD COLUMN IF NOT EXISTS decision_key text;
ALTER TABLE mart.policy_decisions ADD COLUMN IF NOT EXISTS decision_sha256 text;
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'policy_decisions_decision_key_uq') THEN
    ALTER TABLE mart.policy_decisions ADD CONSTRAINT policy_decisions_decision_key_uq UNIQUE (decision_key);
  END IF;
END $$;

ALTER TABLE mart.threads ADD COLUMN IF NOT EXISTS owner_secret_sha256 text;
ALTER TABLE mart.threads ADD COLUMN IF NOT EXISTS preset smallint;

CREATE TABLE IF NOT EXISTS mart.graph_turns (
  thread_id       text NOT NULL REFERENCES mart.threads(thread_id),
  turn_id         uuid NOT NULL,
  kind            text NOT NULL CHECK (kind IN ('message','resume')),
  message_sha256  text NOT NULL,
  message         jsonb NOT NULL,                 -- the user message, or the resume payload (decision/modifications)
  run_uuid        uuid NOT NULL,
  status          text NOT NULL CHECK (status IN ('registered','running','waiting_approval','done','failed')),
  executor        uuid,                           -- current lease holder; every write by an executor is conditioned on it
  lease_until     timestamptz,
  checkpoint_id   text,
  result          jsonb,
  created_at      timestamptz NOT NULL DEFAULT now(),
  updated_at      timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (thread_id, turn_id)
);
CREATE INDEX IF NOT EXISTS graph_turns_thread_status_idx ON mart.graph_turns(thread_id, status);

CREATE TABLE IF NOT EXISTS mart.query_plans (
  query_id            uuid PRIMARY KEY,           -- = episode_windows.query_request_id (the spend_query request)
  episode_id          uuid NOT NULL,
  window_index        smallint NOT NULL,
  detector_version    text NOT NULL,
  contract_sha256     text NOT NULL,
  thread_id           text NOT NULL,
  turn_id             uuid NOT NULL,
  initial_decision_id uuid REFERENCES mart.policy_decisions(decision_id),
  kind                text NOT NULL CHECK (kind IN ('expand','rerank','both')),
  planned_expand      boolean NOT NULL,
  planned_rerank      boolean NOT NULL,
  expand_status       text CHECK (expand_status IN ('done','failed','skipped')),
  expand_reason       text,
  rerank_status       text CHECK (rerank_status IN ('done','failed','skipped')),
  rerank_reason       text,
  created_at          timestamptz NOT NULL DEFAULT now(),
  updated_at          timestamptz NOT NULL DEFAULT now()
);

DO $$ BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'sgp_agent') THEN
    GRANT SELECT, INSERT ON mart.graph_turns, mart.query_plans TO sgp_agent;
    GRANT UPDATE (status, executor, lease_until, checkpoint_id, result, run_uuid, updated_at) ON mart.graph_turns TO sgp_agent;
    GRANT UPDATE (expand_status, expand_reason, rerank_status, rerank_reason, updated_at) ON mart.query_plans TO sgp_agent;
    GRANT SELECT ON mart.policy_decisions, mart.episode_windows, mart.alarms, mart.diagnoses, mart.agent_steps TO sgp_agent;
  END IF;
END $$;
