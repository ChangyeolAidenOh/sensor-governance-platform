-- sql/010_policy_isolation.sql — Stage 4b S4b-01 (schema v8, ST4-50): per-policy state isolation. Idempotent.
-- Run AFTER 007_graph_state.sql.
--   * policy_input_snapshots — the FIRST observation of a window (detection columns + the first OK scoped k=5
--     envelope), materialised once BEFORE the window's query is spent. Policies B1/B3 and E-initial read only
--     this row. There is deliberately no expanded_set / q_used / k=10 column: the baseline path cannot read them.
--     The freeze moment is guaranteed by the driver transaction (advisory lock + source-state check), not by
--     a CHECK constraint.
--   * policy_events — the per-(eval run, policy, window, phase) action ledger the H1 cost is computed from.
--     It carries NO foreign keys: snapshot_sha256 / decision_id / decision_sha256 are provenance identifiers,
--     so the ledger survives clean_episode (which deletes policy_decisions and snapshots).
--   * eval_runs — one row per evaluation run; at most ONE active (running / incomplete) run per database,
--     enforced by a partial unique index, not by a SELECT.
INSERT INTO mart.schema_version (version, note) VALUES (8, 'S4b-01: policy_input_snapshots, policy_events, eval_runs (ST4-50)')
ON CONFLICT (version) DO NOTHING;

CREATE TABLE IF NOT EXISTS mart.policy_input_snapshots (
  snapshot_id        uuid        NOT NULL DEFAULT gen_random_uuid(),
  episode_id         uuid        NOT NULL,
  window_index       smallint    NOT NULL,
  detector_version   text        NOT NULL,
  contract_sha256    text        NOT NULL,
  caller_role        text        NOT NULL,
  gate               text        NOT NULL CHECK (gate IN ('PASS','FLAG','BLOCK')),
  drift              text        NOT NULL CHECK (drift IN ('ok','critical')),
  drift_assessed     boolean     NOT NULL,
  p_value            double precision,
  p_adj              double precision,
  m_t                integer     NOT NULL,
  alarm              boolean,
  alarm_id           text        REFERENCES mart.alarms(alarm_id),
  pred_set           smallint[]  NOT NULL,
  set_size           smallint    NOT NULL,
  top1               smallint,
  k5_request_id      uuid,
  k5_fault_set       smallint[],
  k5_mode            text,
  k5_tau             double precision,
  k5_g_ab            double precision,
  k5_g_c             double precision,
  k5_ev_ab           boolean,
  k5_ev_c            boolean,
  k5_snapshot        jsonb,
  k5_versions        jsonb,
  content_sha256     text        NOT NULL,
  frozen_by_eval_run uuid,
  frozen_at          timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (episode_id, window_index, detector_version, contract_sha256, caller_role),
  FOREIGN KEY (episode_id, window_index, detector_version, contract_sha256)
    REFERENCES mart.episode_windows(episode_id, window_index, detector_version, contract_sha256),
  CHECK ((alarm IS TRUE AND alarm_id IS NOT NULL AND k5_request_id IS NOT NULL AND k5_snapshot IS NOT NULL)
      OR (alarm IS NOT TRUE AND alarm_id IS NULL AND k5_request_id IS NULL))
);
CREATE UNIQUE INDEX IF NOT EXISTS policy_input_snapshots_id_uq ON mart.policy_input_snapshots(snapshot_id);

CREATE TABLE IF NOT EXISTS mart.eval_runs (
  eval_run_id           uuid        PRIMARY KEY,
  started_at            timestamptz NOT NULL DEFAULT now(),
  code_commit           text        NOT NULL,
  prereg_tag            text,
  seed                  integer,
  planned_episodes      jsonb       NOT NULL,
  planned_sha256        text        NOT NULL,
  n_run_ids             integer,
  detector_version      text        NOT NULL,
  contract_sha256       text        NOT NULL,
  expected_k5_versions  jsonb       NOT NULL,
  caller_role           text        NOT NULL,
  status                text        NOT NULL CHECK (status IN ('running','incomplete','complete','archived')),
  timing                jsonb,
  events_export_sha256  text,
  updated_at            timestamptz NOT NULL DEFAULT now()
);
-- at most one active run per database (concurrent starts: exactly one INSERT succeeds)
CREATE UNIQUE INDEX IF NOT EXISTS eval_runs_single_active_uq ON mart.eval_runs ((true)) WHERE status IN ('running','incomplete');

CREATE TABLE IF NOT EXISTS mart.policy_events (
  eval_run_id          uuid        NOT NULL,
  policy               text        NOT NULL CHECK (policy IN ('E','B1','B2','B3')),
  episode_id           uuid        NOT NULL,
  window_index         smallint    NOT NULL,
  phase                text        NOT NULL CHECK (phase IN ('initial','after_query')),
  detector_version     text        NOT NULL,
  contract_sha256      text        NOT NULL,
  snapshot_sha256      text,
  alarm                boolean     NOT NULL,
  alarm_id             text,
  gate                 text,
  action               text        NOT NULL CHECK (action IN ('undecidable','no_alarm','hold','propose','query_more')),
  rule                 smallint    NOT NULL,
  reason               text,
  query_kind           text,
  signals              jsonb       NOT NULL,
  signals_sha256       text        NOT NULL,
  evidence             jsonb,
  evidence_stage       text        NOT NULL CHECK (evidence_stage IN ('none','k5','k10')),
  evidence_k           smallint,
  evidence_request_id  uuid,
  decision_id          uuid,
  decision_sha256      text,
  event_sha256         text        NOT NULL,
  recorded_at          timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (eval_run_id, policy, episode_id, window_index, phase),
  CHECK (phase = 'initial' OR policy = 'E')
);
CREATE INDEX IF NOT EXISTS policy_events_run_episode_idx ON mart.policy_events(eval_run_id, episode_id);

DO $$ BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'sgp_agent') THEN
    GRANT SELECT ON mart.policy_input_snapshots, mart.policy_events, mart.eval_runs TO sgp_agent;
  END IF;
END $$;
