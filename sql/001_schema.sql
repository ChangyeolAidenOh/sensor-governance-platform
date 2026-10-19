-- sql/001_schema.sql — sensor-governance-platform v2 mart (Stage 1, S1-02, rev 2)
-- Idempotent. Tiers (ST1-01): raw/processed = DVC parquet (not in Postgres); staging = episode registry, split,
-- key map; mart = window grid + agent/replay tables keyed by OPAQUE episode_id. Labels and engine_id
-- (which encodes fault_scenario) live only in staging / mart.window_labels, which the agent role cannot read
-- (sql/002_roles.sql). Derived tables (episodes, windows, labels) are rebuilt by scripts/load_mart.py; execution
-- tables (alarms … replay_window_log) and checkpointer tables are never touched by a load (ST1-03).

CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS mart;

CREATE TABLE IF NOT EXISTS mart.schema_version (
  version    integer PRIMARY KEY,
  applied_at timestamptz NOT NULL DEFAULT now(),
  note       text
);
INSERT INTO mart.schema_version (version, note)
VALUES (2, 'S1-02 rev2: opaque episode_id, agent role, derived/execution separation')
ON CONFLICT (version) DO NOTHING;

-- ------------------------------------------------------------ raw: identity of the parquet actually used (ST1-01)
CREATE TABLE IF NOT EXISTS raw.tep_files (
  file_name       text PRIMARY KEY,         -- faulty_testing.parquet …
  split           text NOT NULL,
  faulty          boolean NOT NULL,
  rows            bigint,
  episodes        integer,
  size_bytes      bigint,                   -- from <file>.done.json sidecar
  written_at      text,
  raw_size        bigint, raw_mtime bigint, -- identity of the RData the parquet was derived from
  loader_version  text,
  manifest_sha256 text NOT NULL,
  dvc_md5         text,                     -- outs md5 of data/processed/TEP.dvc if present
  registered_at   timestamptz NOT NULL DEFAULT now()
);

-- ------------------------------------------------------------ staging (loader/evaluator only)
-- engine_id ↔ episode_id map. NEVER truncated: a key is minted once (uuid4) and reused by every later load so
-- execution records keep their references. Agent role has no access to this schema.
CREATE TABLE IF NOT EXISTS staging.episode_keys (
  engine_id  integer PRIMARY KEY,
  episode_id uuid    NOT NULL UNIQUE DEFAULT gen_random_uuid(),
  minted_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS staging.episodes (
  episode_id      uuid    PRIMARY KEY REFERENCES staging.episode_keys(episode_id),
  engine_id       integer NOT NULL UNIQUE REFERENCES staging.episode_keys(engine_id),
  split           text    NOT NULL CHECK (split IN ('training','testing')),
  run_id          integer NOT NULL,
  fault_scenario  smallint NOT NULL CHECK (fault_scenario BETWEEN 0 AND 20),
  run_length      integer NOT NULL,
  "group"         text    NOT NULL CHECK ("group" IN ('dev','calibration','evaluation_pool','demo_pool')),
  eval_rank       integer,
  eval_selected   boolean NOT NULL DEFAULT false,
  demo_preset     smallint,
  UNIQUE (split, run_id, fault_scenario)
);
CREATE INDEX IF NOT EXISTS episodes_group_idx ON staging.episodes ("group", eval_selected);

-- ------------------------------------------------------------ mart: window grid (opaque key, no labels)
CREATE TABLE IF NOT EXISTS mart.windows (
  episode_id   uuid     NOT NULL REFERENCES staging.episodes(episode_id),
  window_index smallint NOT NULL CHECK (window_index >= 0),
  start_cycle  integer  NOT NULL,
  end_cycle    integer  NOT NULL,           -- max_sample for this window
  PRIMARY KEY (episode_id, window_index),
  CHECK (end_cycle - start_cycle + 1 = 20)
);

-- labels: evaluator only. Same key as windows; agent role has no SELECT here.
CREATE TABLE IF NOT EXISTS mart.window_labels (
  episode_id      uuid     NOT NULL,
  window_index    smallint NOT NULL,
  fault_scenario  smallint NOT NULL,
  is_fault_active boolean  NOT NULL,
  PRIMARY KEY (episode_id, window_index),
  FOREIGN KEY (episode_id, window_index) REFERENCES mart.windows(episode_id, window_index)
);

-- agent-facing view: opaque episode_id, grid, group, demo preset number. No engine_id, run_id, fault, labels.
-- (demo preset→fault mapping is public in the pre-registration, but the demo is not an evaluation surface.)
CREATE OR REPLACE VIEW mart.v_windows_agent AS
SELECT w.episode_id, w.window_index, w.start_cycle, w.end_cycle, e."group", e.demo_preset
FROM mart.windows w JOIN staging.episodes e USING (episode_id);

-- ------------------------------------------------------------ mart: load bookkeeping (ST1-01, ST1-03)
CREATE TABLE IF NOT EXISTS mart.load_log (
  load_id            bigserial PRIMARY KEY,
  loaded_at          timestamptz NOT NULL DEFAULT now(),
  schema_version     integer NOT NULL REFERENCES mart.schema_version(version),
  prereg_commit      text    NOT NULL,     -- core_pipeline.prereg_constants.PREREG_COMMIT
  split_json_sha256  text    NOT NULL,
  split_seed         integer NOT NULL,
  manifest_sha256    text,                 -- NULL only when --manifest was not given (dev)
  loader_version     text,
  dvc_md5            text,
  grid_sha256        text    NOT NULL,     -- content hash of (episodes, windows, labels) as derived
  prereg_n           integer NOT NULL,
  n_reduction_decision text,               -- decision ID when prereg_n = 40 (§3.1 time contract)
  window_size        integer NOT NULL,
  stride             integer NOT NULL,
  action             text    NOT NULL CHECK (action IN ('rebuilt','unchanged','refused')),
  counts             jsonb   NOT NULL,
  ok                 boolean NOT NULL DEFAULT false
);

-- ------------------------------------------------------------ mart: execution tables (never truncated by a load)
CREATE TABLE IF NOT EXISTS mart.agent_runs (
  run_uuid    uuid PRIMARY KEY,
  thread_id   text NOT NULL,
  mode        text NOT NULL CHECK (mode IN ('chat','event','replay')),
  llm_model   text, effort text,
  started_at  timestamptz NOT NULL DEFAULT now(),
  finished_at timestamptz
);

CREATE TABLE IF NOT EXISTS mart.agent_steps (
  step_id      bigserial PRIMARY KEY,
  run_uuid     uuid NOT NULL REFERENCES mart.agent_runs(run_uuid),
  request_id   uuid NOT NULL UNIQUE,
  seq          integer NOT NULL,
  skill        text NOT NULL,
  status       text NOT NULL,
  reason_code  text,
  input        jsonb NOT NULL,
  output       jsonb,
  latency_ms   integer, input_tokens integer, output_tokens integer, cost_usd numeric(10,6),
  violation    boolean NOT NULL DEFAULT false,
  created_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS agent_steps_run_idx ON mart.agent_steps(run_uuid, seq);

CREATE TABLE IF NOT EXISTS mart.alarms (
  alarm_id     text PRIMARY KEY,            -- '<episode_id>:<window_index>'
  episode_id   uuid     NOT NULL,
  window_index smallint NOT NULL,
  gate         text NOT NULL CHECK (gate IN ('PASS','FLAG','BLOCK')),
  drift        text NOT NULL CHECK (drift IN ('ok','critical')),
  p_value      double precision, p_adj double precision, m_t integer,
  alarm        boolean NOT NULL,
  detector     text NOT NULL,
  created_at   timestamptz NOT NULL DEFAULT now(),
  FOREIGN KEY (episode_id, window_index) REFERENCES mart.windows(episode_id, window_index)
);

CREATE TABLE IF NOT EXISTS mart.diagnoses (
  alarm_id           text PRIMARY KEY REFERENCES mart.alarms(alarm_id),
  calibration_window smallint NOT NULL CHECK (calibration_window IN (20, 40)),
  pred_set           smallint[] NOT NULL,
  set_size           smallint NOT NULL,
  top1               smallint, top1_prob double precision,
  contributions      jsonb
);

CREATE TABLE IF NOT EXISTS mart.evidence (
  evidence_id bigserial PRIMARY KEY,
  alarm_id    text NOT NULL REFERENCES mart.alarms(alarm_id),
  mode        text NOT NULL CHECK (mode IN ('scoped','open')),
  k           smallint NOT NULL,
  clause_key  text NOT NULL,
  grade       text NOT NULL CHECK (grade IN ('A','B','C','OUT')),
  rank        smallint NOT NULL,
  score       double precision NOT NULL,
  tau         double precision NOT NULL
);
CREATE INDEX IF NOT EXISTS evidence_alarm_idx ON mart.evidence(alarm_id, mode);

CREATE TABLE IF NOT EXISTS mart.proposals (
  proposal_id  uuid PRIMARY KEY,
  alarm_id     text NOT NULL REFERENCES mart.alarms(alarm_id),
  policy       text NOT NULL CHECK (policy IN ('E','B1','B2','B3')),
  action       text NOT NULL CHECK (action IN ('propose','hold','query_more','undecidable','no_alarm')),
  hold_reason  text, ev_ab boolean, request_form jsonb,
  created_at   timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS mart.approvals (
  approval_id    bigserial PRIMARY KEY,
  proposal_id    uuid NOT NULL REFERENCES mart.proposals(proposal_id),
  approver       text NOT NULL,
  decision       text NOT NULL CHECK (decision IN ('approve','modify','reject','override')),
  approval_token uuid UNIQUE,
  token_used     boolean NOT NULL DEFAULT false,
  modifications  jsonb,
  decided_at     timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS mart.work_orders (
  work_order_id   uuid PRIMARY KEY,
  proposal_id     uuid   NOT NULL REFERENCES mart.proposals(proposal_id),
  approval_id     bigint NOT NULL REFERENCES mart.approvals(approval_id),   -- no work order without approval
  cmms_ref        text, idempotency_key text UNIQUE,
  status          text NOT NULL CHECK (status IN ('issued','cancelled')),
  issued_at       timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS mart.autonomy_log (
  log_id bigserial PRIMARY KEY, level_from text NOT NULL, level_to text NOT NULL,
  trigger text NOT NULL, evidence jsonb, at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS mart.replay_runs (
  replay_id            uuid PRIMARY KEY,
  seed                 integer NOT NULL,
  prereg_n             integer NOT NULL,
  load_id              bigint  NOT NULL REFERENCES mart.load_log(load_id),   -- the grid/inputs this replay ran against
  run_id_order_sha256  text    NOT NULL,
  window_size integer NOT NULL, stride integer NOT NULL,
  narrate     boolean NOT NULL DEFAULT false,
  detector    text NOT NULL, code_commit text,
  started_at  timestamptz NOT NULL DEFAULT now(), finished_at timestamptz,
  wall_seconds_first20 double precision
);

CREATE TABLE IF NOT EXISTS mart.replay_window_log (
  replay_id    uuid NOT NULL REFERENCES mart.replay_runs(replay_id),
  policy       text NOT NULL CHECK (policy IN ('E','B1','B2','B3')),
  episode_id   uuid     NOT NULL,
  window_index smallint NOT NULL,
  gate text, drift text,
  p_value double precision, p_adj double precision, m_t integer, alarm boolean,
  set_size smallint, valid_s boolean,
  g_ab double precision, g_c double precision, ev_ab boolean, ev_c boolean,
  q_used boolean NOT NULL DEFAULT false,
  action text NOT NULL CHECK (action IN ('propose','hold','query_more','undecidable','no_alarm')),
  hold_reason text,
  PRIMARY KEY (replay_id, policy, episode_id, window_index),
  FOREIGN KEY (episode_id, window_index) REFERENCES mart.windows(episode_id, window_index)
);
