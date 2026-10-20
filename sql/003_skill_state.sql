-- sql/003_skill_state.sql — Stage 4a (S4a-01): base tables (schema v4). Always followed by 004_skill_state_rev2.sql (v5),
-- which carries the rev-2 columns/tables as explicit ALTER/CREATE so that a rev-1 database upgrades too.
INSERT INTO mart.schema_version (version, note) VALUES (4, 'S4a-01: episode_windows, draft/approval binding columns, threads')
ON CONFLICT (version) DO NOTHING;

CREATE TABLE IF NOT EXISTS mart.episode_windows (
  episode_id        uuid     NOT NULL,
  window_index      smallint NOT NULL,
  detector_version  text     NOT NULL,
  contract_sha256   text     NOT NULL,
  gate              text     NOT NULL CHECK (gate IN ('PASS','FLAG','BLOCK')),
  drift             text     NOT NULL CHECK (drift IN ('ok','critical')),
  drift_assessed    boolean  NOT NULL DEFAULT false,
  p_value           double precision,
  p_adj             double precision,
  m_t               integer  NOT NULL,
  alarm             boolean,
  pred_set          smallint[],
  set_size          smallint,
  top1              smallint,
  q_used            boolean  NOT NULL DEFAULT false,
  query_request_id  uuid,                      -- the request that spent the query; its retries may finish the expansion
  expanded_set      smallint[],                -- NULL = no expansion; '{}' = expansion ran and returned an EMPTY set
  observed_at       timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (episode_id, window_index, detector_version, contract_sha256),
  FOREIGN KEY (episode_id, window_index) REFERENCES mart.windows(episode_id, window_index)
);

-- immutable draft versions: one row per (proposal, version); never updated
CREATE TABLE IF NOT EXISTS mart.proposal_drafts (
  proposal_id     uuid     NOT NULL,
  draft_version   integer  NOT NULL,
  content_sha256  text     NOT NULL,
  form            jsonb    NOT NULL,
  created_by      text,
  created_at      timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (proposal_id, draft_version)
);
ALTER TABLE mart.proposals ADD COLUMN IF NOT EXISTS current_version integer NOT NULL DEFAULT 1;   -- pointer only
ALTER TABLE mart.proposals ADD COLUMN IF NOT EXISTS episode_id uuid;
ALTER TABLE mart.proposals ADD COLUMN IF NOT EXISTS window_index smallint;
ALTER TABLE mart.proposals ADD COLUMN IF NOT EXISTS thread_id text;
ALTER TABLE mart.approvals ADD COLUMN IF NOT EXISTS draft_version integer;
ALTER TABLE mart.approvals ADD COLUMN IF NOT EXISTS content_sha256 text;
ALTER TABLE mart.approvals ADD COLUMN IF NOT EXISTS thread_id text;
ALTER TABLE mart.work_orders ADD COLUMN IF NOT EXISTS draft_version integer;
CREATE UNIQUE INDEX IF NOT EXISTS work_orders_proposal_version_uq ON mart.work_orders(proposal_id, draft_version);

CREATE TABLE IF NOT EXISTS mart.threads (
  thread_id    text PRIMARY KEY,
  owner        text NOT NULL,
  episode_id   uuid,
  window_index smallint,
  created_at   timestamptz NOT NULL DEFAULT now()
);
