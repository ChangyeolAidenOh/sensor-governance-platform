-- sql/004_skill_state_rev2.sql — migration rev 1 → rev 2 of the S4a-01 state (schema v5). Idempotent.
-- Run AFTER 003_skill_state.sql on any DB (fresh or rev-1). Development-only proposal data from rev 1 (one mutable
-- row per proposal, no draft versions) is DISCARDED here: it was smoke data and cannot be re-hashed identically in SQL.
INSERT INTO mart.schema_version (version, note) VALUES (5, 'S4a-01 rev2: query_request_id, immutable proposal_drafts, policy_decisions; rev-1 dev proposals discarded')
ON CONFLICT (version) DO NOTHING;

ALTER TABLE mart.episode_windows ADD COLUMN IF NOT EXISTS query_request_id uuid;
ALTER TABLE mart.episode_windows ADD COLUMN IF NOT EXISTS expanded_set smallint[];

CREATE TABLE IF NOT EXISTS mart.proposal_drafts (
  proposal_id     uuid     NOT NULL,
  draft_version   integer  NOT NULL,
  content_sha256  text     NOT NULL,
  form            jsonb    NOT NULL,
  created_by      text,
  created_at      timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (proposal_id, draft_version)
);
ALTER TABLE mart.proposals ADD COLUMN IF NOT EXISTS current_version integer NOT NULL DEFAULT 1;
ALTER TABLE mart.proposals ADD COLUMN IF NOT EXISTS episode_id uuid;
ALTER TABLE mart.proposals ADD COLUMN IF NOT EXISTS window_index smallint;
ALTER TABLE mart.proposals ADD COLUMN IF NOT EXISTS thread_id text;
ALTER TABLE mart.approvals ADD COLUMN IF NOT EXISTS draft_version integer;
ALTER TABLE mart.approvals ADD COLUMN IF NOT EXISTS content_sha256 text;
ALTER TABLE mart.approvals ADD COLUMN IF NOT EXISTS thread_id text;
ALTER TABLE mart.work_orders ADD COLUMN IF NOT EXISTS draft_version integer;
CREATE UNIQUE INDEX IF NOT EXISTS work_orders_proposal_version_uq ON mart.work_orders(proposal_id, draft_version);

-- the server's record of the policy decision per (alarm, policy, signal snapshot): the CMMS only drafts from a 'propose'
CREATE TABLE IF NOT EXISTS mart.policy_decisions (
  decision_id   uuid PRIMARY KEY,
  alarm_id      text NOT NULL REFERENCES mart.alarms(alarm_id),
  policy        text NOT NULL CHECK (policy IN ('E','B1','B2','B3')),
  signals_sha256 text NOT NULL,
  signals       jsonb NOT NULL,
  action        text NOT NULL CHECK (action IN ('undecidable','no_alarm','hold','propose','query_more')),
  rule          smallint NOT NULL,
  evidence      jsonb,                                   -- snapshot of the scoped retrieve the decision used: [{clause_key, grade, score}]
  decided_at    timestamptz NOT NULL DEFAULT now(),
  seq           bigserial                                -- ordering within (alarm, policy): the latest decision is the only valid one
);
ALTER TABLE mart.policy_decisions ADD COLUMN IF NOT EXISTS evidence jsonb;
ALTER TABLE mart.policy_decisions ADD COLUMN IF NOT EXISTS seq bigserial;
CREATE INDEX IF NOT EXISTS policy_decisions_alarm_idx ON mart.policy_decisions(alarm_id, policy);

-- discard rev-1 development proposals that have no immutable draft (cannot be migrated faithfully)
DELETE FROM mart.work_orders w USING mart.proposals p WHERE w.proposal_id = p.proposal_id
  AND NOT EXISTS (SELECT 1 FROM mart.proposal_drafts d WHERE d.proposal_id = p.proposal_id);
DELETE FROM mart.approvals a USING mart.proposals p WHERE a.proposal_id = p.proposal_id
  AND NOT EXISTS (SELECT 1 FROM mart.proposal_drafts d WHERE d.proposal_id = p.proposal_id);
DELETE FROM mart.proposals p WHERE NOT EXISTS (SELECT 1 FROM mart.proposal_drafts d WHERE d.proposal_id = p.proposal_id);
