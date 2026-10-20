-- sql/005_gate_checks.sql — Stage 4a S4a-00/01 (schema v6, ST4-11). Idempotent. Run AFTER 004_skill_state_rev2.sql.
-- Server-stored gate results. detect_diagnose does NOT trust a caller-supplied gate dict: it reads the row the detect
-- server itself stored for the same (episode, window) under the SAME input contract, gate version and calibration
-- file sha (reviewer 2026-10-21 (c): a PASS computed under an older calibration must not admit a detection today).
-- BLOCK windows are committed to mart.episode_windows by gate_check itself (p_value/p_adj/alarm NULL, m_t unchanged)
-- so that the window order of the state store is preserved and window t+1 can proceed.
INSERT INTO mart.schema_version (version, note) VALUES (6, 'S4a-00/01: mart.gate_checks (server-stored gate results keyed by contract/gate/calibration)')
ON CONFLICT (version) DO NOTHING;

CREATE TABLE IF NOT EXISTS mart.gate_checks (
  episode_id        uuid     NOT NULL,
  window_index      smallint NOT NULL,
  contract_sha256   text     NOT NULL,
  gate_version      text     NOT NULL,            -- core_pipeline.governance.gate_v2.GATE_VERSION
  calib_sha256      text     NOT NULL,            -- sha256 of models/gate_v2/gate_calibration.json as loaded
  gate              text     NOT NULL CHECK (gate IN ('PASS','FLAG','BLOCK')),
  drift             text     NOT NULL CHECK (drift IN ('ok','critical')),
  drift_assessed    boolean  NOT NULL DEFAULT false,
  drift_stat        double precision,
  drift_threshold   double precision,
  composite_min     double precision,
  composite_mean    double precision,
  block_reasons     jsonb    NOT NULL DEFAULT '[]',
  flag_reasons      jsonb    NOT NULL DEFAULT '[]',
  n_history         integer  NOT NULL DEFAULT 0,
  history_age_windows integer,
  request_id        uuid     NOT NULL,            -- the gate_check request that stored it
  checked_at        timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (episode_id, window_index, contract_sha256, gate_version, calib_sha256),
  FOREIGN KEY (episode_id, window_index) REFERENCES mart.windows(episode_id, window_index)
);
