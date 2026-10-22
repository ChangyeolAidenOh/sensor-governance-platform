"""Unit tests for the demo view-models and configuration guards (S4a-06). No DB, no Streamlit, no LLM."""
from __future__ import annotations

import json

import pytest

from demo import views as V
from demo.runtime import DemoConfigError, check_dsns, fingerprint, load_config

SECRET = "s3cr3t-session-token"
TOKEN = "0b1a2c3d-4e5f-4a6b-8c7d-9e0f1a2b3c4d"
MANIFEST_ENTRY = {"preset": 1, "episode_id": "1602739f-9819-486b-b557-e0d86cb3208c", "n_windows": 48, "n_block": 0,
                  "alarm_windows": list(range(8, 48)), "versions": {"detector": "v1"}, "kind": "alarm", "display_window": 8,
                  "alarm_id": "1602739f-9819-486b-b557-e0d86cb3208c:8:detector_v1", "pred_set": [1], "top1": 1,
                  "note": "창 8에서 첫 prefix-BH 알람, 전체 알람 창 39개", "retrieve_k5": {"query": "x"}, "decide_preview": {"action": "hold"}}
INTERRUPT = {"question": "작업지시 초안을 승인하시겠습니까? (approve / modify / reject)", "proposal_id": "p-1", "draft_version": 2,
             "content_sha256": "a" * 64, "turn_id": "t-1", "alarm_id": "ep:8:detector_v1",
             "form": {"alarm_id": "ep:8:detector_v1", "episode_id": "ep", "window_index": 8, "policy": "E", "decision_id": "d-1",
                      "signals_sha256": "b" * 64, "p_adj": 0.01, "pred_set": [1], "evidence_keys": ["P-116-2012:0:5.1"],
                      "action_items": [{"action_type": "inspect", "item": "루프 점검", "clause_key": "P-116-2012:0:5.1"},
                                       {"action_type": "monitor", "item": "감시 강화", "clause_key": "P-116-2012:0:5.2"}]}}
RESULT_DONE = {"status": "done", "turn_id": "t-1", "category": "action", "answer": "answer text",
               "decision": {"action": "propose", "rule": 4, "reason": "r", "query_kind": None, "decision_id": "d-1", "tau": 0.5,
                            "evidence_snapshot": [{"clause_key": "P-116-2012:0:5.1", "grade": "A", "score": 0.9}], "signals": {"g_AB": 0.9}},
               "query": {"status": "spent", "kind": "rerank", "owned": True, "query_id": "q", "expand_status": None, "rerank_status": "done", "k10_reused": False},
               "draft": {"status": "OK", "payload": {"proposal_id": "p-1", "draft_version": 1, "content_sha256": "a" * 64, "form": INTERRUPT["form"]}, "reason": None},
               "approval": {"status": "OK", "payload": {"approval_token": TOKEN, "decision": "approve"}, "reason": None, "state": {"secret": SECRET}},
               "work_order": {"status": "OK", "payload": {"work_order_id": "wo-1", "cmms_ref": "CMMS-1", "idempotent_replay": False}},
               "trace": [{"tool": "work_order_issue", "step": "issue:v1", "status": "OK", "reason": None, "replayed": False, "raw": {"token": TOKEN}}],
               "turn_refs_discarded": 0, "secret": SECRET}


def test_visible_preset_allow_list():
    v = V.visible_preset(MANIFEST_ENTRY)
    assert v == {"preset": 1, "display_window": 8, "kind": "alarm"}
    for k in V.HIDDEN_PRESET_FIELDS:
        assert k not in v
    assert V.visible_presets({"presets": [MANIFEST_ENTRY, {**MANIFEST_ENTRY, "preset": 0}]})[0]["preset"] == 0
    assert "창 8" in V.preset_label(v)


def test_result_view_drops_token_secret_and_payloads():
    rv = V.result_view(RESULT_DONE)
    assert not V.contains_secret_like(rv, (TOKEN, SECRET))
    assert rv["approval"] == {"status": "OK", "reason": {"code": None, "message": None}, "decision": "approve"}
    assert rv["work_order"]["work_order_id"] == "wo-1" and rv["decision"]["n_snapshot"] == 1
    assert rv["draft"]["proposal_id"] == "p-1" and "form" not in rv["draft"]
    assert rv["trace"] == [{"tool": "work_order_issue", "step": "issue:v1", "status": "OK", "reason": None, "replayed": False}]
    assert "interrupt" not in rv and rv["decision"]["evidence_snapshot"] == [{"clause_key": "P-116-2012:0:5.1", "grade": "A", "score": 0.9}]
    assert "signals" not in rv["decision"]


def test_waiting_result_carries_allow_listed_interrupt_only():
    rv = V.result_view({"status": "waiting_approval", "turn_id": "t-1", "interrupt": {**INTERRUPT, "approval_token": TOKEN}})
    it = rv["interrupt"]
    assert it["proposal_id"] == "p-1" and it["draft_version"] == 2 and it["turn_id"] == "t-1"
    assert set(it["form"]) == set(V.FORM_FIELDS) | {"action_items"}
    assert "signals_sha256" not in it["form"] and "episode_id" not in it["form"]
    assert not V.contains_secret_like(rv, (TOKEN,))


def test_modifications_keep_type_and_clause_change_item_only():
    items = INTERRUPT["form"]["action_items"]
    mods = V.modifications_from_items(items, ["새 문안 1", "  새 문안 2 "])
    assert mods == {"action_items": [{"action_type": "inspect", "item": "새 문안 1", "clause_key": "P-116-2012:0:5.1"},
                                     {"action_type": "monitor", "item": "새 문안 2", "clause_key": "P-116-2012:0:5.2"}]}
    with pytest.raises(ValueError):
        V.modifications_from_items(items, ["only one"])
    with pytest.raises(ValueError):
        V.modifications_from_items(items, ["ok", "   "])


def test_inflight_records_and_pending_rebuild():
    m = V.inflight_message("id-1", "지금 어떤 상황이야?")
    r = V.inflight_resume("id-2", "t-1", "modify", {"action_items": []})
    assert m["kind"] == "message" and r["kind"] == "resume" and r["parent_turn_id"] == "t-1"
    with pytest.raises(ValueError):
        V.inflight_resume("id-3", "t-1", "override", None)
    p_msg = {"turn_id": "id-1", "kind": "message", "status": "waiting_approval", "lease_valid": False, "message": {"message": "작업지시 내줘"}}
    assert V.inflight_from_pending(p_msg) == V.inflight_message("id-1", "작업지시 내줘")
    p_res = {"turn_id": "id-2", "kind": "resume", "status": "registered", "lease_valid": False,
             "message": json.dumps({"parent_turn_id": "t-1", "decision": "approve", "modifications": None})}
    assert V.inflight_from_pending(p_res) == V.inflight_resume("id-2", "t-1", "approve", None)
    assert V.inflight_from_pending(None) is None


APP = "postgresql://sgp_agent:pw@localhost:5999/main_v2"
ADMIN = "postgresql://postgres:pw@localhost:5999/main_v2"


def test_dsn_guards():
    facts = check_dsns(APP, ADMIN, synthetic=False)
    assert facts["app_user"] == "sgp_agent" and facts["skill_user"] == "postgres" and facts["dbname"] == "main_v2"
    with pytest.raises(DemoConfigError):
        check_dsns(APP, "postgresql://postgres:pw@localhost:5999/main_test", synthetic=False)       # different DB
    with pytest.raises(DemoConfigError):
        check_dsns(ADMIN, ADMIN, synthetic=False)                                                   # same user / not the role
    with pytest.raises(DemoConfigError):
        check_dsns("postgresql://other:pw@localhost:5999/main_v2", ADMIN, synthetic=False)         # wrong role
    with pytest.raises(DemoConfigError):
        check_dsns(APP, ADMIN, synthetic=True)                                                      # synthetic needs test DBs
    t = check_dsns("postgresql://sgp_agent:pw@localhost:5999/main_test", "postgresql://postgres:pw@localhost:5999/main_test", synthetic=True)
    assert t["dbname"] == "main_test"


def test_load_config_and_fingerprint(tmp_path):
    man = tmp_path / "m.json"
    man.write_text(json.dumps({"schema": "demo_presets/1", "presets": [MANIFEST_ENTRY]}), encoding="utf-8")
    env = {"SGP_AGENT_PG_DSN": APP, "SGP_PG_DSN": ADMIN, "SGP_DEMO_MANIFEST": str(man)}
    cfg = load_config(env)
    assert cfg.synthetic is False and cfg.dsn_facts["app_user"] == "sgp_agent"
    with pytest.raises(DemoConfigError):
        load_config({"SGP_PG_DSN": ADMIN})
    with pytest.raises(DemoConfigError):
        load_config({**env, "SGP_DEMO_SYNTHETIC": "1"})
    with pytest.raises(DemoConfigError):
        load_config({**env, "SGP_DEMO_MANIFEST": str(tmp_path / "missing.json")})
    f1 = fingerprint(cfg, env)
    assert f1 == fingerprint(cfg, env) and f1 != fingerprint(cfg, {**env, "SGP_LLM_NARRATOR": "llm"})
    man.write_text(json.dumps({"schema": "demo_presets/1", "presets": []}), encoding="utf-8")
    assert f1 != fingerprint(cfg, env)                                                          # manifest CONTENT is part of the key
    # the error messages never echo a DSN
    try:
        check_dsns("not a dsn ::", ADMIN, synthetic=False)
    except DemoConfigError as e:
        assert "pw" not in str(e)


ANSWER = ("창 8: gate=PASS, drift=critical, p=0.0002083, alarm=True\n진단 집합(후보): [1] (크기 1, top1=1)\n정책 결정(E): 보류 — 규칙 3 (drift critical)\n"
          "검색 mode=scoped, τ=0.5, g_AB=0.03525, ev_AB=False\n[P-132-2013:0:4.4] (A, score 0.035) 연속식 반응기\n"
          "> 연속식 반응기에 원료 및 용매와 촉매의 공급량\n두 번째 줄\n[P-132-2013:0:4.1] (B, score 0.033) 일반 사항\n"
          "> (마) 폭발성 물질이\n(바) 냉각수가 정지하여\n초안 eb60 v2 생성(sha 83f6…).\n승인 결과: approve\n작업지시 발행: b26a (cmms_ref MOCK)\n요약(LLM 생성): 없음")


def test_answer_blocks_split_is_lossless_and_labels_clauses():
    blocks = V.answer_blocks(ANSWER)
    assert [b["type"] for b in blocks] == ["text", "clause", "clause", "text"]
    assert blocks[1]["title"].startswith("[P-132-2013:0:4.4]") and blocks[1]["lines"] == ["연속식 반응기에 원료 및 용매와 촉매의 공급량", "두 번째 줄"]
    assert blocks[2]["title"].startswith("[P-132-2013:0:4.1]") and blocks[2]["lines"] == ["(마) 폭발성 물질이", "(바) 냉각수가 정지하여"]
    assert blocks[3]["lines"] == ["초안 eb60 v2 생성(sha 83f6…).", "승인 결과: approve", "작업지시 발행: b26a (cmms_ref MOCK)", "요약(LLM 생성): 없음"]
    assert V.answer_from_blocks(blocks) == ANSWER                       # display split loses nothing
    assert V.answer_from_blocks(V.answer_blocks("")) == "" and V.answer_from_blocks(V.answer_blocks("plain\ntext")) == "plain\ntext"
    hdr_only = "[P-1-2013:0:1] (A, score 0.5) t\n초안 x"
    assert V.answer_from_blocks(V.answer_blocks(hdr_only)) == hdr_only and V.answer_blocks(hdr_only)[0]["lines"] == []


def test_turn_sections_judgment_first_clauses_with_criterion():
    view = V.result_view({**RESULT_DONE, "answer": ANSWER, "decision": {**RESULT_DONE["decision"], "action": "hold", "rule": 3, "reason": "drift critical: model_trust_low",
                          "evidence_snapshot": [{"clause_key": "P-132-2013:0:4.4", "grade": "A", "score": 0.035}, {"clause_key": "P-132-2013:0:4.1", "grade": "B", "score": 0.033},
                                                {"clause_key": "P-116-2012:0:5.1", "grade": "A", "score": 0.9}, {"clause_key": "P-132-2013:0:4.1", "grade": "B", "score": 0.9}]}})
    sec = V.turn_sections(view)
    assert sec["judgment"]["label"] == "보류" and sec["judgment"]["rule"] == 3 and sec["judgment"]["reason"] == "drift critical: model_trust_low"
    assert sec["state_line"] == "창 8: gate=PASS, drift=critical, p=0.0002083, alarm=True"
    assert [c["clause_key"] for c in sec["clauses"]] == ["P-132-2013:0:4.4", "P-132-2013:0:4.1", "P-116-2012:0:5.1"]   # duplicates grouped
    assert [c["meets"] for c in sec["clauses"]] == [False, True, True] and sec["clauses"][1]["n_items"] == 2
    assert "충족 2/3 조항" in sec["criterion"] and "스냅샷 4항목, 청크 중복 1" in sec["criterion"]
    assert sec["clauses"][0]["lines"] == ["연속식 반응기에 원료 및 용매와 촉매의 공급량", "두 번째 줄"] and sec["clauses"][2]["lines"] == []
    assert "요약(LLM 생성): 없음" in sec["other_lines"] and "승인 결과: approve" in sec["other_lines"] and sec["state_line"] not in sec["other_lines"]
    assert not any(l.startswith("(바)") for l in sec["other_lines"])                                             # body continuation stays in the clause


def test_turn_sections_without_decision_uses_answer_blocks_only():
    sec = V.turn_sections(V.result_view({"status": "done", "turn_id": "t", "category": "evidence", "answer": ANSWER}))
    assert sec["judgment"] is None and sec["criterion"] is None
    assert [c["clause_key"] for c in sec["clauses"]] == ["P-132-2013:0:4.4", "P-132-2013:0:4.1"] and all(c["meets"] is None for c in sec["clauses"])
    assert V.turn_sections(V.result_view({"status": "done", "turn_id": "t", "answer": None}))["clauses"] == []


def test_clause_label_headline_and_main_lines():
    c = {"clause_key": "P-132-2013:0:4.4", "grade": "A", "score": 0.9, "narrator_score": 0.035, "meets": True, "n_items": 2, "title": "[P-132-2013:0:4.4] (A, score 0.035) 연속식 반응기", "lines": []}
    assert V.clause_label(c, synthetic=True) == "연속식 반응기 · P-132 §4.4 · A등급 · 합성 주입 0.900 / 검색 0.035 · 기준 충족 · 청크 2개"
    assert V.clause_label(c, synthetic=False) == "연속식 반응기 · P-132 §4.4 · A등급 · 최대 청크 점수 0.900 · 기준 충족 · 청크 2개"
    real = {**c, "score": 0.035, "n_items": 1}
    assert V.clause_label(real, synthetic=False) == "연속식 반응기 · P-132 §4.4 · A등급 · score 0.035 · 기준 충족"
    cgrade = {"clause_key": "P-129-2013:0:5.2.2", "grade": "C", "score": 0.008227, "narrator_score": 0.008, "meets": False, "n_items": 2, "title": "[P-129-2013:0:5.2.2] (C, score 0.008) 점검방법", "lines": []}
    assert "합성" not in V.clause_label(cgrade, synthetic=True)                                   # C grade is never injected
    assert V.clause_label({"clause_key": None, "title": "weird", "lines": []}) == "weird"
    view = V.result_view(RESULT_DONE)                                   # approval OK + work_order OK
    h = V.headline(view)
    assert h["kind"] == "issued" and "모의(mock CMMS) 작업지시 발행 완료" in h["title"] and "v1" in h["title"] and "wo-1" in h["title"]
    rej = V.result_view({**RESULT_DONE, "work_order": None, "approval": {"status": "OK", "payload": {"approval_token": None, "decision": "reject"}}})
    assert V.headline(rej)["kind"] == "rejected" and "작업지시 없음" in V.headline(rej)["title"]
    dec_only = V.result_view({"status": "done", "turn_id": "t", "decision": {"action": "hold", "rule": 3, "reason": "r"}})
    assert V.headline(dec_only)["kind"] == "decision" and "보류" in V.headline(dec_only)["title"]
    assert V.headline(V.result_view({"status": "done", "turn_id": "t"})) is None
    sec = V.turn_sections(V.result_view({"status": "done", "turn_id": "t", "answer": ANSWER + "\n이상한 줄"}))
    assert "이상한 줄" in sec["hidden_lines"] and "이상한 줄" not in sec["main_lines"] and "승인 결과: approve" in sec["main_lines"]


DUP_ANSWER = ("창 8: gate=PASS\n[P-132-2013:0:4.1] (B, score 0.033) 일반 사항\n> 첫 청크 본문\n[P-129-2013:0:5.2.2] (C, score 0.008) 점검방법\n> C 본문\n"
              "[P-132-2013:0:4.1] (B, score 0.031) 일반 사항\n> 둘째 청크 본문\n둘째 청크 계속")


def test_duplicate_chunks_keep_every_body_with_their_own_scores():
    view = V.result_view({"status": "done", "turn_id": "t", "answer": DUP_ANSWER,
                          "decision": {"action": "hold", "rule": 5, "reason": "r", "tau": 0.5,
                                       "evidence_snapshot": [{"clause_key": "P-132-2013:0:4.1", "grade": "B", "score": 0.033},
                                                             {"clause_key": "P-129-2013:0:5.2.2", "grade": "C", "score": 0.008},
                                                             {"clause_key": "P-132-2013:0:4.1", "grade": "B", "score": 0.9}]}})
    sec = V.turn_sections(view)
    g = sec["clauses"][0]
    assert g["clause_key"] == "P-132-2013:0:4.1" and g["n_items"] == 2 and g["score"] == 0.9 and g["meets"] is True
    assert [ch["lines"] for ch in g["chunks"]] == [["첫 청크 본문"], ["둘째 청크 본문", "둘째 청크 계속"]]   # both bodies kept, in order
    assert [ch["score"] for ch in g["chunks"]] == [0.033, 0.9] and [ch["narrator_score"] for ch in g["chunks"]] == [0.033, 0.031]
    assert [ch["meets"] for ch in g["chunks"]] == [False, True]                                   # the high score belongs to chunk 2 only
    assert sec["clauses"][1]["n_items"] == 1 and "청크 중복 1" in sec["criterion"] and "최대 청크 score" in sec["criterion"]


def test_template_lines_from_narrator_source():
    for l in ("승인 거부: 승인 불일치", "발행 거부: TOKEN_USED", "open 모드 결과는 참고용이며 제안의 근거(ev_AB)가 될 수 없습니다.", "이 창에서는 prefix-BH 알람이 없습니다.",
              "선택 가능한 알람이 없어 조치안 경로로 진입하지 않습니다.", "요청한 참조 창 3는 현재 상태와 달라 무시하고 선택 창 기준으로 답합니다.", "(조항 없음)",
              "query 상태: spent (expand=None, rerank=done)", "[1]은(는) 집합에 포함됩니다.", "[4]은(는) 집합에 없으므로 그 주장을 근거로 제안하지 않습니다."):
        assert V.is_template_line(l), l
    assert not V.is_template_line("촉매 유량계 등 중요한 측정기기는 정기적으로 테스트하여")
    body_then_open = "[P-1-2013:0:1] (A, score 0.5) t\n> first\ncontinuation 승인 아님\nopen 모드 결과는 참고용입니다."
    b = V.answer_blocks(body_then_open)
    assert b[0]["lines"] == ["first", "continuation 승인 아님"] and b[1]["lines"] == ["open 모드 결과는 참고용입니다."]
    assert V.answer_from_blocks(b) == body_then_open
