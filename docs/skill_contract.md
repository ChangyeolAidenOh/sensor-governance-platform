# Skill 계약서 (Skill Contract) v1.0 — Stage 4a S4a-00/01

**작성일:** 2026-10-21 · **상태:** v1.0 확정(§2 필드 고정, JSON Schema `core_pipeline/skills/schemas/`가 정본; 서버·클라이언트·골든·H0 테스트가 같은 스키마를 검증) · **동결 범위:** §1 불변 조항·§3 호출 규칙(prereg-v1). v0.1 → v1.0에서 §1.2·§1.8의 **표현**이 바뀌었고(사전등록 §9 **A-4**, ST4-10), 정책 의미는 유지된다 · **소비자:** LangGraph 그래프(1차), AutoGen(Should). 둘 다 같은 FastMCP 서버를 호출한다 · **원칙:** LLM은 작가, 검색·판단·검증은 별도. Skill은 수치를 **반환**하고 그래프의 `decide_action`이 **결정**한다.

## 0. 공통

- **전송:** FastMCP 서버 2개(ST4-01), stdio. 그래프는 `core_pipeline/skills/client.py`의 `SkillClient`로 두 서버를 subprocess로 띄우고 세션 동안 유지한다. 그래프 프로세스는 두 서버 모듈을 import하지 않는다(lightgbm ↔ torch 동시 로드 금지). HTTP 전송·Streamlit Cloud 동작은 **배포 검증 전 예정 사항**(Stage 7).
  - `skills_detect`: `gate_check` · `detect_diagnose` · `work_order_draft` · `work_order_approve` · `work_order_issue` · `work_order_status` (lightgbm, psycopg)
  - `skills_retrieve`: `retrieve` (torch, sentence-transformers)
- **문맥(ctx):** 모든 호출은 `ctx = {request_id(UUID), thread_id(세션 발급, ST4-04), run_uuid(UUID), caller_role ∈ {engineer, restricted}, mode ∈ {chat, event, replay}}`를 받는다. `thread_id = alarm_id`(v0.1)는 폐기. `caller_role`·`decision_id`·`episode_id`·`window_index`는 그래프 상태에서만 온다.
- **봉투(envelope):** 항상 `{status, payload, grounding, state, reason, trace}`.
  - `status ∈ {OK, REFUSED, ERROR}` — **실행 결과만.** `UNDECIDABLE`(그래프 분기 결과)과 `NO_EVIDENCE`(근거 판정)는 status가 아니다.
  - `payload`: 도구별 결과(§2). `grounding`: `retrieve`만(§2.3). `state`: `detect_diagnose`·`gate_check(BLOCK)`가 돌려주는 **커밋된 `mart.episode_windows` 행**(ST4-02).
  - `reason = {code(§5), message}` — `status ≠ OK`일 때만.
  - `trace = {request_id, thread_id, run_uuid, skill, tool, contract_version "1.0", latency_ms, versions{detector, diagnoser, gate, calib_sha256, contract_sha256, alpha, tau | embedder, reranker, tau, fault_map_sha256}, replayed}`.
- **요청 소유권·멱등성(ST4-13):** 호출은 실행 **전에** `mart.agent_steps`에 `PENDING`으로 선점된다(`request_id` 유일). 선점에 성공한 프로세스만 실행하고, 최종 봉투(출력 스키마 검증 후)를 같은 행에 확정한다. 같은 `request_id` + 같은 **결합 입력**(ctx의 thread/run/role/mode + skill + tool + 계약 버전 + 인자) → 확정된 봉투를 `trace.replayed=true`로 재반환; 결합 입력이 다르면 `REFUSED(REQUEST_ID_REUSED)`; 아직 실행 중이면 `REFUSED(REQUEST_IN_PROGRESS)`; 확정 기록에 실패하면 `ERROR(INTERNAL)`(행은 `PENDING`으로 남고 `SGP_REQUEST_STALE_SECONDS` 뒤 같은 요청이 재선점). 부수효과가 있는 도구는 두 번 실행되지 않는다.
- **부분 커밋 완성(ST4-13):** 저장은 여러 트랜잭션에 걸칠 수 있으므로, 다음 호출이 누락을 멱등하게 채운다 — BLOCK gate 행이 있는데 창 행이 없으면 `gate_check`가 창 행을 커밋; 창 행이 알람인데 `alarms/diagnoses`가 없으면 `detect_diagnose`가 결정적 재계산으로 채운다.
- **스키마 검증:** 클라이언트는 입력을, 서버는 자기 출력을 스키마로 검증한다. 서버 출력이 스키마를 어기면 `ERROR(SCHEMA_INVALID)`로 바꿔 돌려주지, 조용히 통과시키지 않는다.
- **시간 경계:** 모든 Skill은 `WindowView`(샘플 1..end_cycle만 물리적으로 보유)로만 데이터를 받는다. 창 t 이후 샘플은 읽을 수 없다.
- **stdout 금지:** 서버 프로세스의 stdout은 MCP 채널이다. 로그는 stderr.

## 1. 불변 조항 (사전등록 동결; A-4로 표현 수정된 항목 표시)

1. **BLOCK이면 탐지 없음.** `gate_check`가 `BLOCK`을 판정한 창은 **gate_check 자신이** `mart.episode_windows`에 `gate=BLOCK, p_value/p_adj/alarm=NULL, m_t 불변`으로 커밋한다(ST4-11). 그 창에 `detect_diagnose`를 호출하면 `REFUSED(GATE_BLOCK)`이고 호출은 `agent_steps.violation=true`로 기록되며 아무것도 쓰지 않는다. 다음 창 t+1은 정상 진행한다. 그래프는 `undecidable`로 분기한다(H0-(d)).
2. **(A-4 표현 수정) 근거 없음은 구조적 보류.** 검색 **실행** 성공은 `status=OK`로 반환한다. 조치안 경로의 **근거 조건**은 `grounding.mode == "scoped" ∧ grounding.ev_AB is True`뿐이며 `False`·`None`은 충족하지 않는다. 미충족이면 정책표(§4.5)에 따라 `query_more`는 가능하지만 조치안 서술·초안 생성은 불가하다 — 프롬프트가 아니라 그래프 분기와 `work_order_draft`의 `REFUSED(NO_EVIDENCE)`로 강제한다. (v0.1: "`retrieve`가 `NO_EVIDENCE`를 반환하면" → 판정이 status에 섞여 있던 표현을 분리. 정책 의미 동일.)
3. **승인 없이 작업지시 없음.** `work_order_issue`는 `approval_token` 없이는 `REFUSED(NO_APPROVAL)`. 토큰은 `work_order_approve(decision=approve)`가 (proposal, version, content_sha256, thread)에 묶어 발급하는 1회용이며 재사용은 `REFUSED(TOKEN_USED)`. 토큰 소비와 발행은 한 트랜잭션(H0-(c)).
4. **도구 인자는 상태에서만 온다.** `episode_id`·`window_index`는 격자(`mart.v_windows_agent`)에 존재해야 하고(`ERROR(WINDOW_OUT_OF_RANGE)`), 창은 순서대로만 관측된다(`REFUSED(WINDOW_ORDER)`). `detect_diagnose`는 호출자가 준 gate dict를 믿지 않고 **서버가 저장한** 같은 `(episode, window, contract_sha256, gate_version, calib_sha256)`의 gate 행을 읽는다(없으면 `REFUSED(GATE_NOT_RUN)`; 옛 보정으로 계산된 PASS는 현재 탐지를 허용하지 않는다). `retrieve(scoped, alarm_id)`의 `fault_set`은 그 알람 창에 저장된 집합(확장 실행 후면 `expanded_set`)과 같아야 한다(`REFUSED(ARG_NOT_IN_STATE)`). (0-F: GPT S09가 창 경계를 지어냈다 — ST0-23 (3).)
5. **사용자 제공 진단은 입력이 아니라 질문이다.** 대화의 진단·결함 단언은 그래프 상태 `user_claims`에만 저장되고 Skill 입력으로 들어가지 않는다. 그래프는 알람이 없으면 알람 선택을 요청하고, 있으면 `gate_check → detect_diagnose`를 먼저 호출한 뒤 주장을 집합과 **대조해 서술**한다. 주장이 집합에 없으면 그 주장을 근거로 제안하지 않는다. (0-F: Gemini — ST0-23 (3).)
6. **매핑은 조치안 경로의 하드 필터, 순위는 검색.** `retrieve(mode=scoped)`는 `fault_to_guidance_map.json`의 범위(예측 집합 원소의 A∪B∪C 합집합, 빈 집합이면 C만)를 ACL과 같은 층의 사전 필터로 쓴다. `mode=open`(ACL만)은 대화의 evidence 범주에만 허용되며 그 결과는 `ev_AB`가 될 수 없다(서버가 `grounding.ev_AB=false`로 고정). 순위는 두 모드 모두 하이브리드 + 리랭커.
7. **수치 판단은 Skill 밖 규칙으로.** Skill은 p-value·집합·점수를 돌려주고 "낮다/높다"를 말하지 않는다. `Signals`는 `core_pipeline/policy/signals_adapter.py`가 검증된 봉투에서만 만든다(ST4-07).
8. **(A-4 표현 수정) 제안은 A·B 근거를 요구한다.** C 조항은 참고 근거이며 단독으로 조치안을 뒷받침하지 못한다. `work_order_draft`는 명시적 `decision_id`의 결정이 (alarm, policy)의 **최신** `propose`이고 그 결정의 **스냅샷**(`policy_decisions.evidence`)에 `τ` 이상인 A/B 조항이 있을 때만 초안을 만든다. 아니면 `REFUSED(NO_EVIDENCE)`. 인용은 스냅샷의 A/B 키만(`ev_AB == true` 표기는 `ev_AB is True`로 읽는다).
9. **인과성.** 모든 Skill은 창 t 종료 시점까지의 샘플과 캘리브레이션 그룹의 외부 기준만 읽는다. 알람은 커밋된 prefix 행에서 재계산한 prefix-BH(BLOCK 창은 검정 목록 제외, 이전 판정 불변, FDR 통제 주장 없음). 확장 창은 과거 방향(t−1 ∪ t).

## 2. 도구별 계약 (정본: `core_pipeline/skills/schemas/*.schema.json`)

### 2.1 `gate_check(ctx, episode_id, window_index)` — skills_detect
| | 내용 |
|---|---|
| payload | `gate ∈ {PASS,FLAG,BLOCK}`, `drift ∈ {ok,critical}`, `drift_assessed`, `drift_stat`, `drift_threshold`, `composite_min/mean`, `block_reasons[]`, `flag_reasons[]`, `n_history`, `history_age_windows`, `gate_version`, `calib_sha256`, `contract_sha256`, `stored`(저장본 재반환 여부) |
| state | BLOCK이면 커밋된 창 행; 아니면 null(탐지가 쓴다). 이미 관측된 창이면 저장 행을 `reused=true`로 |
| status | `OK` / `REFUSED(WINDOW_ORDER)` / `ERROR(WINDOW_OUT_OF_RANGE, DATA_MISSING)` |
| 저장 | `mart.gate_checks` (키 `(episode, window, contract_sha256, gate_version, calib_sha256)`) |
| 비보장 | 게이트 판정의 적절성(ST0-11 (3)) |

### 2.2 `detect_diagnose(ctx, episode_id, window_index, expand=1)` — skills_detect
| | 내용 |
|---|---|
| 전제 | 같은 키의 저장 gate 행(§1.4). `expand=2`는 그 창의 1차 탐지 행이 있어야 하고(`DETECT_NOT_RUN`), t=0 불가(`NO_EXPANSION`), **`spend_query`가 돌려준 `query_id`를 제시**해야 한다(`QUERY_REQUIRED`; 다른 query의 id면 `QUERY_USED`). 같은 `query_id`로 재호출하면 저장된 확장 결과를 `reused=true`로 돌려준다 |
| payload | `detection_run`, `score`, `p_value`, `p_adj`(prefix-BH), `alarm`, `m_t`, `alpha=0.10`, `diagnosis{set, set_size, valid(1≤s≤3), coverage_target 0.90, calibration_window 20|40, top1, top1_prob, indeterminate}`, `contributions`(상위 5), `calibration_ref{n_cal, detector_version, diagnoser_version}`, `alarm_id`(알람이면 `<episode>:<window>:<detector>`) |
| state | 커밋된 `episode_windows` 행(`pred_set`, `expanded_set`([]=실행·빈 집합, null=미실행), `q_used`, `reused`, `alarm_id`) |
| 부수효과 | 알람이면 `mart.alarms`·`mart.diagnoses` 행(알람만). 확장은 `expanded_set`만 쓴다(A-3: 탐지 열 불변) |
| status | `OK` / `REFUSED(GATE_BLOCK, GATE_NOT_RUN, WINDOW_ORDER, DETECT_NOT_RUN, NO_EXPANSION, EXPANSION_INVALID, QUERY_USED)` / `ERROR(WINDOW_OUT_OF_RANGE, MODEL_UNAVAILABLE)` |
| 보장 | 같은 입력·같은 모델 버전 → 같은 출력. 이미 관측된 창의 재호출은 저장 행을 돌려준다(대화 모드). 두 인스턴스 동시 관측 → 커밋된 결과 하나 |
| 비보장 | 조건부·결함별 커버리지. `diagnosis.set`은 후보 |

### 2.2b `spend_query(ctx, episode_id, window_index)` — skills_detect (ST4-13)
창당 **하나의 논리적 query**(§3.5 `q_used`)를 확보한다. 1차 탐지 행이 있어야 하고(`DETECT_NOT_RUN`), BLOCK 불가, 이미 다른 요청이 썼으면 `REFUSED(QUERY_USED)`. payload `{query_id, q_used: true, expand_allowed(t>0)}` + `state`. `query_id`는 이 요청의 `request_id`이며 `episode_windows.query_request_id`에 저장된다. 그래프는 `query_more` 결정 시 한 번 호출하고, 확장(`detect_diagnose expand=2`)과 재검색(`retrieve k=10`)에 같은 `query_id`를 넘긴다. 같은 요청의 재시도는 같은 `query_id`를 돌려준다.

### 2.3 `retrieve(ctx, query, mode, fault_set, k ∈ {5,10}, alarm_id?, query_id?)` — skills_retrieve
| | 내용 |
|---|---|
| 처리 | ACL(`caller_role`) → (scoped) 매핑 하드 필터 → BM25(문자 2-gram) + bge-m3 각 20 → RRF(k=60) → bge-reranker-v2-m3(sigmoid 1회, [0,1]) → top-k |
| payload | `clauses[{chunk_key, clause_key, guide_no, title, text, page, grade ∈ {A,B,C,OUT}, rank, score, rrf}]`, `k`, `mode`, `scope_size`, `n_candidates`, `alarm_id`, `reused` |
| 상태 조회 | `alarm_id`가 있으면 알람 창의 저장 행을 **같은 detector·같은 `contract_sha256`**(detect 서버와 같은 `outputs/model_meta/detector_v1.json`)로 읽는다. scoped `fault_set`은 그 행의 집합(확장 실행 후면 `expanded_set`)과 같아야 함 |
| 예산 | scoped `k=10`은 `query_more` 재검색: 알람 창의 `query_request_id`와 같은 `query_id` 필요(`QUERY_REQUIRED`/`QUERY_USED`). scoped `(alarm, k)`는 **한 번만 계산**되고 이후 호출은 저장 봉투를 `reused=true`로 돌려준다(evidence 행 중복 없음, 예산 밖 반복 재순위 없음) |
| **grounding** | `{mode, tau, g_AB, g_C, ev_AB, ev_C, snapshot[{clause_key, grade, score}]}` — `snapshot`은 그래프 decide 노드가 `record_decision(evidence=…)`에 그대로 넣는 값. open이면 `ev_AB=false` 고정 |
| 부수효과 | `alarm_id`가 있으면 `mart.evidence` 행(mode·k·rank·score·tau) |
| status | `OK` / `REFUSED(ACL_DENIED, NO_ALARM, ARG_NOT_IN_STATE, QUERY_REQUIRED, QUERY_USED)` / `ERROR(INDEX_MISMATCH)` — **`NO_EVIDENCE`는 status가 아니다(A-4)** |
| 보장 | ACL 밖 조항은 payload에 들어가지 않는다(필터가 검색 앞). 반환 조항은 색인에 실재하는 `clause_key`. 서버 기동 시 `tau.json`의 `fault_map_sha256`·임베더 이름을 인덱스와 대조해 불일치면 기동 거부 |
| 금지 | LLM이 조항 텍스트를 생성·보강하지 않는다. 서술 노드는 `clauses[].text`를 그대로 인용 |

### 2.4 `work_order_*` — skills_detect (mock CMMS, ST4-03)
| 도구 | 입력 | 출력 payload | REFUSED 코드 |
|---|---|---|---|
| `draft` | `alarm_id, episode_id, window_index, policy ∈ {E,B3}, decision_id, action_items[{action_type(화이트리스트 7종), item, clause_key}], base_proposal_id?` | `{proposal_id, draft_version, content_sha256, form}`(불변 버전 행) | `NO_ALARM, EVENT_MISMATCH, DECISION_MISMATCH, NO_PROPOSE_DECISION, STALE_DECISION, NO_EVIDENCE, FORBIDDEN_ACTION, THREAD_MISMATCH, BAD_REQUEST` |
| `approve` | `draft(제시 본문 전체), approver, decision ∈ {approve,modify,reject,override}, modifications?` | `{approval_token|null, decision}` | `APPROVER_MISMATCH`(`approver` ≠ `mart.threads.owner[ctx.thread_id]` — 세션 시작 시 등록된 소유자만; 도구 호출이 승인자를 만들어낼 수 없음), `APPROVAL_MISMATCH`(제시 본문 ≠ 저장본) |
| `issue` | `draft, approval_token|null` | `{work_order_id, cmms_ref, idempotent_replay}` | `NO_APPROVAL, TOKEN_USED, SUPERSEDED, CONTENT_MISMATCH, THREAD_MISMATCH, APPROVAL_MISMATCH, UNKNOWN_PROPOSAL` |
| `status` | `proposal_id` | 현재 버전·승인 목록·발행 목록 | `UNKNOWN_PROPOSAL, THREAD_MISMATCH` |

B1/B2는 초안을 만들지 않는다(제안 사건은 `policy_decisions`가 기록). `cancel`은 v1.0 범위 밖(데모에 불필요; 필요 시 S4a-06에서 결정). 경합 발행은 unique 충돌 후 기존 작업지시 반환(멱등).
**사람 승인 경계(S4a-04 범위, 여기서는 계약만):** `work_order_approve`·`work_order_issue`는 LLM 도구 목록에 노출하지 않는다 — 그래프의 interrupt 재개 노드만 호출하며, `approver`는 UI 승인 이벤트의 확인된 소유자(= `mart.threads.owner`)다. 서버는 소유자 일치를 강제하고(`APPROVER_MISMATCH`), "사람 승인 없이 토큰 발급 불가"의 나머지 절반(도구 비노출·UI 이벤트 연결)은 S4a-04 완료 기준에 넣는다.

## 3. 그래프 측 호출 규칙 (동결, v0.1 rev 2와 동일; 3.4 보강)
3.1 action·evidence 범주는 `gate_check → detect_diagnose → retrieve` 순서를 건너뛸 수 없다. · 3.2 사용자 제공 진단·수치는 `user_claims`에 별도 저장, Skill 입력 금지, 서술 시 대조 형식. · 3.3 알람 미선택 스레드의 action 질의는 `work_order_draft` 대신 알람 선택 요청. · 3.4 도구 인자 생성 금지: LLM 도구 호출의 `episode_id`·`window_index`·`proposal_id`·`fault_set`이 상태와 다르면 라우터가 폐기하고 상태 값으로 재호출(서버도 §1.4로 거부). 폐기 건수 [보고]. · 3.5 `query_more`는 창당 1회(`q_used`): ¬valid_s → `expand=2`, ¬ev_AB → `k=10` 재검색, 둘 다면 둘 다; 재진입은 규칙 3부터. · 3.6 out_of_scope는 도구 미호출 + 거절. · 3.7 history 범주는 `mart` SQL 템플릿. · 3.8 methodology는 고정 문서. · 3.9 데모 보호 `REFUSED(DEMO_LIMIT)`(그래프). · 3.10 S08v2·S09v2는 Stage 4 개발 검증. · 3.11 evidence 범주만 `open` 허용.

## 4. 골든 (ST4-12: 생성 1회 → 동결 → 회귀는 고정값 비교)
생성 `scripts/make_goldens.py`(테스트 DB 전용, 창 0부터 순차 리플레이, 선택 규칙 1회 적용, 사양 단언 통과 후 `tests/regression/golden/<case>.json`에 창·버전·기대값 동결). 회귀 `tests/regression/test_golden.py`는 동결 파일과 비교만 한다 — 재탐색·자동 xfail 없음. 버전 불일치는 "golden stale" 실패.

| # | 사례 | 선택 규칙·사양 단언 |
|---|---|---|
| G1 | 프리셋 2(104026, 결함 4) | 8..47의 첫 prefix-BH 알람 창. 있으면 결과 기록(계약 기대 "4 ∈ set, ev_AB true"는 관찰로 기록); 없으면 **no_alarm 사례로 보존** + `work_order_draft → REFUSED(NO_ALARM)`. 회귀는 block 사례를 제외한 모든 사례에서 **창 0..47 전체**를 재생·비교한다(무알람 동결 뒤 새 알람이 생기면 실패) |
| G1b | 프리셋 1(101022, 결함 1) | 알람 창 **존재(hard)**, 1 ∈ set, scoped `ev_AB is True`, 상위 조항에 P-132/P-167/P-179 |
| G2 | 프리셋 9(100250, 정상) | 8..47의 첫 비알람 창: `alarm=false` → `decide_action=no_alarm` → draft `REFUSED(NO_ALARM)`. 전체 알람 벡터는 dev 관찰 |
| G3 | 프리셋 10(100251) + 창 2 BLOCK 주입(`SGP_TEST_BLOCK_WINDOWS`, 테스트 DSN에서만 동작) | 창 2: BLOCK 행 커밋(p/alarm null, m_t 불변) · `detect_diagnose(2) → REFUSED(GATE_BLOCK)` · 창 3 진행, m₃ = m₁+1 |
| G4 | 프리셋 8(118178, 결함 18) | 실제 결과 기록(set이 {16..20} 안이면 C-only → ev_AB false → propose 결정이 있어도 draft `REFUSED(NO_EVIDENCE)`). **계약 검증은 별도:** `tests/unit/test_scope_contract.py` — `{18}` → C만, 빈 집합 → C만, 혼합 집합은 A/B 보유, `valid_s`와 `ev_AB`는 독립 |
| G5 | G1b 알람의 **실제** 정책 결정이 `propose`일 때만 | `issue(token=null) → REFUSED(NO_APPROVAL)`, approve 후 issue `OK`. 실제 결정이 hold/query_more면 골든은 `REFUSED(NO_PROPOSE_DECISION)`을 동결하고, 승인 기제는 합성 propose 계약 테스트(`test_h0_servers.py::test_h0_c_…`)가 맡는다. 골든은 `decide_action` 결과를 그대로 기록하며 `propose`를 강제하지 않는다(ST4-13) |

## 5. 오류·거절 코드 (정본: `envelope.schema.json#/$defs/reason`)
`GATE_BLOCK · GATE_NOT_RUN · DETECT_NOT_RUN · WINDOW_ORDER · WINDOW_OUT_OF_RANGE · DATA_MISSING · NO_EXPANSION · EXPANSION_INVALID · QUERY_USED · ARG_NOT_IN_STATE · ACL_DENIED · NO_ALARM · EVENT_MISMATCH · DECISION_MISMATCH · NO_PROPOSE_DECISION · STALE_DECISION · NO_EVIDENCE · FORBIDDEN_ACTION · THREAD_MISMATCH · SUPERSEDED · CONTENT_MISMATCH · APPROVAL_MISMATCH · TOKEN_USED · NO_APPROVAL · UNKNOWN_PROPOSAL · MODEL_UNAVAILABLE · INDEX_MISMATCH · CMMS_DOWN · DEMO_LIMIT · BAD_REQUEST · SCHEMA_INVALID · REQUEST_ID_REUSED · INTERNAL`

## 6. 버전·변경
- **v1.0 rev 2 (2026-10-21, ST4-13):** 요청 선점(PENDING)·결합 입력 멱등·확정 전 출력 검증, 부분 커밋 완성, `spend_query`/`query_id`(확장·k=10 공통 예산), scoped (alarm,k) 1회 계산·재사용, retrieve 상태 조회에 contract 키, `signals_adapter` 사건·계약 대조, `APPROVER_MISMATCH`(스레드 소유자 바인딩), 골든의 실제 정책 결정 보존·전체 창 회귀.
- **v1.0 (2026-10-21, ST4-10~12):** status 3값·grounding 분리(A-4), 봉투·trace 고정, `state`=커밋 행, 서버 저장 gate 조회 키(ST4-11), BLOCK 행 커밋 주체 = gate_check, `request_id` 멱등, `ARG_NOT_IN_STATE`를 scoped fault_set에 적용, thread_id 세션별(ST4-04), work_order 4도구(cancel 제외), 골든 2단계(ST4-12), JSON Schema 정본.
- v0.1 rev 3(2026-10-19, prereg-v1): BLOCK 창 prefix 제외·m_t. v0.1 rev 2: §1·§3 동결.
- §1·§3 변경은 "사전등록 수정(동결 후·수정 등록)", §2 변경은 "구현 세부 결정".
