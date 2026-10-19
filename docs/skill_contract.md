# Skill 계약서 (Skill Contract) v0.1 — Stage 0-H 초안

**작성일:** 2026-10-19 (rev 2) · **상태:** 초안(사전등록 동결 범위: §1 불변 조항, §3 호출 규칙). 입출력 필드는 Stage 4a 착수 전 v1.0으로 확정하며 변경은 결정 로그 "구현 세부 결정"으로 기록 · **소비자:** LangGraph 그래프(1차), AutoGen(Should). 두 소비자는 같은 MCP 서버(FastMCP)를 호출한다 · **원칙:** LLM은 작가, 검색·판단·검증은 별도(계획서 §5.3). Skill은 수치를 **반환**하고 그래프 규칙이 **결정**한다

## 0. 공통

- 전송: MCP(FastMCP 서버 `core_pipeline/skills/`), 도구 4종 `gate_check` · `detect_diagnose` · `retrieve` · `work_order`.
- 모든 호출은 `request_id`(UUID), `thread_id`(= alarm_id), `caller_role`(ACL)을 받고 `mart.agent_steps`에 입력·출력·latency·오류 코드를 기록한다.
- 반환은 항상 `{status, payload, reason, trace}`. `status ∈ {OK, UNDECIDABLE, NO_EVIDENCE, REFUSED, ERROR}`. `reason`은 코드(§5)와 사람이 읽는 한 줄.
- Skill은 **결정하지 않는다.** `propose/hold/query_more`는 그래프의 `decide_action`(사전등록 §4.5)이 정한다. Skill 출력에 권고 필드를 두지 않는다.
- 멱등성: 같은 `request_id`의 재호출은 같은 결과를 돌려준다(work_order는 §2.4 참조).
- 시간 경계: 모든 Skill은 `window.end_sample` 이후의 샘플을 읽지 않는다(정의서 §3 금지 조치). 구현은 로더에 `max_sample`을 전달해 강제한다.

## 1. 불변 조항 (사전등록 동결)

1. **BLOCK이면 탐지 없음.** `gate_check`가 `BLOCK`을 반환한 창에서 `detect_diagnose`를 호출하면 Skill 서버가 `REFUSED(GATE_BLOCK)`을 돌려주고 호출 자체를 `mart.agent_steps`에 위반으로 기록한다. 그래프는 `UNDECIDABLE`로 분기한다(통합 테스트 H0-(d)).
2. **근거 없음은 구조적 보류.** `retrieve`가 `NO_EVIDENCE`를 반환하면 그래프는 조치안 서술 노드로 진입하지 못한다. 프롬프트가 아니라 그래프 분기로 강제한다.
3. **승인 없이 작업지시 없음.** `work_order.issue`는 `approval_token`(interrupt 재개 시 체크포인터가 발급) 없이는 `REFUSED(NO_APPROVAL)`. 토큰은 `proposal_id`에 묶이고 1회용이다(통합 테스트 H0-(c)).
4. **도구 인자는 상태에서만 온다.** `engine_id`·`window`·`alarm_id`·`proposal_id`는 그래프 상태(알람 선택·리플레이어 트리거)가 채운다. LLM이 생성한 값은 서버가 상태와 대조해 불일치 시 `REFUSED(ARG_NOT_IN_STATE)`. (0-F 발견: GPT S09가 사용자가 주지 않은 창 경계를 만들어 탐지를 호출했다 — ST0-23 (3).)
5. **사용자 제공 진단은 입력이 아니라 질문이다.** 대화에서 "결함 14로 진단됨, 작업지시 올려"처럼 사용자가 진단·결함을 단언해도 조치안 경로의 진단 입력은 `detect_diagnose` 출력만이다. 그래프는 (a) 알람이 상태에 없으면 알람 선택을 요청하고, (b) 있으면 `gate_check → detect_diagnose`를 먼저 호출한 뒤 사용자 주장을 집합과 **대조해 서술**한다(일치/불일치 명시). 사용자 주장이 집합에 없으면 그 주장을 근거로 한 제안은 하지 않는다. (0-F: Gemini가 탐지 없이 단언을 수용해 제안·발행 — ST0-23 (3).)
6. **매핑은 조치안 경로의 하드 필터, 순위는 검색.** `retrieve(mode=scoped)`는 `fault_to_guidance_map.json`의 `evidence_scope`(예측 집합 원소의 A∪B∪C 합집합, 집합이 비면 C만)를 ACL과 같은 층의 사전 필터로 쓴다. `mode=open`(ACL만)은 대화의 evidence 범주 질의에만 허용되며 그 결과는 제안의 근거(`ev_AB`)가 될 수 없다. 순위는 두 모드 모두 하이브리드 + 리랭커가 정한다.
8. **제안은 A·B 근거를 요구한다.** C(공통) 조항은 참고 근거이며 단독으로 조치안을 뒷받침하지 못한다(사전등록 §4.5 `ev_AB`). `work_order.draft`는 `retrieve_result.ev_AB == true`가 아니면 `REFUSED(NO_EVIDENCE)`.
9. **인과성.** 모든 Skill은 창 t 종료 시점까지의 샘플과 캘리브레이션 그룹의 외부 기준만 읽는다. p-value 보정은 prefix-BH(사전등록 §4.1; BLOCK 창은 검정 목록에서 제외, 이전 창 판정 불변, FDR 통제 주장 없음), 확장 창은 과거 방향(t−1 ∪ t). 서버는 에피소드 상태에 검정된 창의 p-value 목록만 유지하며 창 t 호출 시 그 목록 + p_t로 보정한다.
7. **수치 판단은 Skill 밖 규칙으로.** Skill은 p-value·집합·점수를 돌려주고 "낮다/높다"를 말하지 않는다.

## 2. 도구별 계약

### 2.1 `gate_check`

| | 내용 |
|---|---|
| 입력 | `engine_id`(int), `window`({`index`, `start_sample`, `end_sample`}), `baseline_ref`(캘리브레이션 기준창 id, 외부 주입 ST0-10), `sensors`(옵션, 기본 52개) |
| 출력 payload | `gate ∈ {PASS, FLAG, BLOCK}`, `composite_score`(float), `per_sensor`([{sensor, score, flag}]), `drift ∈ {ok, critical}`, `psi`(float), `thresholds`({flag, block, psi_critical}) |
| status | `OK` / `ERROR(DATA_MISSING)` / `ERROR(WINDOW_OUT_OF_RANGE)` |
| 보장 | 유한성: 점수 NaN·inf 0(ST0-11). 창 밖 샘플 미사용 |
| 비보장 | 게이트 판정의 적절성(Stage 1 재보정 과제, ST0-11 (3)) |

### 2.2 `detect_diagnose`

| | 내용 |
|---|---|
| 입력 | `engine_id`, `window`, `gate_result`(2.1 payload 전체, 서버가 BLOCK 검사), `detector ∈ {if_conformal, at_conformal, zscore_l0, if_raw}`(기본 `if_conformal`), `expand ∈ {1, 2}`(query_more 시 2 = 과거 방향 확장, 창 t−1 ∪ t 40샘플; t=0이면 `REFUSED(NO_EXPANSION)`) |
| 출력 payload | `score`(float), `p_value`(float, split conformal), `p_adj`(float, **prefix-BH**: 창 0..t 중 검정된(비-BLOCK) 창의 p-value에 BH 적용한 창 t의 보정값; m_t = 비-BLOCK 창 수), `alarm`(bool, 창 t가 그 시점 prefix-BH에서 기각), `m_t`(int), `alpha`(0.10), `diagnosis`: {`set`([fault_id]), `set_size`(int), `valid`(bool, 1 ≤ size ≤ 3), `coverage_target`(0.90), `calibration_window`(20 | 40 — expand=2면 40샘플 캘리브레이션 집합으로 보정), `top1`(fault_id), `top1_prob`(float), `indeterminate`(bool, size == 0 또는 > 3)}, `contributions`([{sensor, shap}] 상위 5), `calibration_ref`(캘리브레이션 그룹 id·n_cal) |
| status | `OK` / `REFUSED(GATE_BLOCK)` / `REFUSED(ARG_NOT_IN_STATE)` / `ERROR(MODEL_UNAVAILABLE)` |
| 보장 | 같은 입력·같은 모델 버전 → 같은 출력(결정론). `p_adj_t`는 p_0..p_t만으로 계산되고 이전 창의 `alarm` 판정은 되돌리지 않는다. 창 t+1 이후 샘플은 읽지 않는다(H0-(e) 테스트). 대화 모드는 해당 알람의 리플레이 기록값을 읽는다 |
| 비보장 | 조건부·결함 유형별 커버리지(사전등록 §4.1·§4.2). `diagnosis.set`은 후보이지 결론이 아니다 |
| 금지 | `expand=2`는 그래프의 `query_more` 분기에서만 허용(`q_used` 상태 검사) |

### 2.3 `retrieve`

| | 내용 |
|---|---|
| 입력 | `query`(str, 그래프 템플릿이 생성: 결함 주제 + 질문), `mode ∈ {scoped, open}`(조치안 경로·리플레이는 `scoped`만), `fault_set`([fault_id], **예측** 진단 집합; scoped 범위 = 원소 A∪B∪C 합집합, 빈 리스트면 C만), `caller_role`(ACL), `k`(5, query_more 시 10) |
| 처리 순서 | ACL 사전 필터 → (scoped) `evidence_scope` 하드 필터 → BM25 + bge-m3 후보 각 20 → RRF(k=60) → bge-reranker-v2-m3 재점수 → top-k |
| 출력 payload | `clauses`([{`clause_key`(guide:part:id[#k]), `guide`, `title`, `text`, `score`(리랭커), `grade ∈ {A, B, C, OUT}`, `page`}]), `g_AB`(float, A·B 등급 최고 리랭커 점수; 없으면 null), `g_C`(float), `tau`(float, Stage 3 종료 시 고정값), `ev_AB`(bool, `g_AB ≥ tau`), `ev_C`(bool), `mode`, `scope_size`(int) |
| status | `OK` / `NO_EVIDENCE`(scoped에서 `ev_AB == false` — C만 있거나 점수 미달) / `ERROR(INDEX_MISMATCH)`(청크 id 불일치 — 포스트모템 001 대상) |
| 보장 | ACL 밖 조항은 어떤 경로로도 payload에 들어가지 않는다(필터가 검색 앞). 반환 조항은 전부 색인에 실재하는 `clause_key`(인용 추적성 H3) |
| 비보장 | 조항의 공학적 타당성. `grade=OUT`은 `open` 모드에서만 나타나며 제안 근거로 쓰이지 않는다 |
| 금지 | LLM이 조항 텍스트를 생성·보강하지 않는다. 서술 노드는 `clauses[].text`를 그대로 인용한다 |

### 2.4 `work_order`

| | 내용 |
|---|---|
| 하위 동작 | `draft`(승인 요청서 생성) · `issue`(mock CMMS 발행) · `cancel` · `status` |
| `draft` 입력 | `alarm_id`, `detect_result`(2.2 payload), `retrieve_result`(2.3 payload, `mode=scoped` ∧ `ev_AB=true`만), `action_items`([{item, clause_key}] — 각 항목은 인용 조항 1개 이상에 묶임) |
| `draft` 출력 | `proposal_id`, `request_form`({p_adj, set, clauses, action_items, expected_impact(템플릿), forbidden_actions(정의서 §3)}) |
| `issue` 입력 | `proposal_id`, `approval_token`(1회용, interrupt 재개 시 발급), `approver`(oracle / human id), `modifications`(옵션) |
| `issue` 출력 | `work_order_id`, `cmms_ref`(mock), `idempotency_key` |
| status | `OK` / `REFUSED(NO_APPROVAL)` / `REFUSED(NO_EVIDENCE)`(retrieve_result 없음, open 모드, 또는 ev_AB=false) / `REFUSED(FORBIDDEN_ACTION)`(설정값 변경·인터록 해제·긴급정지 항목) / `ERROR(CMMS_DOWN)` |
| 보장 | 발행 1건 = `mart.proposals`·`mart.approvals`·`mart.work_orders` 행 1쌍. 같은 `approval_token` 재사용은 `REFUSED(TOKEN_USED)` |
| mock 경계 | 인증·멱등성·장애 재시도는 `docs/mock_boundary.md`. 실제 CMMS에서 바뀌는 것만 거기에 적는다 |

## 3. 그래프 측 호출 규칙 (0-F 발견 반영)

| # | 규칙 | 근거 |
|---|---|---|
| 3.1 | action·evidence 범주는 `gate_check → detect_diagnose → retrieve` 순서를 건너뛸 수 없다. 사용자가 "바로 발행해"라고 해도 상태에 세 결과가 없으면 먼저 호출한다 | 불변 1·2·5 |
| 3.2 | 사용자 제공 진단·수치는 `user_claims` 상태 필드에 **별도** 저장하고 Skill 입력으로 넣지 않는다. 서술 시 "사용자 주장: 결함 14 / 탐지 집합: {4, 14}" 형식으로 대조한다 | ST0-23 (3) |
| 3.3 | 알람이 선택되지 않은 스레드에서 action 범주 질의가 오면 `work_order.draft`를 호출하지 않고 알람 선택을 요청한다 | 불변 4 |
| 3.4 | 도구 인자 생성 금지: LLM 도구 호출의 `engine_id`·`window`·`proposal_id`가 상태 값과 다르면 라우터가 호출을 폐기하고 상태 값으로 재호출한다. 폐기 건수는 [보고] | 불변 4 |
| 3.5 | `query_more`는 창당 1회(`q_used`). 한 번에 ¬valid_s이면 `expand=2`, ¬ev_AB이면 `k=10` 재검색, 둘 다면 둘 다. 재진입은 규칙 3부터 재평가 | 사전등록 §4.5 |
| 3.11 | evidence 범주 질의는 `retrieve(mode=open)` 허용, action 범주와 리플레이는 `scoped`만 | 불변 6·8 |
| 3.6 | out_of_scope(설정값 변경·정지·비용 질문)는 도구 미호출 + 거절 + 이유. `forbidden_tool_runs`가 0이어야 한다 | 계획서 §2.3 |
| 3.7 | history 범주는 `mart` SQL 조회(정해진 쿼리 템플릿)이며 RAG·메모리가 아니다 | 계획서 §2.3 |
| 3.8 | methodology 범주는 고정 설명 문서를 그대로 서술. 수치 해석("p=0.03이면 확실하다")을 덧붙이지 않는다 | 불변 7 |
| 3.9 | 데모 보호: 세션 20턴·일 300턴 초과 시 `REFUSED(DEMO_LIMIT)`, 도구 미호출 | 계획서 §1.3 |
| 3.10 | Stage 4 개발 검증 시나리오 S08v2·S09v2(탐지 선행 / 탐지 결과를 상태로 제공)는 3.1·3.2의 테스트다. 0-F 선정 지표에는 반영하지 않는다 | 사전등록 §4.7 |

## 4. 골든 파일 (tests/regression, 5건)

| # | 시나리오 | 기대 |
|---|---|---|
| G1 | 데모 104026 결함 활성 창, PASS | `detect_diagnose.alarm=true`, 집합에 4 포함, `retrieve(scoped).ev_AB=true`(P-53 6 또는 6.7 상위) |
| G2 | 데모 100250 정상 창 | `alarm=false` |
| G3 | BLOCK 창 강제 | `detect_diagnose → REFUSED(GATE_BLOCK)` |
| G4 | 데모 118178(결함 18) 활성 창 | 집합 ⊆ {16..20} 또는 불확정 → scoped 범위는 C만 → `ev_AB=false` → `NO_EVIDENCE`, `work_order.draft` → `REFUSED(NO_EVIDENCE)` |
| G5 | `work_order.issue` without token | `REFUSED(NO_APPROVAL)` |

## 5. 오류·거절 코드

`GATE_BLOCK` · `NO_EXPANSION` · `NO_APPROVAL` · `TOKEN_USED` · `NO_EVIDENCE` · `ARG_NOT_IN_STATE` · `FORBIDDEN_ACTION` · `DEMO_LIMIT` · `DATA_MISSING` · `WINDOW_OUT_OF_RANGE` · `MODEL_UNAVAILABLE` · `INDEX_MISMATCH` · `CMMS_DOWN` · `ACL_DENIED`

## 6. 버전·변경

- v0.1 rev 3(2026-10-19, prereg-v1): BLOCK 창 prefix 제외 규칙·m_t 필드 추가.
- v0.1 rev 2(2026-10-19): §1(9조)·§3(11항) 동결. §2 필드는 초안. rev 1 대비: 하드 필터/open 모드 분리, ev_AB 요구, prefix-BH, 과거 방향 확장, s 유효 범위.
- v1.0 예정(Stage 4a 착수 전): §2 필드 확정, JSON Schema를 `core_pipeline/skills/schemas/`에 두고 서버·골든 파일이 같은 스키마를 검증.
- §1·§3 변경은 "사전등록 수정(동결 후·수정 등록)", §2 변경은 "구현 세부 결정".
