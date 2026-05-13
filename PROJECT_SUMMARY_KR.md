# 프로젝트 요약서 — 산업 센서 거버넌스 & 예지정비 플랫폼

---

## 프로젝트 개요

**주제:** 센서 데이터 품질 거버넌스 + 잔여수명(RUL) 예측 + MLOps 플랫폼

**핵심 질문:** 센서 데이터를 신뢰할 수 없는 상황에서, 예지정비 모델의 예측을 신뢰할 수 있는가?

**데이터:** NASA C-MAPSS (터보팬 엔진 run-to-failure, 4개 subset, 21개 센서, 508개 엔진)

**기간:** 2026년 5월 | **유형:** 독립 프로젝트

**Dashboard:** [sensor-governance-platform.streamlit.app](https://sensor-governance-platform.streamlit.app/)

---

## 3-Layer 아키텍처

```
Layer 1: 데이터 품질 거버넌스
  -> 완전성 / PSI / 센서 간 상관 -> Quality Gate
  -> Corruption Experiment (5종 x 3단계 = 15 시나리오)

Layer 2: 분석
  2A: 이상탐지 — IF (AUROC 0.955) > AT (0.894)
  2B: RUL 예측 — TFT (RMSE 16.39) > XGBoost (18.81)
  + SHAP / Variable Importance / MC Dropout UQ

Layer 3: MLOps 플랫폼
  MLflow + DVC + FastAPI + Streamlit
  + Monitoring (드리프트 탐지 -> 재학습 트리거)
  + Model Card 자동 생성
```

---

## 핵심 차별화: Corruption Experiment — 5회 반복 진화

### 진화 과정

| 버전 | 변경 | Positive | Neutral | Negative | 핵심 문제 |
|---|---|---|---|---|---|
| v1 | Forward fill 전체 | 0 | 9 | **6** | NaN에만 작동, drift/noise에 무효. random missing -1,488% |
| v2 | Type-aware 분기 | - | - | - | sensor_drop -> RMSE 42.76 폭발 (14개 센서 전부 제거) |
| v3 | 4개 센서로 제한 | 5 | 1 | **5** | 경미한 corruption에서 drop 비용 > 피해 (-332%) |
| v3+ | PSI 심각도 게이트 | 4 | 7 | **1** | stuck-at은 PSI로 탐지 불가 |
| v3++ | Variance 탐지 추가 | 4 | 7 | **1** | 2/4만 탐지, 비용이 더 커서 -34.1%로 악화 |
| **최종** | **Stuck-at -> passthrough** | **7** | **8** | **0** | **탐지 신뢰도 미확보 -> 미개입이 최선** |

### 최종 전략 매핑

| 오염 유형 | 전략 | 회복률 | 근거 |
|---|---|---|---|
| Concept Drift | 재학습 알림 | **100%** (3/3 탐지) | 데이터 수정 불가 -> 탐지 후 재학습 트리거 |
| Gaussian Noise | 평활화 | **72-79%** | 이동평균으로 노이즈 억제, 추세 보존 |
| Sensor Drift | 스마트 센서 제거 | **80.1%** (high only) | PSI > 0.2인 센서만 제거, 경미한 건 유지 |
| Random Missing | 통과 | 0% (안전) | XGBoost 내장 NaN 처리가 보간보다 우수 |
| Stuck-at Fault | 통과 | 0% (안전) | 정상 범위 내 고정 -> PSI/variance 양쪽 미신뢰 |

> **도출 원칙:** 거버넌스 ≠ 항상 개입. 개입 비용 vs 피해를 비교하고, 탐지 신뢰도가 확보된 유형에만 개입. "개입하지 않는 판단"도 거버넌스의 가치. v1의 -1,488% -> 최종 0/15 역효과.

---

## 이상탐지 (Layer 2A) — 3가지 평가 체계 병렬

| 모델 | AUROC | Point F1 | Range F1 | Event F1 | 유형 |
|---|---|---|---|---|---|
| Rolling Z-score | 0.327 | 0.456 | 0.307 | 0.298 | 통계 기반 |
| **Isolation Forest** | **0.955** | **0.789** | **0.594** | **0.564** | 전통 ML (2008) |
| Anomaly Transformer | 0.894 | 0.052 | - | - | DL SOTA (ICLR 2022) |

**3가지 평가 체계 모두에서 IF 일관 우위** -> 평가 방법론에 의존하지 않는 결과.

**USAD 의도적 생략:** Z-score -> IF로 "왜 ML이 필요한가" 서사 성립. IF vs AT 직접 비교가 3-way보다 인사이트 풍부.

**AT 학습 4회 시도:** NaN 폭발(log(0)) -> disc 정체(minimax 상쇄) -> epoch 불안정(AUROC 0.519->0.680->0.573 진동) -> d_model 64->128, lr 1e-4->5e-5로 AUROC 0.894. Minimax 학습 불안정성은 GAN 훈련 역학과 동일 원리.

**IF > AT 이유:** C-MAPSS 점진적 degradation -> feature space 전역 밀도 이탈(IF 탐지). AT는 시점 간 association 변화를 보지만, 천천히 악화되는 시계열에서는 인접 시점 association이 유지.

---

## RUL 예측 (Layer 2B)

| 모델 | Test RMSE | NASA Score | 유형 |
|---|---|---|---|
| XGBoost | 18.81 | 914 | Tabular ML |
| Bi-LSTM | 20.11 | 611 | DL (LSTM) |
| **TFT (best)** | **16.39** | **482** | DL (Transformer) |
| TFT (worst) | 41.82 | 18,153 | 동일 파라미터 |

**TFT 학습 불안정성 (5회 시도):** 동일 파라미터 -> RMSE 16-42 범위. 디버깅 중 argparse default vs 함수 default 불일치 발견(python -m 실행 시 파라미터 변경이 무시됨). VSN winner-take-all: Run 3 sensor_14 72% vs Run 2 sensor_11/3/4/14 균등 분산.

### 해석 가능성: SHAP vs TFT Variable Importance

| 순위 | SHAP (XGBoost) | TFT (best run) |
|---|---|---|
| 1 | sensor_4 (12.1%) | sensor_11 (16.3%) |
| 2 | sensor_11 (10.8%) | sensor_3 (15.7%) |
| 3 | sensor_8 (8.7%) | sensor_4 (15.6%) |
| 4 | sensor_15 (8.1%) | sensor_14 (15.3%) |

**Top-5 overlap 3/5:** sensor_4, sensor_11, sensor_15가 두 방법 합의 -> 가장 신뢰할 수 있는 거버넌스 우선 센서.

### 불확실성 정량화 (MC Dropout)

| Dropout | 90% CI Coverage | 보정 상태 |
|---|---|---|
| 0.3 | 0.42 | 과신(over-confident) |
| 0.5 | 0.56 | 과신(over-confident) |

MC Dropout은 인식론적(epistemic) 불확실성만 포착. 우연적(aleatoric) 불확실성 미반영 -> CI가 체계적으로 좁음. Conformal Prediction 등 보정 기법 필요 (향후 과제).

---

## Cross-Subset Transfer — 전사 확산 시뮬레이션

### 전이 실험 (FD001에서 학습 -> 타 subset에서 평가)

| 전이 | RMSE | Delta | 의미 |
|---|---|---|---|
| FD001 -> FD001 | 18.81 | baseline | 같은 라인 |
| FD001 -> FD003 | 21.84 | **+16%** | 고장 유형만 변화 |
| FD001 -> FD002 | 53.99 | **+187%** | 운전 조건만 변화 |
| FD001 -> FD004 | 54.95 | +192% | 둘 다 변화 |

### 전체 Subset 벤치마크 (각 subset 독립 학습/평가)

| Subset | 운전 조건 | 고장 유형 | XGBoost RMSE | IF AUROC |
|---|---|---|---|---|
| FD001 | 1 | 1 | 18.81 | 0.955 |
| FD002 | 6 | 1 | 42.34 | 0.954 |
| FD003 | 1 | 2 | 17.81 | 0.953 |
| FD004 | 6 | 2 | 50.07 | 0.947 |

**핵심 발견:**
- **운전 조건 변화(+187%)가 고장 유형 변화(+16%)를 10배 이상 압도.** FD002 ~ FD004 -> 운전 조건이 전이 실패의 97%를 설명.
- **IF는 모든 subset에서 안정(AUROC 0.947-0.955).** 이상탐지는 전사 확산 가능. RUL은 조건별 모델 필요.
- **Corruption Experiment의 concept drift(+25.69)와 Transfer(+35.19)는 동일 원리의 다른 표현.** 데이터 분포가 바뀌면 모델이 무너진다. 해결책도 동일: PSI로 탐지, 재학습 트리거.

---

## Cross-Stage Integration

| 실험 | 결과 | 발견 |
|---|---|---|
| IF score -> RUL feature | RMSE +0.046 (실패) | 동일 소스에서 파생 -> 정보 중복 |
| TFT importance -> targeted corruption | Noise 5.3x (성공) | Importance는 noise 민감도만 예측, drift 민감도는 미예측 |

**교훈:** Stage 간 연결이 자동으로 개선을 만들지 않는다. 기존에 없는 information source가 필요.

---

## MLOps 플랫폼 (Layer 3)

| 컴포넌트 | 역할 | 특이사항 |
|---|---|---|
| MLflow | 실험 추적 (SQLite) | M2 segfault -> OMP_NUM_THREADS=1, sklearn.log_model -> pickle 우회 |
| FastAPI | Quality Gate -> RUL E2E 서빙 | Swagger UI 동작 확인 |
| DVC | 데이터 버저닝 + 파이프라인 | .gitignore: data/raw/ -> data/raw/CMAPSSData/ 변경 필요 |
| Monitoring | PSI 드리프트 탐지 -> 재학습 트리거 | FD001 vs FD002: 15개 센서 critical, 자동 retrain |
| Model Card | 모델 문서 자동 생성 | 3모델 각각 limitations/failure modes/governance 요구사항 포함 |
| Streamlit | 5탭 대시보드 + Cloud 배포 | use_container_width -> width="stretch", Arrow mixed type 해결 |
| Docker | docker-compose.yml | MLflow + API + Dashboard 3서비스 스택 |

---

## 발견된 원칙 7가지

1. **거버넌스 ≠ 항상 개입.** 개입 비용 vs 피해 비교. v1의 -1,488% -> 최종 0/15 역효과.
2. **SOTA ≠ 최적.** IF(2008) > AT(2022). 도메인 특성이 모델 선택을 결정.
3. **Variable Importance의 범위 한계.** Noise 민감도(5.3x)는 예측, drift 민감도(1.0x)는 미예측. SHAP/TFT 합의 센서 3/5.
4. **실패한 실험도 문서화 가치.** IF->RUL 정보 중복. MC Dropout coverage 0.56(구조적 한계).
5. **TFT 불안정성은 구조적.** 동일 파라미터 -> RMSE 16-42. 다중 시드 앙상블 필수.
6. **전사 확산의 지배적 장벽은 운전 조건.** +187% vs +16%. IF는 안정(0.947-0.955), RUL은 붕괴.
7. **Corruption과 Transfer는 동일 원리.** 데이터 분포가 바뀌면 모델이 무너진다. PSI로 탐지, 재학습.


