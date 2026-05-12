# 프로젝트 요약서 — 산업 센서 거버넌스 & 예지정비 플랫폼

---

## 프로젝트 개요

**주제:** 센서 데이터 품질 거버넌스 + 잔여수명(RUL) 예측 + MLOps 플랫폼

**핵심 질문:** 센서 데이터를 신뢰할 수 없는 상황에서, 예지정비 모델의 예측을 신뢰할 수 있는가?

**데이터:** NASA C-MAPSS (터보팬 엔진 run-to-failure 시뮬레이션, 4개 subset, 21개 센서)



---

## 아키텍처

```
Raw Sensor Data (C-MAPSS)
        |
  ┌─────┴──────┐
  | LAYER 1    |  데이터 품질 거버넌스
  | Governance |  완전성 / PSI / 센서 간 상관 → Quality Gate
  └─────┬──────┘
        |
  ┌─────┴──────┐
  | LAYER 2    |  이상탐지 + RUL 예측
  | Analytics  |  IF (AUROC 0.955) + TFT (RMSE 16.39)
  └─────┬──────┘
        |
  ┌─────┴──────┐
  | LAYER 3    |  MLOps 플랫폼
  | Platform   |  MLflow + DVC + FastAPI + Streamlit
  └────────────┘
```

---

## 핵심 차별화: Corruption Experiment

5가지 데이터 품질 문제를 의도적으로 주입하여 RUL 예측에 미치는 영향을 정량화하고, 거버넌스 레이어의 회복률(Recovery Rate)을 측정.

### 최종 전략 매핑

| 오염 유형 | 전략 | 회복률 | 근거 |
|---|---|---|---|
| Concept Drift | 재학습 알림 | **100%** (3/3 탐지) | 데이터 수정 불가 → 탐지 후 재학습 트리거 |
| Gaussian Noise | 평활화 | **72–79%** | 이동평균으로 노이즈 억제, 추세 보존 |
| Sensor Drift | 스마트 센서 제거 | **80.1%** (high only) | PSI > 0.2인 센서만 제거 |
| Random Missing | 통과 (미개입) | 0% (안전) | XGBoost 내장 NaN 처리가 보간보다 우수 |
| Stuck-at Fault | 통과 (미개입) | 0% (안전) | 정상 범위 내 고정값 → 제거 비용 > 피해 |

### 히스토리 (5회 반복)

| 버전 | 변경 | Positive | Neutral | Negative |
|---|---|---|---|---|
| v1 | Forward fill 전체 적용 | 0 | 9 | **6** |
| v3 | Type-aware 전략 분기 | 5 | 1 | **5** |
| v3+ | PSI 심각도 게이트 | 4 | 7 | **1** |
| **최종** | **Stuck-at → 통과 전환** | **7** | **8** | **0** |

**도출된 핵심 원칙:** 거버넌스는 "항상 개입"이 아니라 "개입 비용 vs 피해를 비교하여 이득이 있을 때만 개입"이다. v1의 -1,488% 역효과 → 최종 0/15 역효과.

---

## 실험 결과 요약

### 이상 탐지 (Layer 2A)

| 모델 | AUROC | F1 (RUL=50) | 유형 |
|---|---|---|---|
| Rolling Z-score | 0.327 | 0.456 | 통계 기반 |
| **Isolation Forest** | **0.955** | **0.789** | 전통 ML |
| Anomaly Transformer | 0.894 | 0.052 | DL SOTA (ICLR 2022) |

**발견:** IF(2008) > AT(2022). C-MAPSS의 점진적 degradation에서는 feature space 전역 밀도 탐지(IF)가 시점 간 association 변화 탐지(AT)보다 효과적. **SOTA ≠ 최적. 도메인 특성이 모델 선택을 결정.**

### RUL 예측 (Layer 2B)

| 모델 | Test RMSE | NASA Score | 유형 |
|---|---|---|---|
| XGBoost | 18.81 | 914 | Tabular ML |
| Bi-LSTM | 20.11 | 611 | DL (LSTM) |
| **TFT (best)** | **16.39** | **482** | DL (Transformer) |

TFT Variable Selection: sensor_11, 3, 4, 14 → 거버넌스 우선 관리 센서로 직결.
TFT 학습 불안정성: 동일 파라미터 → RMSE 16–42 범위. 배포 시 다중 시드 앙상블 필수.

### Cross-Subset Transfer (전사 확산 시뮬레이션)

| 전이 | RMSE | Δ % | 의미 |
|---|---|---|---|
| FD001 → FD001 | 18.81 | baseline | 같은 라인 |
| FD001 → FD003 | 21.84 | **+16%** | 고장 유형만 변화 |
| FD001 → FD002 | 53.99 | **+187%** | 운전 조건만 변화 |
| FD001 → FD004 | 54.95 | +192% | 둘 다 변화 |

**발견:** 전이 실패의 지배적 원인은 고장 유형(+16%)이 아니라 **운전 조건(+187%)**. 전사 확산 시 운전 조건 정규화가 최우선.

### Cross-Stage Integration

| 실험 | 결과 | 시사점 |
|---|---|---|
| IF score → RUL feature | RMSE +0.046 (실패) | 동일 소스 feature는 정보 중복 |
| TFT importance → targeted corruption | Noise 5.3x 비율 (성공) | Importance가 noise 민감도를 정확히 예측 |

---

## MLOps 플랫폼 (Layer 3)

| 컴포넌트 | 역할 | 상태 |
|---|---|---|
| MLflow | 실험 추적 (파라미터/메트릭/모델 버저닝) |  SQLite backend, XGBoost + Bi-LSTM 로깅 완료 |
| FastAPI | Quality Gate → RUL 예측 E2E 서빙 |  Swagger UI 확인 완료 |
| DVC | 데이터 버저닝 + 재현 가능 파이프라인 |  preprocess stage 검증 완료 |
| Streamlit | 5탭 대시보드 (Community Cloud 배포) |  라이브 배포 |

---

## 발견된 원칙

1. **거버넌스 ≠ 항상 개입.** 개입 비용 vs 피해를 비교. v1의 -1,488% → 최종 0% 역효과.
2. **SOTA ≠ 최적.** IF(2008)가 Anomaly Transformer(2022)를 점진적 degradation에서 이김.
3. **Variable Importance의 활용 범위에 한계.** Noise 민감도(5.3x)는 예측하나 drift 민감도(1.0x)는 미예측.
4. **실패한 실험도 문서화 가치.** IF score → RUL feature 실패 = 정보 중복이라는 유효한 발견.
5. **TFT 불안정성은 구조적.** 동일 파라미터 → RMSE 16–42. 배포 시 다중 시드 프로토콜 필수.
6. **전사 확산의 지배적 장벽은 운전 조건.** 고장 유형 변화(+16%)보다 운전 조건 변화(+187%)가 10배 이상 치명적.
