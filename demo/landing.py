"""demo/landing.py: public landing page for https://process-reliability-agent.streamlit.app/ (no DB, no LLM, no secrets).

    streamlit run demo/landing.py

What it shows: what the v2 Process Reliability Agent is, how the conversational demo works, the 10 demo alarm presets
(allow-listed fields of the committed manifest `outputs/demo_presets.json`, no truth labels), the evaluation status,
and how to run the real interactive demo locally. The interactive agent needs Postgres and the two Skill servers and
therefore runs locally (`demo/app.py`); a public replay version is being prepared on the `demo-public` branch.
Text here is drawn from the project plan v1.1 and the README; no performance figures for v2 are stated because the
preregistered evaluation has not been run.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "outputs" / "demo_presets.json"
REPO = "https://github.com/ChangyeolAidenOh/sensor-governance-platform"

PRESET_COLUMNS = ("preset", "kind", "display_window", "n_windows", "n_block", "alarm_windows", "pred_set", "top1")

st.set_page_config(page_title="Process Reliability Agent", layout="wide")

st.title("Process Reliability Agent")
st.caption("sensor-governance-platform v2: 공정 센서 이상 탐지, 보정된 불확실성, 문서 근거, 사람 승인 흐름을 갖춘 대화형 LangGraph 에이전트")

DEMO_URL = (os.environ.get("PUBLIC_DEMO_URL") or "").strip()
try:
    DEMO_URL = (st.secrets.get("PUBLIC_DEMO_URL") or DEMO_URL).strip()
except Exception:
    pass

if DEMO_URL:
    st.success("대화형 데모를 바로 써 볼 수 있습니다. 사례 10개는 사전계산된 실제 결과를 재생하고, 마지막 사례는 승인 게이트를 보여 주기 위한 합성 사례입니다. "
               "첫 접속 시 컨테이너가 깨어나는 데 15~30초가 걸릴 수 있습니다.")
    st.link_button("대화형 데모 열기", DEMO_URL, type="primary")
else:
    st.info("이 페이지는 프로젝트 소개용 정적 페이지입니다. 실제 대화형 데모는 Postgres와 Skill 서버가 필요해 현재는 로컬에서 실행합니다(아래 '로컬에서 실행하기'). "
            "공개 URL용 재생 데모는 준비 중입니다.")
st.caption("평가(Stage 5)는 사전등록된 가설대로 준비 중이며, 결과 수치는 아직 없습니다.")

st.header("무엇을 하는가")
st.markdown("""
화학공정 센서 스트림(Tennessee Eastman Process)에 대해

1. **데이터 품질 게이트**를 통과한 구간에만 탐지 모델을 적용하고,
2. 이상 점수와 결함 유형 추정에 **보정된 불확실성**(conformal p-value, conformal prediction set)을 붙이며,
3. 공정 안전 기술지침(KOSHA GUIDE P 시리즈)에서 **근거 조항을 인용**해 조치안을 만들고,
4. **엔지니어가 승인한 뒤에만** 작업지시를 발행합니다.

방문자는 데모 알람 하나를 고르고, 그 알람에 대해 묻고, 에이전트가 무엇을 호출했고 무엇을 근거로 답했는지 트레이스로 보며, 조치안을 승인, 수정, 거부할 수 있습니다.
근거 조항을 찾지 못하면 조치안을 내지 않고 보류합니다.
""")

st.header("구조")
st.code("""
user question -> rule router (default) / LLM router (option) -> LangGraph graph
                                                                  |
        +---------- Skill calls (decided by the graph; the LLM never calls tools) ----------+
        |  skills_detect  : quality gate -> detection (Isolation Forest) -> conformal p-value -> diagnosis set  |
        |  skills_retrieve: fault-scope filter -> hybrid search -> reranker (tau) -> clause citation (grade A/B) |
        +----------------------------------------------------------------------------------------------------+
                                                                  |
                 action policy: propose / hold / query_more  ->  approval gate (approve, modify, reject)
                                                                  |
                 template narration (default) + LLM summary (option) -> answer + call trace + cost ledger
""", language="text")

st.markdown("""
- 본체 LLM은 Stage 0 스파이크의 사전 규칙(형식 오류 0, 예외 0인 후보 중 턴당 비용 최소)으로 `claude-haiku-5-5`가 선정됐고, 기본 모드는 규칙 라우터와 템플릿 서술($0)입니다.
- 모든 LLM 호출은 예약 -> 정산 장부로 기록되며, 비용은 네 묶음으로 분리해 표시합니다.
- 리플레이 평가(H1)는 같은 Skill, 같은 정책을 사전등록된 평가 풀(최대 60 run id, 1,260 에피소드)에 적용해 세 기준선과 행동 비용을 비교하는 장치입니다.
""")

st.header("데모 알람 프리셋")
if MANIFEST.is_file():
    try:
        doc = json.loads(MANIFEST.read_text(encoding="utf-8"))
        rows = []
        for p in doc.get("presets", []):
            v = {k: p.get(k) for k in PRESET_COLUMNS}
            rows.append({"preset": v["preset"], "종류": v["kind"], "표시 창": v["display_window"], "창 수": v["n_windows"],
                         "BLOCK 창": v["n_block"], "알람 창": ", ".join(str(w) for w in (v["alarm_windows"] or [])) or "-",
                         "진단 후보 수": len(v["pred_set"] or []), "top1": v["top1"]})
        st.dataframe(rows, width="stretch", hide_index=True)
        st.caption(f"출처: outputs/demo_presets.json (schema {doc.get('schema')}, code {str(doc.get('code_commit'))[:7]}). "
                   "표시 규칙: 창 0..47 중 첫 알람 창. 정답 결함 라벨은 데모 매니페스트에 넣지 않습니다.")
    except Exception as e:
        st.warning(f"프리셋 표를 읽지 못했습니다: {type(e).__name__}")
else:
    st.warning("outputs/demo_presets.json이 이 배포에 없습니다.")

st.header("평가 (사전등록)")
st.markdown("""
가설, 수용 기준, 분석 절차는 결과를 보기 전에 git 태그 `prereg-v1`로 고정했습니다.

| 가설 | 내용 |
|---|---|
| H1 | 세 신호(탐지 p-value, 진단 집합 크기, 문서 근거성)로 행동을 고르는 정책이 "탐지되면 항상 제안", 고정 임계치 기준선보다 리플레이 행동 비용이 낮다 (1차 종점) |
| H2 | 인과적 알람 규칙이 기록한 알람의 실현 오탐률이 사전 지정 범위 안에 든다 |
| H3 | 조치 근거는 실제 조항으로 추적 가능하고, 근거가 없으면 조치안을 내지 않는다 (RAG 평가셋 50문항) |
| H4 | 결함 유형 conformal prediction set의 커버리지가 목표 근처다 |
| H5, H6 | 대화 60문항 x 3회 반복에서 라우터 범주 정확도, 답변 정확도, 범위 밖 거절률 |

현재 상태: 평가 도구와 전용 평가 DB를 준비하는 단계이며, 본 평가는 동결 뒤 한 번만 실행합니다.
""")

st.header("로컬에서 실행하기")
st.markdown(f"""
대화형 데모는 저장소의 `v2` 브랜치에서 실행합니다. Postgres(docker compose)와 두 Skill 서버가 필요하며, 상세 절차는 저장소 README에 있습니다.

```bash
git clone {REPO} && cd sensor-governance-platform && git checkout v2
docker compose up -d postgres
streamlit run demo/app.py
```

- 저장소: {REPO}
- v1(센서 거버넌스, 예지정비) 결과와 대시보드: [v1.0 릴리스]({REPO}/releases/tag/v1.0), [sensor-governance-platform.streamlit.app](https://sensor-governance-platform.streamlit.app/)
""")

st.caption("원천 프로젝트: sensor-governance-platform v1 / consumer-signal-agentic-platform / scm-control-tower-agent. 가져온 것, 바꾼 것, 새로 만든 것은 README Lineage 표에 분리해 기록합니다.")
