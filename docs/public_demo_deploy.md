# 공개 데모 배포 (demo-public 브랜치)

랜딩: https://process-reliability-agent.streamlit.app (Community Cloud, `demo/landing.py`).
대화형 데모: Cloud Run(도쿄) 컨테이너 + Neon Postgres(싱가포르, DB `sgp_v2`). 결정일 2026-11-05.

## 설계 요약

- 두 Skill 서버는 원래 "한 번 계산하고 저장한 결과를 재생"하는 구조다(detect: `mart.episode_windows`, retrieve: `mart.agent_steps`).
  공개 DB에는 데모 10 에피소드를 사전계산해 두고, 공개 앱은 저장된 행만 읽는다(`sql/006` 주석의 배포 설계 그대로).
- 공개 앱에서 바뀌는 것은 세 가지뿐이고 서버·검색 코드는 수정하지 않는다.
  1. `skills_detect` 서버만 서브프로세스로 띄운다(저장된 창 서빙 + 작업지시 초안·승인·발행 실행). `retrieve`는
     `core_pipeline/skills/public_skills.py`의 `StoredRetrieve`가 저장 행에서 답한다(bge 모델 없음). 저장되지 않은 검색은
     `REFUSED(DEMO_LIMIT)`이고 앱이 안내 문구를 띄운다.
  2. 연결 풀 `min_size=0`, 연결 검사 켬: Neon이 5분 유휴 뒤 잠들고 다음 접속에 자동으로 깬다.
  3. `SGP_DEMO_SYNTHETIC_PRESETS=11`: 프리셋 11만 합성 승인 사례(근거 등급 상향 + 정책 제안 고정, ST4-29 확장). 나머지 10개는 실제.
- 라벨 울타리: `window_labels`·`staging`은 공개 DB에 그대로 있지만 앱은 `sgp_agent`(앱)·`sgp_skill`(detect 서버, sgp_agent 상속)로만
  접속한다. 관리자 DSN은 Cloud Run secrets에 넣지 않는다.
- 사용 상한(`demo/public_guard.py`): 세션 15턴·20분, 하루 300턴·누적 90분(UTC). LLM 기본 off.

## 로컬 환경 변수 규칙

`.env`의 표준 키(`SGP_PG_DSN`, `SGP_AGENT_PG_DSN`, `SGP_TEST_PG_DSN`, ...)는 평가용 그대로 둔다. 공개 데모용은 접두사 `SGP_PUBLIC_`:

- `SGP_PUBLIC_NEON_ADMIN_DSN`: Neon 직접 연결(호스트에 `-pooler` 없음), DB `sgp_v2`
- `SGP_PUBLIC_NEON_AGENT_DSN`, `SGP_PUBLIC_NEON_SKILL_DSN`: `python scripts/public_demo_build.py neon`이 `build/public/neon_roles.env`에 써 준다

규칙: 평가 셸에서는 `SGP_PUBLIC_*`를 export하지 않는다. 로컬 빌드 DB 이름은 `sgp_public`이고, 모든 공개 스크립트는 dbname에
`eval`이 들어가거나 `sgp_v2`(로컬 프로젝트 DB)이면 거부한다. Cloud Run에는 표준 키 두 개(`SGP_AGENT_PG_DSN`, `SGP_PG_DSN`)만 있다.

## 절차

worktree(`~/PycharmProjects/sensor_governance_platform_demo`, 공유 `.venv` 활성화)에서. v2 트리는 `../sensor_governance_platform`.

```bash
bash cloudrun/link_local_assets.sh            # data/processed, 모델 바이너리, 검색 인덱스를 v2 트리로 심링크, .env 복사
python scripts/public_demo_build.py local     # createdb → load → schema → roles → saver → verify → prune → precompute → manifest → prime → check → dump
python scripts/public_demo_build.py neon      # restore → roles(새 비밀번호) → check
bash cloudrun/prepare_image_assets.sh         # cloudrun/assets에 모델·parquet 복사(이미지용)
bash cloudrun/deploy.sh                       # secrets + gcloud run deploy --source .
```

- `local`은 단계별로 재개할 수 있다(`--from prune`, `--only check`). `createdb`는 `sgp_public`이 있으면 멈추고 `--reset`으로 지우고 다시 만든다.
- `verify`는 전체 적재 상태에서 돌아야 한다(3b가 그룹 크기를 동결값과 비교). `prune`은 그다음이다.
- `prime`은 로컬 bge 모델을 로드한다(4 GB, 수 분). 실패하면 `build/public/prime_report.json`의 `failures`를 본다. 합성 프리셋이
  `waiting_approval`에 도달하지 못하면 `--synthetic-run-id`로 다른 run id를 골라 `--from prune`부터 다시 한다.
- `neon` 뒤 `build/public/neon_roles.env`의 두 DSN 줄을 `.env`에 추가한다(파일은 build/ 아래라 커밋되지 않는다).
- 배포 뒤 Cloud Run URL을 Community Cloud의 랜딩 앱 Secrets에 `PUBLIC_DEMO_URL = "https://..."`로 넣으면 랜딩에 "대화형 데모 열기" 버튼이 생긴다.

## 비용과 차단

- Cloud Run 설정: 1 vCPU, 1 GiB, max 1 인스턴스, min 0, CPU는 요청 중에만, 타임아웃 30분, 세션 어피니티, 도쿄(Tier 1).
- 무료 한도(결제 계정 합산): 월 180,000 vCPU-초 = 1 vCPU로 50시간의 "웹소켓이 열려 있는 시간". 리크루터 트래픽(월 수 시간)은 $0.
- 이론상 최악(누군가 탭을 한 달 내내 열어 둠)은 약 $60/월이고, 앱의 턴·시간 상한은 DB·compute 부하를 막을 뿐 Streamlit 웹소켓 자체를
  서버에서 끊지는 못한다. 그래서 예산 알림(₩5,000, 50/90/100/150%)이 울리면 한 줄로 막는다:

```bash
bash cloudrun/deploy.sh off      # ingress internal: 공개 접근 차단, 서비스는 유지
bash cloudrun/deploy.sh on       # 복구
```

- Neon 무료 한도: 월 100 CU-hour(0.25 CU × 400시간 깨어 있는 시간), 저장 0.5 GB. 풀 `min_size=0`이라 방문자가 없으면 잠든다.

## 결정 로그에 적을 항목(평가 채팅에서)

- ST0-18 갱신: 공개 URL은 랜딩(demo-public, `demo/landing.py`), 대화형 데모는 Cloud Run.
- ST4-29 확장: 합성 승인 사례를 공개 데모에 포함하되 `SGP_DEMO_SYNTHETIC_PRESETS`로 프리셋 단위 표기(`kind=synthetic`, 매니페스트
  `synthetic: true`, 봉투 trace `synthetic: true`, 화면 배너). 테스트 DB 규칙(`SGP_DEMO_SYNTHETIC`)은 그대로.
- ST4-30 보완: 공개 DB의 Skill DSN 사용자는 `sgp_skill`(sgp_agent 상속, `sql/011`).
