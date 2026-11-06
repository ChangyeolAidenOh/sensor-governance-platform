# 공개 데모 배포 (demo-public 브랜치)

- 랜딩: https://process-reliability-agent.streamlit.app (Community Cloud, `demo/landing.py`, GitHub About 링크). Secrets의
  `PUBLIC_DEMO_URL`이 "대화형 데모 열기" 버튼을 만든다.
- 대화형 데모: https://process-reliability-agent-834499990852.asia-southeast1.run.app (Cloud Run, 싱가포르). 프로젝트 번호와
  리전으로 정해지는 고정 주소다(해시 주소 `…-5nnw6zuwuq-as.a.run.app`도 같은 서비스).
- DB: Neon Free, 싱가포르, 데이터베이스 `sgp_v2`, Postgres 18. 데모 11 에피소드만 들어 있다(약 18 MB).
- 결정일 2026-11-05, 배포 완료 2026-11-06.

## 설계 요약

- 두 Skill 서버는 원래 "한 번 계산하고 저장한 결과를 재생"하는 구조다(detect: `mart.episode_windows`, retrieve: `mart.agent_steps`).
  공개 DB에는 데모 에피소드를 사전계산해 두고, 공개 앱은 저장된 행만 읽는다(`sql/006` 주석의 배포 설계 그대로).
- 공개 앱에서 바뀌는 것은 세 가지뿐이고 서버·검색 코드는 수정하지 않는다.
  1. `skills_detect` 서버만 서브프로세스로 띄운다(저장된 창 서빙 + 작업지시 초안·승인·발행 실행). `retrieve`는
     `core_pipeline/skills/public_skills.py`의 `StoredRetrieve`가 저장 행에서 답한다(bge 모델 없음). 저장되지 않은 검색은
     `REFUSED(DEMO_LIMIT)`이고 앱이 입력창 위에 안내를 띄운다.
  2. 연결 풀 `min_size=0`, 연결 검사 켬: Neon이 5분 유휴 뒤 잠들고 다음 접속에 자동으로 깬다.
  3. `SGP_DEMO_SYNTHETIC_PRESETS=11`: 프리셋 11만 합성 승인 사례(근거 등급 상향 + 정책 제안 고정, ST4-29 확장). 나머지 10개는 실제.
     합성 프리셋은 프리셋 1과 같은 결함 유형의 다른 run id 에피소드를 쓴다(같은 알람을 쓰면 합성 결정이 프리셋 1의 이력에 섞인다).
- 라벨 울타리: `window_labels`·`staging`은 공개 DB에 그대로 있지만 앱은 `sgp_agent`(앱)·`sgp_skill`(detect 서버, sgp_agent 상속,
  `sql/011`)로만 접속한다. 관리자 DSN은 Cloud Run secrets에 넣지 않는다.
- 사용 상한(`demo/public_guard.py`): 세션 15턴·20분, 하루 300턴·누적 90분(UTC). 기존 `mart.threads`·`graph_turns`로 집계. LLM 기본 off.
- 공개 모드 화면: 제목 "Process Reliability Agent Demo", 결정 ID와 비용 패널 숨김, 답변 전문을 본문에 표시(서술기 텍스트,
  봉투 아님), 추천 질문 버튼 7개(`outputs/public_demo_questions.json`).

## 측정값

- 로컬 Mac → Neon 싱가포르: 연결+쿼리 1회 약 800 ms. 한 턴이 30~50회 연결을 열므로 턴당 약 20초(평가 코드의 연결 방식 때문이며
  공개 데모에서는 손대지 않음).
- Cloud Run 싱가포르 → Neon: 턴당 5~6초, 콜드 스타트(컨테이너 + detect 서버 모델 로드) 약 20초.
- detect 서버 init 최대 RSS 367 MB(macOS), 앱 프로세스 약 200 MB. Cloud Run 1 GiB에서 동작.

## 로컬 환경 변수 규칙

`.env`의 표준 키(`SGP_PG_DSN`, `SGP_AGENT_PG_DSN`, `SGP_TEST_PG_DSN`, …)는 평가용 그대로 둔다. 공개 데모용은 접두사 `SGP_PUBLIC_`:

- `SGP_PUBLIC_NEON_ADMIN_DSN`: Neon 직접 연결(호스트에 `-pooler` 없음), DB `sgp_v2`
- `SGP_PUBLIC_NEON_AGENT_DSN`, `SGP_PUBLIC_NEON_SKILL_DSN`: `python scripts/public_demo_build.py neon`이 `build/public/neon_roles.env`에 써 준다

규칙: 평가 셸에서는 `SGP_PUBLIC_*`를 export하지 않는다. 로컬 빌드 DB 이름은 `sgp_public`이고, 모든 공개 스크립트는 dbname에
`eval`이 들어가거나 `sgp_v2`(로컬 프로젝트 DB)이면 거부한다. Cloud Run에는 표준 키 두 개(`SGP_AGENT_PG_DSN`, `SGP_PG_DSN`)만 있다.
`neon_roles.env`는 `source`하지 말고 `grep '^KEY=' file | cut -d= -f2-`로 읽는다(DSN의 `&`가 셸 연산자로 해석된다).

## worktree에서 작업할 때

- 파이썬·pytest 명령마다 `PYTHONPATH=.`을 붙인다. 공유 `.venv`의 editable 설치가 `core_pipeline`을 v2 트리로 보내고 `demo`는 경로에
  없기 때문이다. 빌드 스크립트는 서브프로세스에 `PYTHONPATH`를 루트로 넘긴다.
- demo-public에서는 `core_pipeline` 아래 **기존 파일을 수정하지 않고 새 파일만 추가**한다(경로를 빠뜨렸을 때 v2 코드가 조용히 대신
  도는 일을 막는 조건).
- `docker compose exec`는 디렉터리 이름으로 프로젝트를 찾아 worktree에서 실패한다. 스크립트는 `docker exec -i sgp_postgres`를 쓴다.
- `data/processed`는 추적 파일이 든 실제 디렉터리라 하위 경로(`TEP`, `gate_reference_v1.parquet`)만 절대 경로로 링크한다
  (`cloudrun/link_local_assets.sh`). parquet 없이도 `load_mart`는 "성공"하지만 `input_contract_sha256`가 달라지므로 load 출력의
  `808601386456…`과 `[warn] manifest not found` 부재를 확인한다.
- `precompute_demo.py --goldens`는 새 DB에서 알람 프리셋이 항상 DIFF다(창 행의 `alarm_id`가 DB마다 새로 발급되는 episode UUID를
  품는다). `version_diffs`가 `{}`인지만 보고 `--goldens ""`로 돌린다.

## 절차

worktree(`~/PycharmProjects/sensor_governance_platform_demo`, 공유 `.venv` 활성화)에서. v2 트리는 `../sensor_governance_platform`.

```bash
bash cloudrun/link_local_assets.sh
PYTHONPATH=. python scripts/public_demo_build.py local     # createdb → load → schema → roles → saver → verify → prune → precompute → manifest → prime → check → dump
PYTHONPATH=. python scripts/public_demo_build.py neon      # restore → roles(새 비밀번호) → check
bash cloudrun/prepare_image_assets.sh                      # cloudrun/assets에 모델 2종 + gate_reference_v1.{parquet,json} 복사
bash cloudrun/deploy.sh                                    # secrets + gcloud run deploy --source .
```

- `local`은 단계별로 재개할 수 있다(`--from prune`, `--only check`). `createdb`는 `sgp_public`이 있으면 멈추고 `--reset`으로 지우고 다시 만든다.
- `verify`는 전체 적재 상태에서 돌아야 한다(3b가 그룹 크기를 동결값과 비교). `prune`은 그다음이다.
- `prime`은 로컬 bge 모델을 로드한다(수 분). 실패하면 `build/public/prime_report.json`의 `failures`를 본다. 합성 프리셋이
  `waiting_approval`에 도달하지 못하면 `--synthetic-run-id`로 다른 run id를 골라 `--from prune`부터 다시 한다.
- `neon`은 DB에 이전 빌드가 있으면 멈추고, `--reset`이면 raw/staging/mart와 체크포인터 표를 지우고 복원한다(역할과 비밀번호는 유지).
- 재구축(모델·계약·정책·스키마가 바뀐 경우): `local --reset` → `neon --reset` → `prepare_image_assets` → `deploy`. Cloud Run secrets는 그대로.
- 역할 비밀번호는 `build/public/neon_roles.env`(커밋 안 됨)와 Secret Manager에만 있다. 잃으면 Neon SQL Editor에서
  `ALTER ROLE sgp_agent PASSWORD '...'`, `ALTER ROLE sgp_skill PASSWORD '...'`로 새로 정하고 파일을 고친 뒤 `bash cloudrun/deploy.sh secrets`.

## Cloud Run 설정과 GCP 전제

- 1 vCPU, 1 GiB, max 1 인스턴스, min 0, CPU는 요청 중에만, 타임아웃 30분, 세션 어피니티, `asia-southeast1`(Neon과 같은 리전; 도쿄는
  Tier 1이지만 연결당 지연 때문에 포기).
- 이미지: `ENV PYTHONPATH=/app`이 필수다. Skill 서버는 스크립트로 실행되어 `/app`이 경로에 없고, 없으면 `McpError: Connection closed`로
  서버 로그 한 줄 없이 죽는다. 빌드 시 import 검사는 `/tmp`에서 돌려 cwd가 가리지 않게 한다.
- `.dockerignore`·`.gcloudignore`의 디렉터리 패턴은 루트 고정(`/data`, `/build`)으로 쓰고 `!/cloudrun/assets/**`로 자산을 되살린다.
  슬래시 없는 `data`는 중첩된 같은 이름까지 삼킨다.
- 새 GCP 프로젝트의 기본 compute 서비스 계정에는 Editor가 없으므로 `roles/cloudbuild.builds.builder`와 `roles/storage.objectViewer`를
  먼저 준다(없으면 소스 업로드 403).

## 비용과 차단

- 무료 한도(결제 계정 합산, Tier 2): 월 128,571 vCPU-초 = 1 vCPU로 약 36시간의 "웹소켓이 열려 있는 시간". 리크루터 트래픽(월 수 시간)은 $0.
- 이론상 최악(누군가 탭을 한 달 내내 열어 둠)은 약 $84/월이고, 앱의 턴·시간 상한은 DB·compute 부하를 막을 뿐 Streamlit 웹소켓을
  서버에서 끊지는 못한다. 예산 알림(₩5,000, 50/90/100/150%)이 울리면 한 줄로 막는다:

```bash
bash cloudrun/deploy.sh off      # ingress internal: 공개 접근 차단, 서비스는 유지
bash cloudrun/deploy.sh on       # 복구
```

- Neon 무료 한도: 월 100 CU-hour(0.25 CU × 400시간 깨어 있는 시간), 저장 0.5 GB. 풀 `min_size=0`이라 방문자가 없으면 잠든다.
  첫 달에는 Neon 콘솔 Monitoring에서 compute 사용량을 한 번 확인한다.

## 랜딩 유지와 잠듦

Community Cloud는 12시간 트래픽이 없으면 앱을 재우고 방문자가 버튼을 눌러 깨워야 한다. 대응은 둘 중 하나를 실험 결과로 고른다.
1. UptimeRobot HTTP 모니터(5분 간격)를 걸고 24시간 뒤 잠들지 않았는지 본다. 깨어 있으면 끝.
2. 잠들어 있으면 `.github/workflows/keep_landing_awake.yml`(10시간마다 demo-public에 빈 커밋)을 main에도 올린다. 스케줄 워크플로는
   기본 브랜치에서만 돈다.

## 결정 로그에 적을 항목(평가 채팅에서)

- ST0-18 갱신: 공개 URL은 랜딩(demo-public, `demo/landing.py`), 대화형 데모는 Cloud Run 싱가포르 + Neon.
- ST4-29 확장: 합성 승인 사례를 공개 데모에 포함하되 `SGP_DEMO_SYNTHETIC_PRESETS`로 프리셋 단위 표기(`kind=synthetic`, 매니페스트
  `synthetic: true`, 봉투 trace `synthetic: true`, 화면 배너, 별도 에피소드). 테스트 DB 규칙(`SGP_DEMO_SYNTHETIC`)은 그대로.
- ST4-30 보완: 공개 DB의 Skill DSN 사용자는 `sgp_skill`(sgp_agent 상속, `sql/011`).
