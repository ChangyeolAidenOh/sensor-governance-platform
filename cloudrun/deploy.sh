#!/usr/bin/env bash
# cloudrun/deploy.sh: build and deploy the public demo to Cloud Run (demo-public branch).
#   bash cloudrun/deploy.sh            # build + deploy
#   bash cloudrun/deploy.sh secrets    # only (re)write the two DSN secrets from build/public/neon_roles.env
#   bash cloudrun/deploy.sh off | on   # kill switch: block / restore public ingress without deleting the service
# Settings (decision 2026-11-05): Tokyo (Tier 1 free quota), 1 vCPU / 1 GiB, max 1 instance, min 0, CPU only during
# requests, 30 min request timeout, session affinity for the Streamlit websocket. The admin DSN is never set here.
set -euo pipefail
PROJECT="${GCP_PROJECT:-process-reliability-agent}"
REGION="${GCP_REGION:-asia-southeast1}"
SERVICE="${CLOUDRUN_SERVICE:-process-reliability-agent}"
ROLES_ENV="build/public/neon_roles.env"
MODE="${1:-deploy}"

gcloud config set project "$PROJECT" >/dev/null

if [ "$MODE" = "off" ]; then
  gcloud run services update "$SERVICE" --region "$REGION" --ingress internal
  echo "public ingress blocked (service kept); restore with: bash cloudrun/deploy.sh on"; exit 0
fi
if [ "$MODE" = "on" ]; then
  gcloud run services update "$SERVICE" --region "$REGION" --ingress all
  echo "public ingress restored"; exit 0
fi

write_secret() {
  local name="$1" value="$2"
  if gcloud secrets describe "$name" >/dev/null 2>&1; then
    printf '%s' "$value" | gcloud secrets versions add "$name" --data-file=-
  else
    printf '%s' "$value" | gcloud secrets create "$name" --replication-policy=automatic --data-file=-
  fi
}

if [ "$MODE" = "secrets" ] || [ "$MODE" = "deploy" ]; then
  test -f "$ROLES_ENV" || { echo "$ROLES_ENV not found (run: python scripts/public_demo_build.py neon)"; exit 1; }
  AGENT_DSN="$(grep '^SGP_PUBLIC_NEON_AGENT_DSN=' "$ROLES_ENV" | cut -d= -f2-)"
  SKILL_DSN="$(grep '^SGP_PUBLIC_NEON_SKILL_DSN=' "$ROLES_ENV" | cut -d= -f2-)"
  case "$AGENT_DSN" in *pooler*) echo "agent DSN must be the direct Neon connection"; exit 1;; esac
  gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com >/dev/null
  write_secret sgp-public-agent-dsn "$AGENT_DSN"
  write_secret sgp-public-skill-dsn "$SKILL_DSN"
  PROJECT_NUMBER="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')"
  SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
  for s in sgp-public-agent-dsn sgp-public-skill-dsn; do
    gcloud secrets add-iam-policy-binding "$s" --member "serviceAccount:$SA" --role roles/secretmanager.secretAccessor >/dev/null
  done
  echo "secrets written and readable by $SA"
fi

if [ "$MODE" = "deploy" ]; then
  test -d cloudrun/assets/models/diagnoser_v1 || { echo "cloudrun/assets missing (run: bash cloudrun/prepare_image_assets.sh)"; exit 1; }
  gcloud run deploy "$SERVICE" --source . --region "$REGION" --platform managed --allow-unauthenticated \
    --cpu 1 --memory 1Gi --min-instances 0 --max-instances 1 --concurrency 20 --timeout 1800 --session-affinity --port 8080 \
    --set-secrets "SGP_AGENT_PG_DSN=sgp-public-agent-dsn:latest,SGP_PG_DSN=sgp-public-skill-dsn:latest" \
    --set-env-vars "SGP_PUBLIC_DEMO=1,SGP_DEMO_MANIFEST=outputs/demo_presets_public.json,SGP_DEMO_SYNTHETIC_PRESETS=11,FASTMCP_CHECK_FOR_UPDATES=off"
  gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)'
fi
