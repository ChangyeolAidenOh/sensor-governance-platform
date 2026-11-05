#!/usr/bin/env bash
# cloudrun/prepare_image_assets.sh: copy (not link) the files the Dockerfile needs into cloudrun/assets, which is
# git-ignored and listed in neither .dockerignore nor .gcloudignore, so the build context carries them.
#   bash cloudrun/prepare_image_assets.sh [../sensor_governance_platform]
set -euo pipefail
SRC="${1:-../sensor_governance_platform}"
SRC_ABS="$(cd "$SRC" && pwd)"
DST=cloudrun/assets
rm -rf "$DST"
mkdir -p "$DST/models" "$DST/data/processed"
cp -R "$SRC_ABS/models/detector_v1" "$DST/models/detector_v1"
cp -R "$SRC_ABS/models/diagnoser_v1" "$DST/models/diagnoser_v1"
cp "$SRC_ABS"/data/processed/gate_reference_v1.* "$DST/data/processed/"
ls -la "$DST/data/processed/"
find "$DST" -name '.DS_Store' -delete
du -sh "$DST"
test -f outputs/demo_presets_public.json || { echo "outputs/demo_presets_public.json missing: run the local build first"; exit 1; }
echo "assets ready"
