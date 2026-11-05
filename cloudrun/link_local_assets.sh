#!/usr/bin/env bash
# cloudrun/link_local_assets.sh: give the demo-public worktree the untracked local files the build needs, as symlinks
# into the v2 working tree (model binaries, processed data, retrieval index). Symlinks are git-ignored there already.
#   cd ~/PycharmProjects/sensor_governance_platform_demo && bash cloudrun/link_local_assets.sh [../sensor_governance_platform]
set -euo pipefail
SRC="${1:-../sensor_governance_platform}"
SRC_ABS="$(cd "$SRC" && pwd)"
test -d "$SRC_ABS/data/processed/TEP" || { echo "no data/processed/TEP under $SRC_ABS"; exit 1; }
link() {
  local target="$1" dest="$2"
  if [ -e "$dest" ] && [ ! -L "$dest" ]; then echo "exists (not a link), leaving: $dest"; return; fi
  mkdir -p "$(dirname "$dest")"
  ln -sfn "$target" "$dest"
  echo "link $dest -> $target"
}
link "$SRC_ABS/data/processed" data/processed
link "$SRC_ABS/models/detector_v1" models/detector_v1
link "$SRC_ABS/models/diagnoser_v1" models/diagnoser_v1
link "$SRC_ABS/models/retrieval_v1/dense.npy" models/retrieval_v1/dense.npy
link "$SRC_ABS/models/retrieval_v1/chunks.jsonl" models/retrieval_v1/chunks.jsonl
if [ ! -e .env ]; then
  cp "$SRC_ABS/.env" .env
  echo "copied .env from $SRC_ABS (local DSNs, test DSN, SGP_PUBLIC_* keys); edit here if the demo needs different values"
fi
git status --porcelain --untracked-files=all | grep -v '^$' || true
