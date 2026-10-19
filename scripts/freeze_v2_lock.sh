#!/usr/bin/env bash
# Freeze requirements_v2_lock.txt from the active .venv (Python 3.11).
# 1) verifies every Layer-1 pin in constraints_v2.txt is installed at exactly that version
# 2) fills the Layer-2 lines (replaces ==FILL with the installed version)
# 3) writes requirements_v2_lock.full.txt (complete pip freeze) for transitive reproducibility
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY="${PY:-python}"
$PY -c 'import sys; assert sys.version_info[:2]==(3,11), sys.version' 
$PY -m pip freeze --exclude-editable > /tmp/freeze.txt

# 1) Layer-1 drift check
fail=0
while IFS= read -r line; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  pkg="${line%%==*}"; want="${line##*==}"
  have=$(grep -i "^${pkg}==" /tmp/freeze.txt | head -1 | cut -d= -f3 || true)
  if [[ "$have" != "$want" ]]; then echo "LAYER-1 DRIFT: $pkg want $want have '${have:-missing}'"; fail=1; fi
done < constraints_v2.txt
[[ $fail -eq 0 ]] || { echo "abort: fix .venv before freezing (pip install -c constraints_v2.txt ...)"; exit 1; }

# 2) fill Layer-2 pins
$PY - <<'PYEOF'
import re, pathlib
freeze = {}
for l in pathlib.Path('/tmp/freeze.txt').read_text().splitlines():
    if '==' in l and not l.startswith('#'):
        n, v = l.split('==', 1); freeze[n.lower().replace('_','-')] = v
src = pathlib.Path('requirements_v2_lock.txt').read_text().splitlines()
out, missing = [], []
for l in src:
    m = re.match(r'^([A-Za-z0-9_.\-]+)(\[[^\]]*\])?==FILL\s*$', l)
    if m:
        name = m.group(1).lower().replace('_','-')
        v = freeze.get(name)
        if v is None: missing.append(name); out.append(l); continue
        out.append(f"{m.group(1)}{m.group(2) or ''}=={v}")
    else:
        out.append(l)
pathlib.Path('requirements_v2_lock.txt').write_text('\n'.join(out) + '\n')
if missing: raise SystemExit(f"not installed in .venv: {missing}")
print("Layer-2 pins filled.")
PYEOF

# 3) full transitive freeze
{ echo "# full pip freeze of .venv at prereg-v1 — $(date -u +%F)"; cat /tmp/freeze.txt; } > requirements_v2_lock.full.txt
grep -c '==FILL' requirements_v2_lock.txt && { echo "abort: FILL markers remain"; exit 1; } || true
echo "done: requirements_v2_lock.txt + requirements_v2_lock.full.txt"
