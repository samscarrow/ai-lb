#!/usr/bin/env bash
set -euo pipefail
PY=${1:-python3}
if [ -z "${VENV_PATH:-}" ]; then
  VENV_HOME="${VENV_HOME:-$HOME/.venvs}"
  slug=$(basename "$PWD" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+|-+$//g')
  pyver=$($PY -c "import sys; print('%d.%d' % (sys.version_info.major, sys.version_info.minor))")
  VENV_PATH="${VENV_HOME}/${slug}-py${pyver}"
fi
mkdir -p "$(dirname "$VENV_PATH")"
[ -d "$VENV_PATH" ] || "$PY" -m venv "$VENV_PATH"
. "$VENV_PATH/bin/activate"
python -m pip install -U pip setuptools wheel || true
python - <<'PY' || true
import importlib.util, sys, subprocess
sys.exit(0) if importlib.util.find_spec("uv") else subprocess.call([sys.executable, "-m", "pip", "install", "-q", "uv"])
PY
# use the ACTIVE environment explicitly
if [ -f pyproject.toml ] && command -v uv >/dev/null 2>&1; then
  uv sync --active || true
fi
echo "Venv ready: $VENV_PATH"
