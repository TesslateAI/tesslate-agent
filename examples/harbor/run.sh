#!/usr/bin/env bash
# Iteration helper: run the tesslate-studio harbor job with the local
# .env loaded. Any extra args are forwarded to `harbor run`, so you can
# do things like:
#
#   ./run.sh                              # use the YAML as-is
#   ./run.sh -i 'terminal-bench/fix-git'  # override the task filter
#   ./run.sh -m openai/gpt-4o-mini        # override the model
#   ./run.sh --ak max_iterations=10       # cap iterations from the CLI
#   ./run.sh -n 4                         # bump concurrency
#
# All of these compose with the YAML; CLI flags win on conflicts.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$HERE/tesslate-studio-job.yaml"
ENV_FILE="$HERE/.env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "error: $ENV_FILE not found." >&2
  echo "Copy .env.example to .env and fill in your credentials." >&2
  exit 1
fi

# Resolve harbor: prefer the harbor venv this repo was tested against,
# fall back to whatever is on PATH.
HARBOR_VENV="${TESSLATE_HARBOR_VENV:-$HOME/Tesslate-Studio/research/harbor/.venv}"
if [[ -x "$HARBOR_VENV/bin/harbor" ]]; then
  HARBOR="$HARBOR_VENV/bin/harbor"
else
  HARBOR="$(command -v harbor || true)"
fi
if [[ -z "$HARBOR" ]]; then
  echo "error: harbor CLI not found. Install with: uv tool install harbor" >&2
  exit 1
fi

exec "$HARBOR" run \
  --config "$CONFIG" \
  --env-file "$ENV_FILE" \
  -y \
  "$@"
