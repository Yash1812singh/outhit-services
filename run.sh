#!/usr/bin/env bash
# Start all five services (Ctrl+C stops them all). Gateway: http://localhost:8001
set -euo pipefail
cd "$(dirname "$0")"
[ -f .env ] || { echo "Missing .env - copy .env.example to .env and fill it in."; exit 1; }
exec .venv/bin/honcho start -e .env "$@"
