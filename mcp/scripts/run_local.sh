#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
: "${MOSSY_MCP_STATUS_KEY:?Set MOSSY_MCP_STATUS_KEY before starting the bridge}"

exec uvicorn app:app --reload --host 127.0.0.1 --port 8000
