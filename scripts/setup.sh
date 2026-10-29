#!/usr/bin/env bash
# Download OpenWebText and tokenize it with GPT-2 (see scripts/prepare_data.py).
#
#   scripts/setup.sh                 # on Modal: writes to the Volume at /vol/data/openwebtext
#   scripts/setup.sh modal 8         #   ... with 8 shards (~0.9B tokens) instead of the default 4
#   scripts/setup.sh local 1         # locally: writes to ./data/openwebtext (1 shard, for dev)
#
# Training runs on Modal, so "modal" (the default) is the one you want for real runs:
# the files land on the persistent Volume, next to the HF cache, and every GPU function
# can read them at /vol/data/openwebtext. Each shard is ~110M tokens (~220 MB as uint16).
set -euo pipefail

cd "$(dirname "$0")/.."

MODE="${1:-modal}"
SHARDS="${2:-4}"

case "$MODE" in
  modal)
    uv sync --group data
    uv run modal run infra/modal_app.py::prepare_data --shards "$SHARDS"
    ;;
  local)
    uv sync --group data
    uv run python scripts/prepare_data.py --out-dir data/openwebtext --shards "$SHARDS"
    ;;
  *)
    echo "usage: $0 [modal|local] [num_shards]" >&2
    exit 1
    ;;
esac
