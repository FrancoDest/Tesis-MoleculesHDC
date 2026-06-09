#!/usr/bin/env bash
set -euo pipefail

if [ -d /tesis/final/src/graphHD/original ]; then
  export PYTHONPATH="/tesis/final/src/graphHD/original:/tesis/final/src/graphHD/pipeline:${PYTHONPATH:-}"
elif [ -d /opt/graphHD/original ]; then
  export PYTHONPATH="/opt/graphHD/original:/opt/graphHD/pipeline:${PYTHONPATH:-}"
fi

exec "$@"
