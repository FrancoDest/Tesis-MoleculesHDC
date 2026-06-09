#!/usr/bin/env bash
set -euo pipefail

if [ -d /tesis/Tesis-PolymerHDC/src/graphHD/original ]; then
  export PYTHONPATH="/tesis/Tesis-PolymerHDC/src/graphHD/original:/tesis/Tesis-PolymerHDC/src/graphHD/pipeline:${PYTHONPATH:-}"
elif [ -d /opt/graphHD/original ]; then
  export PYTHONPATH="/opt/graphHD/original:/opt/graphHD/pipeline:${PYTHONPATH:-}"
fi

exec "$@"
