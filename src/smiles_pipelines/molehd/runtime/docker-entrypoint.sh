#!/bin/sh
set -e

if [ -f /tesis/original/MoleHD.py ]; then
    echo "Usando codigo de MoleHD montado desde /tesis..."
    export PYTHONPATH="/tesis/original${PYTHONPATH:+:$PYTHONPATH}"
fi

mkdir -p /tesis/outputs
mkdir -p /tesis/models

exec "$@"
