#!/bin/sh
set -e

if [ -f /tesis/original/features.py ]; then
    echo "Usando codigo de mol2vec montado desde /tesis..."
    export PYTHONPATH="/tesis/original${PYTHONPATH:+:$PYTHONPATH}"
fi

exec "$@"
