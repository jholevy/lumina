#!/usr/bin/env bash
# Lumina — Lanceur qui utilise le venv Python 3.11 avec Real-ESRGAN
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -d "$DIR/venv" ]; then
    source "$DIR/venv/bin/activate"
fi
exec python3 "$DIR/run.py" "$@"
