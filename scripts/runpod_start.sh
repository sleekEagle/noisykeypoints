#!/bin/bash
# Start BKinD training on a RunPod pod in the background (survives closing the browser / SSH).
#   bash scripts/runpod_start.sh /workspace/runs.py [--stop-pod] [--max-hours 100] ...
# Pulls the latest code first. The log goes to /workspace/logs/<date>.log; follow it with the
# `tail -f` command this prints. Arguments are passed on to scripts/runpod_bkind.py.
set -e
cd "$(dirname "$0")/.."
git pull -q
mkdir -p /workspace/logs
# date + runs-file name, so two launchers started in the same minute get separate logs
LOG=/workspace/logs/$(date +%Y-%m-%d_%H%M%S)_$(basename "$1" .py).log
PYTHONUNBUFFERED=1 nohup python -u scripts/runpod_bkind.py "$@" > "$LOG" 2>&1 &
echo "started (process $!). Follow it with:"
echo "  tail -f $LOG"
