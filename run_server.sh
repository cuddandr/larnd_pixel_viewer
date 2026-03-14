#!/usr/bin/env bash

set -euo pipefail

# Defaults
APP="pixel_viewer:server"
HOST="127.0.0.1"
PORT=8050
WORKERS=4
RELOAD=false

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Run a Gunicorn server.

Options:
  -a <app>      WSGI app to serve (default: $APP)
  -h <host>     Bind host         (default: $HOST)
  -p <port>     Bind port         (default: $PORT)
  -w <workers>  Number of workers (default: $WORKERS)
  -r            Enable auto-reload
  --help        Show this help message

Examples:
  $(basename "$0")                       # run with all defaults
  $(basename "$0") -p 9000 -w 2 -r       # custom port, 2 workers, reload on
  $(basename "$0") -a myapp:app -p 8080  # different app entrypoint
EOF
  exit 0
}

# Handle --help manually before getopts (getopts doesn't support long flags)
for arg in "$@"; do
  [[ "$arg" == "--help" ]] && usage
done

while getopts ":a:h:p:w:r" opt; do
  case $opt in
    a) APP="$OPTARG" ;;
    h) HOST="$OPTARG" ;;
    p) PORT="$OPTARG" ;;
    w) WORKERS="$OPTARG" ;;
    r) RELOAD=true ;;
    :) echo "Error: option -$OPTARG requires an argument." >&2; exit 1 ;;
    \?) echo "Error: unknown option -$OPTARG" >&2; exit 1 ;;
  esac
done

# Build command
CMD="gunicorn $APP --bind $HOST:$PORT --workers $WORKERS"
$RELOAD && CMD="$CMD --reload"

echo "Starting: $CMD"
exec $CMD
