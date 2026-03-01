#!/usr/bin/env bash
# Start the Gradio web UI for MedGemma
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.env" 2>/dev/null || true

PORT="${GRADIO_PORT:-7860}"
PID_FILE="$SCRIPT_DIR/gradio.pid"
LOG_FILE="$SCRIPT_DIR/gradio.log"

# Check if already running
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Gradio UI is already running (PID $(cat "$PID_FILE"))"
    exit 0
fi

# Activate venv if it exists
if [[ -d "$SCRIPT_DIR/.venv" ]]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

echo "Starting Gradio UI on port $PORT"
echo "Log file: $LOG_FILE"

python3 "$SCRIPT_DIR/app.py" > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
echo "Gradio UI started (PID $(cat "$PID_FILE"))"

# Wait briefly for it to come up
for i in $(seq 1 12); do
    if curl -sf "http://localhost:$PORT" > /dev/null 2>&1; then
        echo "Gradio UI is ready at http://localhost:$PORT"
        exit 0
    fi
    sleep 2
done

echo "Gradio UI may still be starting. Check: http://localhost:$PORT"
echo "Logs: tail -f $LOG_FILE"
