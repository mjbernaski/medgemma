#!/usr/bin/env bash
# Start the vLLM server for MedGemma 27B
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.env" 2>/dev/null || true

MODEL="${MODEL_NAME:-google/medgemma-27b-it}"
PORT="${VLLM_PORT:-8000}"
LOG_FILE="$SCRIPT_DIR/vllm.log"
PID_FILE="$SCRIPT_DIR/vllm.pid"

# Check if already running
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "vLLM server is already running (PID $(cat "$PID_FILE"))"
    exit 0
fi

echo "Starting vLLM server with model: $MODEL on port $PORT"
echo "Log file: $LOG_FILE"

# Activate venv if it exists
if [[ -d "$SCRIPT_DIR/.venv" ]]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --max-num-seqs 4 \
    --gpu-memory-utilization 0.90 \
    --dtype bfloat16 \
    --limit-mm-per-prompt image=1 \
    --trust-remote-code \
    > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
echo "vLLM server started (PID $(cat "$PID_FILE"))"
echo "Waiting for server to be ready..."

# Wait for the server to respond (up to 10 minutes for model loading)
for i in $(seq 1 120); do
    if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "vLLM server is ready on port $PORT"
        exit 0
    fi
    # Check if process is still alive
    if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "ERROR: vLLM server process died. Check $LOG_FILE"
        rm -f "$PID_FILE"
        exit 1
    fi
    sleep 5
done

echo "WARNING: Server did not become ready within 10 minutes."
echo "It may still be loading. Check: curl http://localhost:$PORT/health"
echo "Logs: tail -f $LOG_FILE"
