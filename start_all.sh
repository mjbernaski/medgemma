#!/usr/bin/env bash
# Start both vLLM server and Gradio UI
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.env" 2>/dev/null || true

VLLM_PORT="${VLLM_PORT:-8000}"

echo "=== Starting MedGemma Services ==="

# Start vLLM server
echo ""
echo "--- Step 1: vLLM Server ---"
bash "$SCRIPT_DIR/start_vllm.sh"

# Verify vLLM is responding before starting UI
if ! curl -sf "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
    echo "WARNING: vLLM server may not be fully ready yet."
    echo "Starting Gradio UI anyway — it will retry connections automatically."
fi

# Start Gradio UI
echo ""
echo "--- Step 2: Gradio UI ---"
bash "$SCRIPT_DIR/start_ui.sh"

echo ""
echo "=== All services started ==="
echo "  API:    http://localhost:$VLLM_PORT"
echo "  Web UI: http://localhost:${GRADIO_PORT:-7860}"
