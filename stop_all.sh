#!/usr/bin/env bash
# Stop both vLLM server and Gradio UI
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

stop_service() {
    local name="$1"
    local pid_file="$2"

    if [[ -f "$pid_file" ]]; then
        local pid
        pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            echo "Stopping $name (PID $pid)..."
            kill "$pid"
            # Wait up to 15 seconds for graceful shutdown
            for i in $(seq 1 30); do
                if ! kill -0 "$pid" 2>/dev/null; then
                    echo "$name stopped."
                    rm -f "$pid_file"
                    return 0
                fi
                sleep 0.5
            done
            echo "$name did not stop gracefully, sending SIGKILL..."
            kill -9 "$pid" 2>/dev/null || true
            echo "$name killed."
        else
            echo "$name is not running (stale PID file)."
        fi
        rm -f "$pid_file"
    else
        echo "$name is not running (no PID file)."
    fi
}

echo "=== Stopping MedGemma Services ==="
stop_service "Gradio UI" "$SCRIPT_DIR/gradio.pid"
stop_service "vLLM server" "$SCRIPT_DIR/vllm.pid"
echo "=== All services stopped ==="
