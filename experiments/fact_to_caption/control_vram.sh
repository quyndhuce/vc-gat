#!/bin/bash
# =============================================================================
# control_vram.sh
# =============================================================================
# Helper script to control the 90% GPU VRAM Holder
# Usage:
#   ./control_vram.sh status   -> Shows current GPU usage and holder process
#   ./control_vram.sh pause    -> Instantly releases held VRAM (0 MB held)
#   ./control_vram.sh resume   -> Resumes holding 90% VRAM
#   ./control_vram.sh stop     -> Terminates the holder daemon
#   ./control_vram.sh start    -> Starts the holder daemon in background
# =============================================================================

TRIGGER_FILE="/tmp/release_vram"
PYTHON="/home/nguyendinhquy/miniconda3/envs/qwen_graph/bin/python"
HOLDER_DIR="/home/nguyendinhquy/graph-tourism/fuzzy_graph"

case "$1" in
    pause|release|free)
        echo "Releasing held VRAM..."
        touch "$TRIGGER_FILE"
        sleep 1
        nvidia-smi
        echo "VRAM released! (Trigger file set at $TRIGGER_FILE)"
        ;;
    resume|hold)
        echo "Resuming 90% VRAM holding..."
        rm -f "$TRIGGER_FILE"
        sleep 2
        nvidia-smi
        echo "VRAM holding active!"
        ;;
    status)
        echo "--- GPU VRAM Status ---"
        nvidia-smi
        echo ""
        echo "--- Holder Process Status ---"
        ps aux | grep "app.py" | grep -v grep || echo "No app.py running."
        if [ -f "$TRIGGER_FILE" ]; then
            echo "Status: PAUSED (Release file present at $TRIGGER_FILE)"
        else
            echo "Status: ACTIVE (Holding 90% VRAM)"
        fi
        ;;
    stop|kill)
        echo "Stopping VRAM holder daemon..."
        pkill -9 -f "fuzzy_graph/app.py" || true
        pkill -9 -f "python app.py" || true
        sleep 1
        nvidia-smi
        echo "Holder daemon stopped."
        ;;
    start)
        echo "Starting 90% VRAM holder daemon..."
        rm -f "$TRIGGER_FILE"
        cd "$HOLDER_DIR"
        nohup "$PYTHON" app.py > app.log 2>&1 &
        sleep 2
        nvidia-smi
        echo "Holder daemon running."
        ;;
    *)
        echo "Usage: $0 {status|pause|resume|start|stop}"
        exit 1
        ;;
esac
