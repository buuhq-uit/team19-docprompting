#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  Team 19 - DocPrompting API Server Launcher
# ═══════════════════════════════════════════════════════════════
#
#  Sử dụng:
#    bash start_api.sh              # GPU mode, port 8000
#    bash start_api.sh --cpu        # CPU mode
#    bash start_api.sh --port 9000  # Port tùy chỉnh
#
# ═══════════════════════════════════════════════════════════════

set -e

# Mặc định
HOST="0.0.0.0"
PORT=8000
USE_CPU=""
EXTRA_ARGS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu)
            USE_CPU="1"
            export USE_CPU=1
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Detect working directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Team 19 - DocPrompting API Server                      ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "  📂 Working directory: $SCRIPT_DIR"
echo "  🌐 Host: $HOST"
echo "  🔌 Port: $PORT"
echo "  🖥️  CPU mode: ${USE_CPU:-"No (GPU)"}"
echo ""

# Set PYTHONPATH
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Check dependencies
echo "📦 Checking dependencies..."
pip install fastapi uvicorn --quiet 2>/dev/null || true

echo ""
echo "🚀 Starting API server..."
echo "   Swagger UI: http://$HOST:$PORT/docs"
echo "   Health:     http://$HOST:$PORT/health"
echo ""

# Start uvicorn
if [ -n "$USE_CPU" ]; then
    PYTHONPATH="$SCRIPT_DIR" python team19_generator_pipeline_api.py --host "$HOST" --port "$PORT" --cpu $EXTRA_ARGS
else
    PYTHONPATH="$SCRIPT_DIR" python team19_generator_pipeline_api.py --host "$HOST" --port "$PORT" $EXTRA_ARGS
fi
