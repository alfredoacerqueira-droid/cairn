#!/usr/bin/env bash
# GPU benchmark harness: creates an isolated .venv-gpu and runs gpu_bench.py
# with full onnxruntime-gpu support. Never modifies the main env.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="$REPO_ROOT/.venv-gpu"

if [ ! -d "$VENV" ]; then
    echo "Creating isolated GPU venv at $VENV ..."
    python3 -m venv "$VENV"
fi

echo "Upgrading pip in .venv-gpu ..."
"$VENV/bin/pip" install --upgrade pip

echo "Installing cairn (editable) into .venv-gpu ..."
"$VENV/bin/pip" install -e "$REPO_ROOT"

echo "Installing fastembed-gpu (pulls onnxruntime-gpu) ..."
"$VENV/bin/pip" install fastembed-gpu

echo "Verifying onnxruntime GPU provider ..."
PROVIDER_CHECK=$("$VENV/bin/python" -c "
import onnxruntime as ort
ps = ort.get_available_providers()
print('PROVIDERS', ps)
import sys
sys.exit(0 if 'CUDAExecutionProvider' in ps else 3)
" 2>&1)
PROVIDER_EXIT=$?

echo "$PROVIDER_CHECK"

if [ $PROVIDER_EXIT -eq 3 ]; then
    echo ""
    echo "GPU provider NOT available in .venv-gpu (WSL CUDA libs likely missing);"
    echo "GPU benchmark cannot run — the 10-50x claim stays UNVERIFIED in this env."
    exit 3
fi

echo ""
echo "CUDA provider available. Running benchmark..."
exec "$VENV/bin/python" "$SCRIPT_DIR/gpu_bench.py" "$@"
