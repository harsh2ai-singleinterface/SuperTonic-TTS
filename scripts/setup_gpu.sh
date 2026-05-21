#!/usr/bin/env bash
#
# One-shot GPU/CUDA setup for supertonic_server.
#
# Why this script exists:
#   `supertonic` depends on the CPU-only `onnxruntime` distribution. That
#   distribution ships the SAME `onnxruntime` import package as
#   `onnxruntime-gpu`, so whichever is installed last wins. Any plain
#   `pip install` / `uv pip install` that re-resolves `supertonic`'s
#   dependency tree reinstalls the CPU build and silently clobbers the GPU
#   one — after which the server logs an error and falls back to CPU.
#
#   This script (re)installs everything in the one order that leaves the
#   GPU build active: runtime + dev deps first, then onnxruntime-gpu LAST.
#   Run it after any dependency change to restore the GPU build.
#
# Usage:
#   ./scripts/setup_gpu.sh                      # auto-detects the venv
#   PY=/path/to/venv/bin/python ./scripts/setup_gpu.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- Resolve the target Python interpreter -------------------------------
PY="${PY:-}"
if [[ -z "$PY" ]]; then
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        PY="$VIRTUAL_ENV/bin/python"
    elif [[ -x "$REPO_ROOT/../tts/bin/python" ]]; then
        PY="$REPO_ROOT/../tts/bin/python"   # the project's `tts` venv
    else
        PY="$(command -v python3)"
    fi
fi
echo ">> interpreter: $PY"

if ! command -v uv >/dev/null 2>&1; then
    echo "error: 'uv' not found on PATH (expected ~/.local/bin/uv)" >&2
    exit 1
fi

# --- 1. Runtime + dev deps (this re-pulls the CPU onnxruntime) -----------
echo ">> installing runtime + dev dependencies"
uv pip install --python "$PY" \
    -r "$REPO_ROOT/requirements.txt" \
    -r "$REPO_ROOT/requirements-dev.txt"

# --- 2. Drop both onnxruntime builds for a clean slate -------------------
echo ">> removing any existing onnxruntime / onnxruntime-gpu"
uv pip uninstall --python "$PY" onnxruntime onnxruntime-gpu || true

# --- 3. Reinstall the GPU build LAST so it wins --------------------------
echo ">> installing onnxruntime-gpu + CUDA runtime wheels"
uv pip install --python "$PY" --reinstall -r "$REPO_ROOT/requirements-gpu.txt"

# --- 4. Verify CUDA is actually usable -----------------------------------
echo ">> verifying CUDA"
PYTHONPATH="$REPO_ROOT/src" "$PY" -c \
    "from supertonic_server.cuda import cuda_is_usable; \
     ok = cuda_is_usable(); \
     print('CUDA usable:', ok); \
     raise SystemExit(0 if ok else 1)"

echo ">> GPU setup complete."
