#!/usr/bin/env bash
# Linux x86_64: run local faster-whisper with pip-installed CUDA 12 and cuDNN 9.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install uv first." >&2
  exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >&2 || true
else
  echo "Warning: nvidia-smi unavailable; check your NVIDIA driver." >&2
fi

uv sync --extra mic --extra cuda

LIBRARY_DIRS="$(uv run --no-sync python - <<'PY'
from pathlib import Path
import site

roots = [Path(path) for path in site.getsitepackages()]
found = []
for package, soname in (("cublas", "libcublas.so.12"), ("cudnn", "libcudnn.so.9")):
    path = next(
        (
            root / "nvidia" / package / "lib"
            for root in roots
            if (root / "nvidia" / package / "lib" / soname).exists()
        ),
        None,
    )
    if path is None:
        raise SystemExit(f"Missing {soname} in uv environment; check cuda extras.")
    found.append(str(path))
print(":".join(found))
PY
)"

export LD_LIBRARY_PATH="${LIBRARY_DIRS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# Test dynamic linker access BEFORE starting the microphone.
uv run --no-sync python - <<'PY'
import ctypes
for name in ("libcublas.so.12", "libcudnn.so.9"):
    try:
        ctypes.CDLL(name)
    except OSError as exc:
        raise SystemExit(f"Cannot load {name}: {exc}")
print("NVIDIA CUDA 12 / cuDNN 9 libraries found")
PY

exec uv run --no-sync voice-rag-whisper --device cuda --compute-type float16 "$@"
