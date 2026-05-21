"""GPU runtime-library bootstrap for ONNX Runtime.

``onnxruntime-gpu``'s CUDA execution provider ``dlopen()``s the CUDA 12 and
cuDNN 9 shared libraries when an inference session is created. When those
libraries come from the ``nvidia-*-cu12`` pip wheels (rather than a system
CUDA install) they live under ``site-packages/nvidia/<component>/lib/`` — a
directory the dynamic linker does not search by default, so the provider
fails to load and ONNX Runtime silently falls back to CPU.

:func:`preload_cuda_libraries` loads those libraries into the process with
``RTLD_GLOBAL`` so their symbols are already resolved by the time ONNX
Runtime loads ``libonnxruntime_providers_cuda.so``. This avoids having to
set ``LD_LIBRARY_PATH`` before the interpreter starts.

Both functions here are no-ops / return ``False`` when the wheels aren't
installed (e.g. on a CPU-only box), so importing this module is always safe.
"""

from __future__ import annotations

import ctypes
import glob
import logging
import os
import site

logger = logging.getLogger("supertonic_server.cuda")

# Preload order matters: cuDNN depends on cuBLAS, cuBLAS on the CUDA runtime.
_COMPONENT_ORDER = (
    "cuda_runtime",
    "cuda_nvrtc",
    "cublas",
    "cufft",
    "curand",
    "cudnn",
)

_preloaded: bool | None = None
_cuda_usable: bool | None = None


def _nvidia_lib_roots() -> list[str]:
    """Every ``site-packages/nvidia`` directory on the current path."""
    roots: list[str] = []
    candidates = set(site.getsitepackages())
    candidates.add(site.getusersitepackages())
    for sp in candidates:
        nvidia = os.path.join(sp, "nvidia")
        if os.path.isdir(nvidia):
            roots.append(nvidia)
    return roots


def preload_cuda_libraries() -> bool:
    """Preload pip-installed CUDA/cuDNN libraries into this process.

    Idempotent: the first call does the work, later calls return the cached
    result without re-loading.

    Returns:
        True if at least one CUDA library was loaded, False if no
        ``nvidia-*-cu12`` wheels were found.
    """
    global _preloaded
    if _preloaded is not None:
        return _preloaded

    loaded = 0
    for nvidia in _nvidia_lib_roots():
        for component in _COMPONENT_ORDER:
            libdir = os.path.join(nvidia, component, "lib")
            if not os.path.isdir(libdir):
                continue
            for so in sorted(glob.glob(os.path.join(libdir, "*.so*"))):
                try:
                    ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                    loaded += 1
                except OSError as exc:  # missing transitive dep — skip, keep going
                    logger.debug("skipped %s: %s", so, exc)

    _preloaded = loaded > 0
    if _preloaded:
        logger.info("preloaded %d CUDA runtime libraries from pip wheels", loaded)
    else:
        logger.debug("no pip-installed CUDA libraries found")
    return _preloaded


def cuda_is_usable() -> bool:
    """Return True only if ONNX Runtime can actually load its CUDA provider.

    ``onnxruntime.get_available_providers()`` lists ``CUDAExecutionProvider``
    even when its shared library can't be loaded, so it is not a reliable
    check. This probes the real thing: after preloading the CUDA runtime, it
    ``dlopen()``s ``libonnxruntime_providers_cuda.so`` — the exact library
    ONNX Runtime loads internally. If that succeeds, CUDA inference works.

    The result is cached after the first call.
    """
    global _cuda_usable
    if _cuda_usable is not None:
        return _cuda_usable

    _cuda_usable = False
    try:
        import onnxruntime as ort
    except ImportError:
        return _cuda_usable

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        return _cuda_usable  # CPU-only onnxruntime build

    preload_cuda_libraries()
    provider_so = os.path.join(
        os.path.dirname(ort.__file__), "capi", "libonnxruntime_providers_cuda.so"
    )
    try:
        ctypes.CDLL(provider_so, mode=ctypes.RTLD_GLOBAL)
        _cuda_usable = True
    except OSError as exc:
        logger.warning("CUDA provider unavailable: %s", exc)

    return _cuda_usable
