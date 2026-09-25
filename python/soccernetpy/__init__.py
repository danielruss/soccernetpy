"""Python bindings for SOCcerNET occupation and industry coding."""

import os
from pathlib import Path

_package_dir = Path(__file__).resolve().parent
for _ort_name in ("libonnxruntime.so", "libonnxruntime.1.24.2.dylib"):
    _bundled_ort = _package_dir / _ort_name
    if _bundled_ort.is_file():
        os.environ.setdefault("ORT_DYLIB_PATH", str(_bundled_ort))
        break

from .soccernetpy import *
