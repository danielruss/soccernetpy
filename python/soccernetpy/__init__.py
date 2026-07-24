"""Python bindings for SOCcerNET occupation and industry coding."""

import os
from pathlib import Path

_bundled_ort = Path(__file__).resolve().parent / "libonnxruntime.so"
if _bundled_ort.is_file():
    os.environ.setdefault("ORT_DYLIB_PATH", str(_bundled_ort))

from .soccernetpy import *
