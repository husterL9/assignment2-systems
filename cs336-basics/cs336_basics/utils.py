from __future__ import annotations

import contextlib
import os

import torch


def _env_nvtx_enabled() -> bool:
    flag = os.getenv("CS336_NVTX", "")
    return flag not in ("", "0", "false", "False", "no", "NO")


def nvtx_is_enabled(enabled: bool | None = None) -> bool:
    if enabled is None:
        enabled = _env_nvtx_enabled()
    return bool(enabled) and torch.cuda.is_available() and hasattr(torch.cuda, "nvtx")


@contextlib.contextmanager
def nvtx_range(message: str, enabled: bool | None = None):
    do_nvtx = nvtx_is_enabled(enabled)
    if do_nvtx:
        torch.cuda.nvtx.range_push(message)
    try:
        yield
    finally:
        if do_nvtx:
            torch.cuda.nvtx.range_pop()
