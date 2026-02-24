#!/usr/bin/env python3
"""Benchmark naive scaled dot-product attention across sequence/model sizes.

This script benchmarks the assignment's attention implementation
(`cs336_basics.model.scaled_dot_product_attention`) with:
- fixed batch size (default 8),
- no head dimension (Q/K/V shaped [batch, seq_len, d_model]),
- cartesian product over d_model and seq_len grids,
- warmup + synchronized forward/backward timing,
- CUDA memory usage measurement before backward,
- OOM detection and reporting.
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from cs336_basics.model import scaled_dot_product_attention


MIB = 1024 * 1024


@dataclass
class BenchmarkResult:
    d_model: int
    seq_len: int
    status: str
    forward_ms: float | None
    backward_ms: float | None
    mem_before_backward_alloc_mib: float | None
    mem_before_backward_reserved_mib: float | None
    peak_alloc_mib: float | None
    peak_reserved_mib: float | None
    attn_matrix_mib: float
    saved_for_backward_est_mib: float
    saved_for_backward_upper_mib: float
    error: str = ""


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def bytes_per_element(dtype: torch.dtype) -> int:
    return torch.empty([], dtype=dtype).element_size()


def clear_cuda_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device=device)
    gc.collect()


def attention_call(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    # No explicit head dimension and no mask, as requested.
    return scaled_dot_product_attention(Q=q, K=k, V=v, mask=None)


def benchmark_one(
    *,
    batch_size: int,
    d_model: int,
    seq_len: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    device: torch.device,
) -> BenchmarkResult:
    bpe = bytes_per_element(dtype)
    attn_matrix_mib = (batch_size * seq_len * seq_len * bpe) / MIB
    saved_for_backward_est_mib = attn_matrix_mib
    saved_for_backward_upper_mib = 2.0 * attn_matrix_mib

    clear_cuda_memory(device)

    try:
        shape = (batch_size, seq_len, d_model)
        q = torch.randn(shape, device=device, dtype=dtype)
        k = torch.randn(shape, device=device, dtype=dtype)
        v = torch.randn(shape, device=device, dtype=dtype)

        # Forward warmup.
        for _ in range(warmup):
            _ = attention_call(q, k, v)
            sync_device(device)

        # Forward timing.
        sync_device(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = attention_call(q, k, v)
            sync_device(device)
        t1 = time.perf_counter()
        forward_ms = (t1 - t0) * 1000.0 / iters

        # Backward setup: use leaf tensors with grad.
        qg = q.detach().clone().requires_grad_(True)
        kg = k.detach().clone().requires_grad_(True)
        vg = v.detach().clone().requires_grad_(True)
        grad_out = torch.randn_like(qg)

        # Backward warmup (includes a forward for graph construction).
        for _ in range(warmup):
            out = attention_call(qg, kg, vg)
            sync_device(device)
            out.backward(grad_out)
            sync_device(device)
            qg.grad = None
            kg.grad = None
            vg.grad = None

        # Memory right before backward: after forward graph is created.
        out = attention_call(qg, kg, vg)
        sync_device(device)
        mem_before_alloc = torch.cuda.memory_allocated(device=device) / MIB
        mem_before_reserved = torch.cuda.memory_reserved(device=device) / MIB
        out.backward(grad_out)
        sync_device(device)
        qg.grad = None
        kg.grad = None
        vg.grad = None

        # Backward timing (timing backward only; each iter has its own fresh graph).
        backward_total = 0.0
        for _ in range(iters):
            out = attention_call(qg, kg, vg)
            sync_device(device)
            tb0 = time.perf_counter()
            out.backward(grad_out)
            sync_device(device)
            tb1 = time.perf_counter()
            backward_total += tb1 - tb0
            qg.grad = None
            kg.grad = None
            vg.grad = None

        backward_ms = backward_total * 1000.0 / iters
        peak_alloc = torch.cuda.max_memory_allocated(device=device) / MIB
        peak_reserved = torch.cuda.max_memory_reserved(device=device) / MIB

        return BenchmarkResult(
            d_model=d_model,
            seq_len=seq_len,
            status="ok",
            forward_ms=forward_ms,
            backward_ms=backward_ms,
            mem_before_backward_alloc_mib=mem_before_alloc,
            mem_before_backward_reserved_mib=mem_before_reserved,
            peak_alloc_mib=peak_alloc,
            peak_reserved_mib=peak_reserved,
            attn_matrix_mib=attn_matrix_mib,
            saved_for_backward_est_mib=saved_for_backward_est_mib,
            saved_for_backward_upper_mib=saved_for_backward_upper_mib,
        )
    except (torch.cuda.OutOfMemoryError, RuntimeError) as err:
        err_msg = str(err)
        if "out of memory" not in err_msg.lower() and not isinstance(err, torch.cuda.OutOfMemoryError):
            raise
        clear_cuda_memory(device)
        return BenchmarkResult(
            d_model=d_model,
            seq_len=seq_len,
            status="oom",
            forward_ms=None,
            backward_ms=None,
            mem_before_backward_alloc_mib=None,
            mem_before_backward_reserved_mib=None,
            peak_alloc_mib=None,
            peak_reserved_mib=None,
            attn_matrix_mib=attn_matrix_mib,
            saved_for_backward_est_mib=saved_for_backward_est_mib,
            saved_for_backward_upper_mib=saved_for_backward_upper_mib,
            error=err_msg.split("\n")[0][:200],
        )


def print_table(results: Iterable[BenchmarkResult]) -> None:
    print(
        "d_model,seq_len,status,forward_ms,backward_ms,"
        "mem_before_bwd_alloc_mib,mem_before_bwd_reserved_mib,"
        "peak_alloc_mib,peak_reserved_mib,attn_matrix_mib,"
        "saved_bwd_est_mib,saved_bwd_upper_mib,error"
    )
    print(
        "# 中文: 模型维度,序列长度,状态,前向耗时(ms),反向耗时(ms),"
        "反向前已分配显存(MiB),反向前已预留显存(MiB),峰值已分配显存(MiB),峰值已预留显存(MiB),"
        "注意力矩阵显存(MiB),反向保存显存估计(MiB),反向保存显存上界(MiB),错误信息"
    )
    for r in results:
        print(
            f"{r.d_model},{r.seq_len},{r.status},"
            f"{'' if r.forward_ms is None else f'{r.forward_ms:.3f}'},"
            f"{'' if r.backward_ms is None else f'{r.backward_ms:.3f}'},"
            f"{'' if r.mem_before_backward_alloc_mib is None else f'{r.mem_before_backward_alloc_mib:.1f}'},"
            f"{'' if r.mem_before_backward_reserved_mib is None else f'{r.mem_before_backward_reserved_mib:.1f}'},"
            f"{'' if r.peak_alloc_mib is None else f'{r.peak_alloc_mib:.1f}'},"
            f"{'' if r.peak_reserved_mib is None else f'{r.peak_reserved_mib:.1f}'},"
            f"{r.attn_matrix_mib:.1f},{r.saved_for_backward_est_mib:.1f},"
            f"{r.saved_for_backward_upper_mib:.1f},{r.error}"
        )


def maybe_write_csv(results: list[BenchmarkResult], path: str) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "d_model",
                "seq_len",
                "status",
                "forward_ms",
                "backward_ms",
                "mem_before_bwd_alloc_mib",
                "mem_before_bwd_reserved_mib",
                "peak_alloc_mib",
                "peak_reserved_mib",
                "attn_matrix_mib",
                "saved_bwd_est_mib",
                "saved_bwd_upper_mib",
                "error",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.d_model,
                    r.seq_len,
                    r.status,
                    r.forward_ms,
                    r.backward_ms,
                    r.mem_before_backward_alloc_mib,
                    r.mem_before_backward_reserved_mib,
                    r.peak_alloc_mib,
                    r.peak_reserved_mib,
                    r.attn_matrix_mib,
                    r.saved_for_backward_est_mib,
                    r.saved_for_backward_upper_mib,
                    r.error,
                ]
            )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--d-models", type=str, default="16,32,64,128")
    p.add_argument("--seq-lens", type=str, default="256,1024,4096,8192,16384")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    p.add_argument("--nvtx", action="store_true", help="Enable NVTX ranges (sets CS336_NVTX=1).")
    p.add_argument(
        "--result-file",
        type=str,
        default="",
    )
    args = p.parse_args()

    if args.nvtx:
        os.environ["CS336_NVTX"] = "1"

    if not args.result_file:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.result_file = f"nsys_reports/attention_benchmark_{ts}.csv"

    if args.batch_size != 8:
        print(f"[warning] assignment asks for batch size 8; got batch size {args.batch_size}")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("This benchmark is intended for CUDA to test OOM/memory behavior.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    dtype = dtype_from_name(args.dtype)
    d_models = parse_int_list(args.d_models)
    seq_lens = parse_int_list(args.seq_lens)

    print(
        f"Running attention benchmark on {torch.cuda.get_device_name(0)} "
        f"(dtype={dtype}, batch_size={args.batch_size}, warmup={args.warmup}, iters={args.iters})"
    )
    print(f"d_models={d_models}")
    print(f"seq_lens={seq_lens}")

    results: list[BenchmarkResult] = []
    for d_model in d_models:
        for seq_len in seq_lens:
            print(f"\n[case] d_model={d_model}, seq_len={seq_len}")
            res = benchmark_one(
                batch_size=args.batch_size,
                d_model=d_model,
                seq_len=seq_len,
                dtype=dtype,
                warmup=args.warmup,
                iters=args.iters,
                device=device,
            )
            if res.status == "ok":
                print(
                    "  ok "
                    f"forward={res.forward_ms:.3f} ms "
                    f"backward={res.backward_ms:.3f} ms "
                    f"mem_before_bwd_alloc={res.mem_before_backward_alloc_mib:.1f} MiB "
                    f"peak_alloc={res.peak_alloc_mib:.1f} MiB"
                )
            else:
                print(f"  oom ({res.error})")
            results.append(res)

    print("\n=== summary ===")
    print_table(results)

    first_oom = [r for r in results if r.status == "oom"]
    if first_oom:
        smallest = sorted(first_oom, key=lambda r: (r.seq_len, r.d_model))[0]
        print(
            "\nFirst OOM by (seq_len, d_model): "
            f"(seq_len={smallest.seq_len}, d_model={smallest.d_model})."
        )
        print(
            "At this size, attention matrix memory is about "
            f"{smallest.attn_matrix_mib:.1f} MiB per matrix; "
            f"saved-for-backward is roughly O(B*L^2) ({smallest.saved_for_backward_est_mib:.1f}-"
            f"{smallest.saved_for_backward_upper_mib:.1f} MiB range depending on what is saved)."
        )
    else:
        print("\nNo OOM in tested grid.")

    maybe_write_csv(results, args.result_file)
    if args.result_file:
        print(f"Saved results to {args.result_file}")


if __name__ == "__main__":
    main()
