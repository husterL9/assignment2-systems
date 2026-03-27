from __future__ import annotations

import argparse
import csv
import math
import multiprocessing as py_mp
import queue
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import pandas as pd
import torch
import torch.distributed as dist
from torch.multiprocessing.spawn import ProcessExitedException, ProcessRaisedException, spawn

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_BACKENDS = ("gloo", "nccl")
DEFAULT_WORLD_SIZES = (2, 4, 6)
DEFAULT_SIZES_MB = (1, 10, 100, 1024)


@dataclass(frozen=True)
class BenchmarkSpec:
    backend: str
    world_size: int
    size_mb: int
    tensor_bytes: int
    numel: int
    warmup_iters: int
    min_iters: int
    max_iters: int
    target_runtime_s: float
    verify: bool
    master_addr: str
    master_port: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark single-node all-reduce latency and bandwidth.")
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=sorted(DEFAULT_BACKENDS),
        default=list(DEFAULT_BACKENDS),
        help="Distributed backends to benchmark.",
    )
    parser.add_argument(
        "--world-sizes",
        nargs="+",
        type=int,
        default=list(DEFAULT_WORLD_SIZES),
        help="Number of processes to spawn for each run.",
    )
    parser.add_argument(
        "--sizes-mb",
        nargs="+",
        type=int,
        default=list(DEFAULT_SIZES_MB),
        help="Tensor sizes in MiB for float32 tensors.",
    )
    parser.add_argument("--warmup-iters", type=int, default=2, help="Warmup all-reduce iterations per run.")
    parser.add_argument("--min-iters", type=int, default=3, help="Minimum timed iterations per run.")
    parser.add_argument("--max-iters", type=int,  default=20, help="Maximum timed iterations per run.")
    parser.add_argument(
        "--target-runtime-s",
        type=float,
        default=1.5,
        help="Approximate timed duration budget per configuration.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results") / "all_reduce",
        help="Directory for CSV tables, plots, and the markdown report.",
    )
    parser.add_argument(
        "--master-addr",
        type=str,
        default="127.0.0.1",
        help="Address used to initialize the process group.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip plot generation and only write CSV and markdown summary files.",
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Skip markdown report generation.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the one-shot correctness check after warmup.",
    )
    return parser.parse_args()


def mib_to_bytes(size_mb: int) -> int:
    return size_mb * 1024 * 1024


def size_label(size_mb: int) -> str:
    if size_mb % 1024 == 0:
        return f"{size_mb // 1024}GB"
    return f"{size_mb}MB"


def backend_label(backend: str) -> str:
    if backend == "gloo":
        return "Gloo + CPU"
    if backend == "nccl":
        return "NCCL + GPU"
    return backend


def status_row(spec: BenchmarkSpec, status: str, error: str = "") -> dict[str, Any]:
    device_type = "gpu" if spec.backend == "nccl" else "cpu"
    return {
        "backend": spec.backend,
        "backend_label": backend_label(spec.backend),
        "device_type": device_type,
        "world_size": spec.world_size,
        "size_mb": spec.size_mb,
        "size_label": size_label(spec.size_mb),
        "tensor_bytes": spec.tensor_bytes,
        "numel": spec.numel,
        "warmup_iters": spec.warmup_iters,
        "timed_iters": 0,
        "calibration_ms": math.nan,
        "latency_ms": math.nan,
        "rank_mean_latency_ms_min": math.nan,
        "rank_mean_latency_ms_median": math.nan,
        "rank_mean_latency_ms_max": math.nan,
        "rank_mean_spread_pct": math.nan,
        "rank_iteration_min_ms": math.nan,
        "rank_iteration_max_ms": math.nan,
        "effective_bandwidth_GBps": math.nan,
        "device_name": "",
        "status": status,
        "error": error,
    }


def is_oom_error(error: BaseException) -> bool:
    message = str(error).lower()
    return isinstance(error, torch.OutOfMemoryError) or "out of memory" in message


def pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def barrier(backend: str, device_index: int | None) -> None:
    if backend == "nccl" and device_index is not None:
        dist.barrier(device_ids=[device_index])
        return
    dist.barrier()


def choose_iters(calibration_s: float, min_iters: int, max_iters: int, target_runtime_s: float) -> int:
    if calibration_s <= 0:
        return max(min_iters, 1)
    target_iters = math.ceil(target_runtime_s / calibration_s)
    return max(min_iters, min(max_iters, target_iters))


def prepare_device(rank: int, spec: BenchmarkSpec) -> tuple[torch.device, str, int | None]:
    if spec.backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL benchmark requires CUDA, but no GPU is visible.")
        device_count = torch.cuda.device_count()
        if spec.world_size > device_count:
            raise RuntimeError(
                f"NCCL benchmark requested {spec.world_size} processes, but only {device_count} GPU(s) are visible."
            )
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        return device, torch.cuda.get_device_name(rank), rank
    return torch.device("cpu"), "CPU", None


def benchmark_worker(rank: int, spec: BenchmarkSpec, result_queue: py_mp.Queue) -> None:
    initialized = False
    completed_successfully = False
    device_index: int | None = None
    try:
        device, device_name, device_index = prepare_device(rank, spec)
        dist.init_process_group(
            backend=spec.backend,
            init_method=f"tcp://{spec.master_addr}:{spec.master_port}",
            rank=rank,
            world_size=spec.world_size,
        )
        initialized = True

        tensor = torch.empty(spec.numel, dtype=torch.float32, device=device)
        expected_sum = spec.world_size * (spec.world_size + 1) / 2.0

        for _ in range(spec.warmup_iters):
            tensor.fill_(rank + 1)
            sync_device(device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False)
            sync_device(device)

        if spec.verify:
            observed = float(tensor[0].item())
            if not math.isclose(observed, expected_sum, rel_tol=1e-5, abs_tol=1e-5):
                raise RuntimeError(
                    f"Verification failed for {backend_label(spec.backend)} with world_size={spec.world_size}: "
                    f"expected {expected_sum}, got {observed}."
                )

        tensor.fill_(rank + 1)
        sync_device(device)
        barrier(spec.backend, device_index)
        calibration_start = time.perf_counter()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False)
        sync_device(device)
        calibration_s = time.perf_counter() - calibration_start

        timed_iters = choose_iters(
            calibration_s=calibration_s,
            min_iters=spec.min_iters,
            max_iters=spec.max_iters,
            target_runtime_s=spec.target_runtime_s,
        )

        local_sum_s = 0.0
        local_min_s = float("inf")
        local_max_s = 0.0

        barrier(spec.backend, device_index)
        for _ in range(timed_iters):
            tensor.fill_(rank + 1)
            sync_device(device)
            iter_start = time.perf_counter()
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False)
            sync_device(device)
            iteration_s = time.perf_counter() - iter_start
            local_sum_s += iteration_s
            local_min_s = min(local_min_s, iteration_s)
            local_max_s = max(local_max_s, iteration_s)

        metrics = torch.tensor(
            [local_sum_s, local_min_s, local_max_s, calibration_s],
            dtype=torch.float64,
            device=device,
        )
        gathered = [torch.zeros_like(metrics) for _ in range(spec.world_size)]
        dist.all_gather(gathered, metrics)

        if rank == 0:
            rank_total_latencies_s = sorted(item[0].item() / timed_iters for item in gathered)
            rank_min_latencies_s = [item[1].item() for item in gathered]
            rank_max_latencies_s = [item[2].item() for item in gathered]
            calibration_values_s = [item[3].item() for item in gathered]
            latency_s = max(rank_total_latencies_s)
            spread_pct = 0.0
            if latency_s > 0:
                spread_pct = 100.0 * (max(rank_total_latencies_s) - min(rank_total_latencies_s)) / latency_s
            effective_bandwidth_gbps = (
                (2.0 * (spec.world_size - 1) / spec.world_size) * spec.tensor_bytes / latency_s / 1e9
            )

            result_queue.put(
                {
                    "backend": spec.backend,
                    "backend_label": backend_label(spec.backend),
                    "device_type": "gpu" if spec.backend == "nccl" else "cpu",
                    "world_size": spec.world_size,
                    "size_mb": spec.size_mb,
                    "size_label": size_label(spec.size_mb),
                    "tensor_bytes": spec.tensor_bytes,
                    "numel": spec.numel,
                    "warmup_iters": spec.warmup_iters,
                    "timed_iters": timed_iters,
                    "calibration_ms": 1000.0 * max(calibration_values_s),
                    "latency_ms": 1000.0 * latency_s,
                    "rank_mean_latency_ms_min": 1000.0 * min(rank_total_latencies_s),
                    "rank_mean_latency_ms_median": 1000.0 * rank_total_latencies_s[len(rank_total_latencies_s) // 2],
                    "rank_mean_latency_ms_max": 1000.0 * max(rank_total_latencies_s),
                    "rank_mean_spread_pct": spread_pct,
                    "effective_bandwidth_GBps": effective_bandwidth_gbps,
                    "rank_iteration_min_ms": 1000.0 * min(rank_min_latencies_s),
                    "rank_iteration_max_ms": 1000.0 * max(rank_max_latencies_s),
                    "device_name": device_name,
                    "status": "ok",
                    "error": "",
                }
            )
        completed_successfully = True
    finally:
        if initialized and dist.is_initialized():
            if completed_successfully:
                try:
                    barrier(spec.backend, device_index)
                except Exception:
                    pass
            try:
                dist.destroy_process_group()
            except Exception:
                pass


def run_benchmark(spec: BenchmarkSpec) -> dict[str, Any]:
    if spec.backend == "gloo" and not dist.is_gloo_available():
        return status_row(spec, status="skipped", error="This PyTorch build does not provide Gloo.")

    if spec.backend == "nccl":
        if not dist.is_nccl_available():
            return status_row(spec, status="skipped", error="This PyTorch build does not provide NCCL.")
        if not torch.cuda.is_available():
            return status_row(spec, status="skipped", error="CUDA is not available, so NCCL cannot run.")
        if spec.world_size > torch.cuda.device_count():
            return status_row(
                spec,
                status="skipped",
                error=f"Requested {spec.world_size} processes but only {torch.cuda.device_count()} GPU(s) are visible.",
            )

    context = py_mp.get_context("spawn")
    result_queue: py_mp.Queue = context.Queue()

    try:
        spawn(benchmark_worker, args=(spec, result_queue), nprocs=spec.world_size, join=True)
    except (ProcessRaisedException, ProcessExitedException, RuntimeError) as error:
        status = "oom" if is_oom_error(error) else "failed"
        return status_row(spec, status=status, error=f"{type(error).__name__}: {error}")

    try:
        return result_queue.get(timeout=5.0)
    except queue.Empty:
        return status_row(spec, status="failed", error="Worker finished without returning a benchmark result.")


def build_specs(args: argparse.Namespace) -> list[BenchmarkSpec]:
    specs: list[BenchmarkSpec] = []
    float32_bytes = torch.finfo(torch.float32).bits // 8
    for backend in args.backends:
        for world_size in args.world_sizes:
            for size_mb in args.sizes_mb:
                tensor_bytes = mib_to_bytes(size_mb)
                specs.append(
                    BenchmarkSpec(
                        backend=backend,
                        world_size=world_size,
                        size_mb=size_mb,
                        tensor_bytes=tensor_bytes,
                        numel=tensor_bytes // float32_bytes,
                        warmup_iters=args.warmup_iters,
                        min_iters=args.min_iters,
                        max_iters=args.max_iters,
                        target_runtime_s=args.target_runtime_s,
                        verify=not args.no_verify,
                        master_addr=args.master_addr,
                        master_port=pick_free_port(),
                    )
                )
    return specs


def write_results_csv(results: Sequence[dict[str, Any]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "all_reduce_results.csv"
    if not results:
        path.write_text("", encoding="utf-8")
        return path

    fieldnames: list[str] = []
    for result in results:
        for key in result.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    return path


def write_table_csv(df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    return output_path


def sort_ok_results(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["status"] == "ok"].copy().sort_values(["backend", "world_size", "size_mb"])


def build_latency_table(ok_df: pd.DataFrame) -> pd.DataFrame:
    if ok_df.empty:
        return pd.DataFrame()
    table = ok_df.pivot(index="size_label", columns=["backend_label", "world_size"], values="latency_ms")
    ordered_rows = [label for label in ok_df.sort_values("size_mb")["size_label"].drop_duplicates() if label in table.index]
    return table.reindex(ordered_rows)


def build_bandwidth_table(ok_df: pd.DataFrame) -> pd.DataFrame:
    if ok_df.empty:
        return pd.DataFrame()
    table = ok_df.pivot(index="size_label", columns=["backend_label", "world_size"], values="effective_bandwidth_GBps")
    ordered_rows = [label for label in ok_df.sort_values("size_mb")["size_label"].drop_duplicates() if label in table.index]
    return table.reindex(ordered_rows)


def plot_metric(
    ok_df: pd.DataFrame,
    output_path: Path,
    value_column: str,
    ylabel: str,
    title: str,
) -> Path | None:
    if ok_df.empty:
        return None

    backend_order = [backend for backend in DEFAULT_BACKENDS if backend in set(ok_df["backend"])]
    figure, axes = plt.subplots(1, len(backend_order), figsize=(6 * len(backend_order), 4.5), squeeze=False)

    for axis, backend in zip(axes[0], backend_order):
        subset = ok_df[ok_df["backend"] == backend].copy()
        if subset.empty:
            axis.set_visible(False)
            continue

        x_order = sorted(subset["size_mb"].unique())
        x_positions = list(range(len(x_order)))

        for world_size in sorted(subset["world_size"].unique()):
            line_df = subset[subset["world_size"] == world_size].sort_values("size_mb")
            if line_df.empty:
                continue
            line_x = [x_order.index(size) for size in line_df["size_mb"]]
            y_values = line_df[value_column].tolist()
            axis.plot(line_x, y_values, marker="o", linewidth=2, label=f"{world_size} procs")

        axis.set_title(backend_label(backend))
        axis.set_xlabel("Tensor size")
        axis.set_ylabel(ylabel)
        axis.set_xticks(x_positions, [size_label(size) for size in x_order])
        axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        axis.legend()

    figure.suptitle(title)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def format_table_for_report(table: pd.DataFrame, value_digits: int) -> str:
    if table.empty:
        return "No successful runs were available for this table."
    formatted = table.copy()
    formatted.columns = [f"{backend} | {world_size} procs" for backend, world_size in formatted.columns]
    return formatted.round(value_digits).to_string()


def overlap_ratio(
    ok_df: pd.DataFrame,
    metric: str,
    numerator_backend: str,
    denominator_backend: str,
) -> float | None:
    subset = ok_df[ok_df["backend"].isin([numerator_backend, denominator_backend])]
    if subset.empty:
        return None
    pivot = subset.pivot(index=["world_size", "size_mb"], columns="backend", values=metric).dropna()
    if pivot.empty:
        return None
    return float((pivot[numerator_backend] / pivot[denominator_backend]).median())


def size_scaling_ratio(ok_df: pd.DataFrame, metric: str) -> float | None:
    grouped_ratios: list[float] = []
    for _, group in ok_df.groupby(["backend", "world_size"]):
        if 1 not in set(group["size_mb"]) or 1024 not in set(group["size_mb"]):
            continue
        small = float(group[group["size_mb"] == 1][metric].iloc[0])
        large = float(group[group["size_mb"] == 1024][metric].iloc[0])
        if small > 0:
            grouped_ratios.append(large / small)
    if not grouped_ratios:
        return None
    grouped_ratios.sort()
    return grouped_ratios[len(grouped_ratios) // 2]


def process_scaling_ratio(ok_df: pd.DataFrame, backend: str) -> float | None:
    grouped_ratios: list[float] = []
    subset = ok_df[ok_df["backend"] == backend]
    for size_mb, group in subset.groupby("size_mb"):
        sizes_available = set(group["world_size"])
        if 2 not in sizes_available or 6 not in sizes_available:
            continue
        latency_two = float(group[group["world_size"] == 2]["latency_ms"].iloc[0])
        latency_six = float(group[group["world_size"] == 6]["latency_ms"].iloc[0])
        if latency_two > 0:
            grouped_ratios.append(latency_six / latency_two)
    if not grouped_ratios:
        return None
    grouped_ratios.sort()
    return grouped_ratios[len(grouped_ratios) // 2]


def generate_commentary(ok_df: pd.DataFrame, all_df: pd.DataFrame) -> list[str]:
    if ok_df.empty:
        skipped_count = int((all_df["status"] == "skipped").sum())
        failed_count = int(((all_df["status"] == "failed") | (all_df["status"] == "oom")).sum())
        return [
            "这次运行没有拿到成功的 benchmark 数据，请先检查 PyTorch 分布式后端、GPU 可见性和可用内存。",
            f"当前结果里共有 {skipped_count} 个配置被跳过，另有 {failed_count} 个配置失败或 OOM。",
        ]

    comments: list[str] = []

    backend_speedup = overlap_ratio(
        ok_df=ok_df,
        metric="effective_bandwidth_GBps",
        numerator_backend="nccl",
        denominator_backend="gloo",
    )
    if backend_speedup is not None:
        comments.append(
            f"在相同进程数和张量大小的重叠配置上，NCCL + GPU 的有效带宽中位数大约是 Gloo + CPU 的 {backend_speedup:.2f} 倍。"
        )
    else:
        available_labels = ", ".join(sorted(ok_df["backend_label"].unique()))
        comments.append(f"这次成功跑通的后端只有 {available_labels}，所以跨后端对比只覆盖了可用配置。")

    bandwidth_growth = size_scaling_ratio(ok_df=ok_df, metric="effective_bandwidth_GBps")
    if bandwidth_growth is not None:
        comments.append(
            f"把张量从 1MB 放大到 1GB 时，有效带宽的中位数通常还能提升到原来的 {bandwidth_growth:.2f} 倍，说明大消息更能摊薄启动和同步开销。"
        )
    else:
        comments.append("随着张量变大，延迟通常会上升；但大消息往往能更好地摊薄固定开销，所以带宽指标更值得一起看。")

    scaling_observations: list[str] = []
    for backend in DEFAULT_BACKENDS:
        ratio = process_scaling_ratio(ok_df=ok_df, backend=backend)
        if ratio is None:
            continue
        scaling_observations.append(f"{backend_label(backend)} 下 6 个进程相对 2 个进程的延迟中位数约为 {ratio:.2f} 倍")
    if scaling_observations:
        comments.append("；".join(scaling_observations) + "，说明进程数增加会带来额外同步成本。")

    return comments[:3]


def write_report(
    all_df: pd.DataFrame,
    latency_table: pd.DataFrame,
    bandwidth_table: pd.DataFrame,
    latency_plot_path: Path | None,
    bandwidth_plot_path: Path | None,
    output_dir: Path,
) -> Path:
    report_path = output_dir / "all_reduce_report.md"
    ok_df = sort_ok_results(all_df)
    commentary = generate_commentary(ok_df=ok_df, all_df=all_df)

    lines = [
        "# All-Reduce Benchmark Report",
        "",
        "## Files",
        "",
        "- `all_reduce_results.csv`: raw per-configuration results and statuses.",
        "- `latency_table_ms.csv`: pivoted latency table for successful runs.",
        "- `bandwidth_table_GBps.csv`: pivoted effective-bandwidth table for successful runs.",
    ]

    if latency_plot_path is not None:
        lines.append(f"- `{latency_plot_path.name}`: latency comparison plot.")
    if bandwidth_plot_path is not None:
        lines.append(f"- `{bandwidth_plot_path.name}`: effective-bandwidth comparison plot.")

    lines.extend(
        [
            "",
            "## Commentary",
            "",
        ]
    )
    lines.extend([f"- {comment}" for comment in commentary])

    lines.extend(
        [
            "",
            "## Latency Table (ms)",
            "",
            "```",
            format_table_for_report(latency_table, value_digits=3),
            "```",
            "",
            "## Effective Bandwidth Table (GB/s)",
            "",
            "```",
            format_table_for_report(bandwidth_table, value_digits=3),
            "```",
        ]
    )

    if latency_plot_path is not None:
        lines.extend(["", "## Latency Plot", "", f"![]({latency_plot_path.name})"])
    if bandwidth_plot_path is not None:
        lines.extend(["", "## Effective Bandwidth Plot", "", f"![]({bandwidth_plot_path.name})"])

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    if args.min_iters < 1:
        raise ValueError("--min-iters must be at least 1.")
    if args.max_iters < args.min_iters:
        raise ValueError("--max-iters must be greater than or equal to --min-iters.")
    if args.target_runtime_s <= 0:
        raise ValueError("--target-runtime-s must be positive.")
    if any(size_mb <= 0 for size_mb in args.sizes_mb):
        raise ValueError("All entries in --sizes-mb must be positive.")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = build_specs(args)
    results: list[dict[str, Any]] = []

    for index, spec in enumerate(specs, start=1):
        print(
            f"[{index}/{len(specs)}] backend={spec.backend} world_size={spec.world_size} "
            f"size={size_label(spec.size_mb)}"
        )
        result = run_benchmark(spec)
        results.append(result)
        if result["status"] == "ok":
            print(
                f"  ok: latency={result['latency_ms']:.3f} ms, "
                f"effective_bandwidth={result['effective_bandwidth_GBps']:.3f} GB/s"
            )
        else:
            print(f"  {result['status']}: {result['error']}")

    results_csv_path = write_results_csv(results, output_dir)
    all_df = pd.DataFrame(results)
    ok_df = sort_ok_results(all_df)

    latency_table = build_latency_table(ok_df)
    bandwidth_table = build_bandwidth_table(ok_df)

    latency_table_path = write_table_csv(latency_table, output_dir / "latency_table_ms.csv")
    bandwidth_table_path = write_table_csv(bandwidth_table, output_dir / "bandwidth_table_GBps.csv")

    latency_plot_path: Path | None = None
    bandwidth_plot_path: Path | None = None
    if not args.skip_plots:
        latency_plot_path = plot_metric(
            ok_df=ok_df,
            output_path=output_dir / "all_reduce_latency.png",
            value_column="latency_ms",
            ylabel="Latency (ms)",
            title="All-Reduce Latency by Tensor Size and Process Count",
        )
        bandwidth_plot_path = plot_metric(
            ok_df=ok_df,
            output_path=output_dir / "all_reduce_effective_bandwidth.png",
            value_column="effective_bandwidth_GBps",
            ylabel="Effective bandwidth (GB/s)",
            title="All-Reduce Effective Bandwidth by Tensor Size and Process Count",
        )

    report_path: Path | None = None
    if not args.skip_report:
        report_path = write_report(
            all_df=all_df,
            latency_table=latency_table,
            bandwidth_table=bandwidth_table,
            latency_plot_path=latency_plot_path,
            bandwidth_plot_path=bandwidth_plot_path,
            output_dir=output_dir,
        )

    print(f"Wrote raw results to {results_csv_path}")
    print(f"Wrote latency table to {latency_table_path}")
    print(f"Wrote bandwidth table to {bandwidth_table_path}")
    if latency_plot_path is not None:
        print(f"Wrote latency plot to {latency_plot_path}")
    if bandwidth_plot_path is not None:
        print(f"Wrote bandwidth plot to {bandwidth_plot_path}")
    if report_path is not None:
        print(f"Wrote markdown report to {report_path}")


if __name__ == "__main__":
    main()
