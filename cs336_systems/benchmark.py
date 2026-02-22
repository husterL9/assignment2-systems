# benchmark.py
import argparse
import os
import timeit
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.utils import nvtx_is_enabled, nvtx_range
from typing import Callable, Optional


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab-size", type=int, default=10_000)
    p.add_argument("--context-length", type=int, default=512)
    p.add_argument("--d-model", type=int, default=768)
    p.add_argument("--num-layers", type=int, default=12)
    p.add_argument("--num-heads", type=int, default=12)
    p.add_argument("--d-ff", type=int, default=3072)
    p.add_argument("--rope-theta", type=float, default=10_000.0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--backward", action="store_true")  # 是否测前后向
    p.add_argument("--optimizer", action="store_true")  # 是否测优化器 step
    p.add_argument("--time-forward-only", action="store_true")  # 仅统计 forward 时间
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--nvtx", action="store_true")  # nsys/nvtx 标注
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--model-size", type=str, default="")
    p.add_argument("--result-file", type=str, default="")
    p.add_argument(
        "--mixed-precision",
        choices=("none", "bf16"),
        default="none",
        help="Optional mixed precision mode. 'bf16' enables autocast with bfloat16.",
    )
    args = p.parse_args()

    if args.nvtx:
        os.environ["CS336_NVTX"] = "1"

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    if args.optimizer and not args.backward:
        args.backward = True
    if args.mixed_precision == "bf16" and not device.startswith("cuda"):
        raise ValueError("--mixed-precision bf16 requires a CUDA device.")

    autocast_context: Callable[[], object]
    if args.mixed_precision == "bf16":
        autocast_context = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast_context = nullcontext

    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(device)
    model.train()

    optimizer = None
    if args.optimizer:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # 随机输入与目标
    x = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)
    y: Optional[torch.Tensor] = None

    if args.backward:
        y = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)

    nvtx_enabled = nvtx_is_enabled(args.nvtx) and device.startswith("cuda")

    times = []
    for i in range(args.warmup + args.steps):
        is_warmup = i < args.warmup
        step_label = "step_warmup" if is_warmup else "step_measured"
        if args.backward:
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            else:
                model.zero_grad(set_to_none=True)

        with nvtx_range(step_label, nvtx_enabled):
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            if device == "mps":
                torch.mps.synchronize()
            start = timeit.default_timer()

            with nvtx_range("forward", nvtx_enabled):
                with autocast_context():
                    logits = model(x)
                if args.time_forward_only:
                    if device.startswith("cuda"):
                        torch.cuda.synchronize()
                    if device == "mps":
                        torch.mps.synchronize()
                    end = timeit.default_timer()
            if args.backward:
                assert y is not None
                with nvtx_range("backward", nvtx_enabled):
                    with autocast_context():
                        loss = F.cross_entropy(logits.reshape(-1, args.vocab_size), y.reshape(-1))
                    loss.backward()
            if optimizer is not None:
                with nvtx_range("optimizer_step", nvtx_enabled):
                    optimizer.step()

            if not args.time_forward_only:
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                if device == "mps":
                    torch.mps.synchronize()
                end = timeit.default_timer()
        if i >= args.warmup:
            times.append(end - start) # type: ignore

    mean = sum(times) / len(times)
    var = sum((t - mean) ** 2 for t in times) / len(times)
    std = var ** 0.5
    scope = "forward" if args.time_forward_only else "step"
    print(
        f"{scope} mean: {mean:.6f}s  std: {std:.6f}s  ({len(times)} steps)  "
        f"precision={args.mixed_precision}"
    )

    if args.result_file:
        result_path = Path(args.result_file)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not result_path.exists() or result_path.stat().st_size == 0
        with result_path.open("a", encoding="utf-8") as f:
            if write_header:
                f.write(
                    "model_size,context_length,batch_size,backward,optimizer,time_forward_only,mixed_precision,mean_s,std_s,steps,device\n"
                )
            f.write(
                f"{args.model_size},{args.context_length},{args.batch_size},"
                f"{int(args.backward)},{int(args.optimizer)},"
                f"{int(args.time_forward_only)},{args.mixed_precision},"
                f"{mean:.6f},{std:.6f},{len(times)},{device}\n"
            )

if __name__ == "__main__":
    main()
