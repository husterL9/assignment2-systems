# benchmark.py
import argparse
import timeit
import torch
import torch.nn.functional as F
from cs336_basics.model import BasicsTransformerLM


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
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(device)

    # 随机输入与目标
    x = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)
    y = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)

    times = []
    for i in range(args.warmup + args.steps):
        if args.backward:
            model.zero_grad(set_to_none=True)

        if device == "cuda":
            torch.cuda.synchronize()
        if device =="mps":
            torch.mps.synchronize()
        start = timeit.default_timer()

        logits = model(x)
        if args.backward:
            loss = F.cross_entropy(logits.view(-1, args.vocab_size), y.view(-1))
            loss.backward()

        if device == "cuda":
            torch.cuda.synchronize()
        if device =="mps":
            torch.mps.synchronize()
            
        end = timeit.default_timer()
        if i >= args.warmup:
            times.append(end - start)

    mean = sum(times) / len(times)
    var = sum((t - mean) ** 2 for t in times) / len(times)
    std = var ** 0.5
    print(f"mean: {mean:.6f}s  std: {std:.6f}s  ({len(times)} steps)")

if __name__ == "__main__":
    main()
