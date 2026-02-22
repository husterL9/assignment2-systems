#!/usr/bin/env python3
"""Part (a): inspect dtypes under CUDA autocast(fp16) mixed precision."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        fc1_out = self.relu(self.fc1(x))
        ln_out = self.ln(fc1_out)
        logits = self.fc2(ln_out)
        return logits, fc1_out, ln_out


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This script expects a CUDA GPU for fp16 autocast.")

    device = "cuda"
    torch.manual_seed(0)

    in_features = 16
    out_features = 7
    batch_size = 8

    model = ToyModel(in_features=in_features, out_features=out_features).to(device)
    model.train()

    x = torch.randn(batch_size, in_features, device=device, dtype=torch.float32)
    target = torch.randint(0, out_features, (batch_size,), device=device, dtype=torch.long)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits, fc1_out, ln_out = model(x)
        loss = F.cross_entropy(logits, target)

    loss.backward()

    param_dtypes = sorted({str(p.dtype) for p in model.parameters()})
    grad_dtypes = sorted({str(p.grad.dtype) for p in model.parameters() if p.grad is not None})

    print(f"model params dtypes (inside autocast): {param_dtypes}")
    print(f"fc1 output dtype: {fc1_out.dtype}")
    print(f"layernorm output dtype: {ln_out.dtype}")
    print(f"logits dtype: {logits.dtype}")
    print(f"loss dtype: {loss.dtype}")
    print(f"grad dtypes: {grad_dtypes}")


if __name__ == "__main__":
    main()
