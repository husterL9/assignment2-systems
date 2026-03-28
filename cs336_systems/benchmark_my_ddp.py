import argparse
from copy import deepcopy
from dataclasses import dataclass
import time
from typing import Union
import torch.nn.functional as F

import torch
import torch.distributed as dist
import torch.optim as optim

from cs336_basics.model import BasicsTransformerLM
import torch.multiprocessing as mp

from cs336_systems.ddp_individual_parameters import DDPIndividualParameters
from tests.common import (_cleanup_process_group, 
                          _setup_process_group, 
                          validate_ddp_net_equivalence)

@dataclass
class BenchmarkConfig:
    vocab_size: int = 1000
    seq_len: int = 1024
    global_batch_size: int = 8
    d_model:int=768
    d_ff:int=3072
    num_heads:int=12
    num_layers:int =12
    rope_theta:float=10_000.0
    steps: int = 30
    warmup_steps: int = 10
    lr: float = 3e-4
    dtype: str = "bf16"  # "fp32", "fp16", "bf16"

def get_random_batch(batch_size: int, seq_len: int, vocab_size: int, device: Union[torch.device,str]):
    idx = torch.randint(
        low=0, high=vocab_size, size=(batch_size, seq_len), device=device
    )
    targets = torch.randint(
        low=0, high=vocab_size, size=(batch_size, seq_len), device=device
    )
    return idx, targets

def _test_myDDP(rank: int, world_size: int,args):
    
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    device = _setup_process_group(rank=rank, world_size=world_size, backend=backend)
    print(f"[rank {rank}] after setup, device={device}, local_rank={rank}", flush=True)
    if device.type == "cuda":
        print("======================\n")
        dist.barrier(device_ids=[device.index])
    else:
        dist.barrier()    
    print(f"[rank {rank}] after barrier", flush=True)
    tmp = torch.randn(1000, 768, device=device, dtype=torch.float32)
    print(f"[rank {rank}] tmp before: {tmp[0,0].item()}", flush=True)
    dist.broadcast(tmp, src=0)
    torch.cuda.synchronize(device)
    print(f"[rank {rank}] tmp after: {tmp[0,0].item()}", flush=True)

    cfg = BenchmarkConfig(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        global_batch_size=args.global_batch_size,
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        rope_theta=args.rope_theta,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        lr=args.lr,
        dtype=args.dtype,
    )
    assert cfg.global_batch_size % world_size == 0, \
        "global_batch_size must be divisible by world_size"
    torch.manual_seed(rank)

    non_parallel_model = BasicsTransformerLM(cfg.vocab_size,
                                             cfg.seq_len,cfg.d_model,
                                             cfg.num_layers
                                             ,cfg.num_heads,cfg.d_ff,
                                             cfg.rope_theta).to(device)
    non_parallel_model.train()  
    ddp_base = deepcopy(non_parallel_model)
    tmp = torch.randn(1000, 768, device=device)
    print(f"[rank {dist.get_rank()}] tmp before:", tmp[0, 0].item())
    dist.broadcast(tmp, src=0)
    torch.cuda.synchronize()
    print(f"[rank {dist.get_rank()}] tmp after:", tmp[0, 0].item())
    ddp_model = DDPIndividualParameters(ddp_base)
    for (non_parallel_param_name, non_parallel_model_parameter), (
        ddp_model_param_name,
        ddp_model_parameter,
    ) in zip(non_parallel_model.named_parameters(), ddp_model.named_parameters()):
        # is_no_grad_fixed_param = (
        #     "no_grad_fixed_param" in ddp_model_param_name or "no_grad_fixed_param" in non_parallel_param_name
        # )
        # if rank == 0 or is_no_grad_fixed_param:
        if rank == 0 :
            assert torch.allclose(non_parallel_model_parameter, ddp_model_parameter)
        else:
            assert not torch.allclose(non_parallel_model_parameter, ddp_model_parameter)

    validate_ddp_net_equivalence(ddp_model)

    assert cfg.global_batch_size % world_size == 0
    local_bs = int(cfg.global_batch_size / world_size)
    loss_fn = F.cross_entropy
    # Optimizer for the DDP model
    ddp_optimizer = optim.SGD(ddp_model.parameters(), lr=0.1)
    # Optimizer for the non-parallel model
    non_parallel_optimizer = optim.SGD(non_parallel_model.parameters(), lr=0.1)

    total_steps=cfg.warmup_steps+cfg.steps

    # barrier before timing
    dist.barrier()
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    iter_times = []
    comm_times = []
    noDPP_iter_times=[]
    for i in range(total_steps):
        ddp_optimizer.zero_grad()
        non_parallel_optimizer.zero_grad()

        torch.manual_seed(42 + i)

        all_x, all_y = get_random_batch(
            batch_size=cfg.global_batch_size,
            seq_len=cfg.seq_len,
            vocab_size=cfg.vocab_size,
            device=device,
        )
        # Run the non-parallel model on all the data and take a gradient step
        non_parallel_data = all_x.to(device)
        non_parallel_labels = all_y.to(device)
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        start = time.perf_counter()

        non_parallel_outputs = non_parallel_model(non_parallel_data)
        non_parallel_loss = loss_fn(non_parallel_outputs.view(-1,cfg.vocab_size), non_parallel_labels.view(-1))
        non_parallel_loss.backward()
        non_parallel_optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        end = time.perf_counter()
        noDPP_iter_time=end-start
        # At this point, the parameters of non-parallel model should differ
        # from the parameters of the DDP model (since we've applied the
        # gradient step to the non-parallel model, but not to the DDP model).
        if rank == 0:
            for non_parallel_model_parameter, ddp_model_parameter in zip(
                non_parallel_model.parameters(), ddp_model.parameters()
            ):
                if non_parallel_model_parameter.requires_grad and ddp_model_parameter.requires_grad:
                    # The only parameters that change are those that require_grad
                    assert not torch.allclose(non_parallel_model_parameter, ddp_model_parameter)
                else:
                    # parameters that don't require_grad shouldn't change
                    assert torch.allclose(non_parallel_model_parameter, ddp_model_parameter)

        # While the non-parallel model does a forward pass on all the data (20 examples),
        # each DDP rank only sees 10 (disjoint) examples.
        # However, the end result should be the same as doing a forward pass on all 20 examples.
        offset = rank * local_bs
        ddp_data = all_x[offset : offset + local_bs, :].to(device)
        ddp_labels = all_y[offset : offset + local_bs, :].to(device)

        # synchronize before whole-iteration timing
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        iter_start = time.perf_counter()


        ddp_outputs = ddp_model(ddp_data)
        ddp_loss = loss_fn(ddp_outputs.view(-1,cfg.vocab_size), ddp_labels.view(-1))
        ddp_loss.backward()

        # communication timing
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        comm_start = time.perf_counter()

        # Run student-written code that needs to execute after the backward pass,
        # but before the optimizer step (e.g., to wait for all DDP ranks to sync gradients)
        ddp_model.finish_gradient_synchronization()
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        comm_end = time.perf_counter()

        ddp_optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        iter_end = time.perf_counter()

        iter_time = iter_end - iter_start
        comm_time = comm_end - comm_start


        if i >= cfg.warmup_steps:
            iter_times.append(iter_time)
            comm_times.append(comm_time)
            noDPP_iter_times.append(noDPP_iter_time)
        if rank == 0:
            print(
                f"step={i:03d} "
                f"loss={ddp_loss.item():.4f} "
                f"iter_time={iter_time*1000:.2f} ms "
                f"comm_time={comm_time*1000:.2f} ms"
                f"comm_time={noDPP_iter_time*1000:.2f} ms"
            )
        # At this point, the non-parallel model should exactly match the parameters of the DDP model
        if rank == 0:
            for non_parallel_model_parameter, ddp_model_parameter in zip(
                non_parallel_model.parameters(), ddp_model.parameters()
            ):
                assert torch.allclose(non_parallel_model_parameter, ddp_model_parameter)
        # After training is done, we should have the same weights on both the non-parallel baseline
    # and the model trained with DDP.
    if rank == 0:
        for non_parallel_model_parameter, ddp_model_parameter in zip(
            non_parallel_model.parameters(), ddp_model.parameters()
        ):
            assert torch.allclose(non_parallel_model_parameter, ddp_model_parameter)
            # aggregate results across ranks
    iter_tensor = torch.tensor(iter_times, device=device, dtype=torch.float64)
    comm_tensor = torch.tensor(comm_times, device=device, dtype=torch.float64)
    noDPP_iter_tensor=torch.tensor(noDPP_iter_times, device=device, dtype=torch.float64)
    mean_iter = iter_tensor.mean()
    mean_comm = comm_tensor.mean()
    mean_noDPP_iter=noDPP_iter_tensor.mean()
    # average across ranks
    dist.all_reduce(mean_iter, op=dist.ReduceOp.SUM)
    dist.all_reduce(mean_comm, op=dist.ReduceOp.SUM)
    mean_iter /= world_size
    mean_comm /= world_size

    if rank == 0:
        mean_iter_ms = mean_iter.item() * 1000
        mean_comm_ms = mean_comm.item() * 1000
        mean_noDPP_iter_ms=mean_noDPP_iter.item()*1000
        comm_ratio = mean_comm.item() / mean_iter.item()

        print("\n===== Benchmark Result =====")
        print(f"Average step time: {mean_iter_ms:.2f} ms")
        print(f"Average gradient communication time: {mean_comm_ms:.2f} ms")
        print(f"Average noDPP_iter time: {mean_noDPP_iter_ms:.2f} ms")

        print(f"Communication ratio: {comm_ratio*100:.2f}%")

    _cleanup_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nproc_per_node", type=int, default=2)
    parser.add_argument("--master_addr", type=str, default="127.0.0.1")
    parser.add_argument("--master_port", type=str, default="29500")
    parser.add_argument("--vocab_size", type=int, default=1000)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--global_batch_size", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=1600)
    parser.add_argument("--num_heads", type=int, default=25)
    parser.add_argument("--d_ff", type=int, default=6400)
    parser.add_argument("--num_layers", type=int, default=48)
    parser.add_argument("--rope_theta", type=float, default=10_000.0)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    args = parser.parse_args()

    world_size = args.nproc_per_node
    mp.spawn(_test_myDDP, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()