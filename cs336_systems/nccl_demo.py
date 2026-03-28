import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def worker(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    x = torch.randn(800, 800, device=f"cuda:{rank}")
    if rank == 0:
        x.fill_(1.0)
    else:
        x.fill_(2.0)

    print(f"[rank {rank}] before:", x[0, 0].item())
    dist.broadcast(x, src=0)
    torch.cuda.synchronize(rank)
    print(f"[rank {rank}] after:", x[0, 0].item())

    dist.destroy_process_group()

if __name__ == "__main__":
    mp.spawn(worker, args=(2,), nprocs=2)