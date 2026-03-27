import torch
import torch.distributed as dist


class DDPIndividualParameters(torch.nn.Module):
    def __init__(self,module: torch.nn.Module):
        super().__init__()
        self.module=module
        with torch.no_grad():
            for param in self.module.parameters():
                dist.broadcast(param, src=0)

    def forward(self,x):
        return self.module(x)
    
    @torch.no_grad()
    def finish_gradient_synchronization(self):
        world_size = dist.get_world_size()
        for param in self.module.parameters():
            if param.grad is None:
                continue
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= world_size
        return 