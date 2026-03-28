import torch
import torch.distributed as dist

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
class DDPIndividualParameters(torch.nn.Module):
    def __init__(self,module: torch.nn.Module):
        super().__init__()
        self.module=module
        with torch.no_grad():
            # for name, param in module.named_parameters():
            #     print(f"[rank {dist.get_rank()}] before broadcast: {name}, device={param.device}, shape={tuple(param.shape)}")
            #     torch.cuda.synchronize(param.device)
            #     dist.broadcast(param.data, src=0)
            #     torch.cuda.synchronize(param.device)
            #     print(f"[rank {dist.get_rank()}] after broadcast: {name}")
            for param in self.module.parameters():
                
                dist.broadcast(param, src=0)
    def forward(self,x):
        return self.module(x)
    
    @torch.no_grad()
    def finish_gradient_synchronization(self):
        world_size = dist.get_world_size()
        grads_buffer=[]
        for param in self.module.parameters():
            if param.grad is None:
                continue
            grads_buffer.append(param.grad)
            # param.grad /= world_size
        flat = _flatten_dense_tensors(grads_buffer)
        dist.all_reduce(flat.grad, op=dist.ReduceOp.SUM)
        flat.div_(world_size)
        synced_grads = _unflatten_dense_tensors(flat, grads_buffer)
        for grad, synced_grad in zip(grads_buffer, synced_grads):
            grad.copy_(synced_grad)
        return 