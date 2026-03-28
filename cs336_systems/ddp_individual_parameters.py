import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
class DDPIndividualParameters(torch.nn.Module):
    def __init__(self,module: torch.nn.Module):
        super().__init__()
        self.module=module
        self._post_accumulate_handles=[]
        self._grad_sync_handles=[]
        with torch.no_grad():
            # for name, param in module.named_parameters():
            #     print(f"[rank {dist.get_rank()}] before broadcast: {name}, device={param.device}, shape={tuple(param.shape)}")
            #     torch.cuda.synchronize(param.device)
            #     dist.broadcast(param.data, src=0)
            #     torch.cuda.synchronize(param.device)
            #     print(f"[rank {dist.get_rank()}] after broadcast: {name}")
            for param in self.module.parameters():
                dist.broadcast(param, src=0)
            for param in self.module.parameters():
                if not param.requires_grad:
                    continue

                handle = param.register_post_accumulate_grad_hook(
                    self._make_post_accumulate_hook()
                )
                self._post_accumulate_handles.append(handle)

    def _make_post_accumulate_hook(self):
        def hook(param: torch.Tensor):
            if param.grad is None:
                return
            work = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
            self._grad_sync_handles.append((param, work))

        return hook

    def forward(self,*input,**kwargs):
        return self.module(*input,**kwargs)
    
    @torch.no_grad()
    def finish_gradient_synchronization(self):
        for param, work in self._grad_sync_handles:
            work.wait()
            param.grad /= self.world_size

        self._grad_sync_handles.clear()
        return 