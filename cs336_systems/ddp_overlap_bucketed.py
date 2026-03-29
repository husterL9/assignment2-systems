import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

class Bucket:
    def __init__(self,bucket_size_mb) -> None:
        self.bucket_size_bytes:float=bucket_size_mb*1_000*1_000
        self.pending=0
        self.flat_grads:torch.Tensor | None = None
        self.param_list: list[torch.nn.Parameter] = []
        self.ready_param_ids:set[int]=set()
        self.work: object | None = None

    def append_param(self, param: torch.nn.Parameter)-> None:
        self.param_list.append(param)

class DDP_bucket(torch.nn.Module):
    def __init__(self,module: torch.nn.Module,bucket_size_mb:float):
        super().__init__()
        self.module=module
        self._buckets=[]
        self._param_to_bucket: dict[int, Bucket] = {}
        self.world_size = dist.get_world_size()
        params = [p for p in module.parameters() if p.requires_grad]
        reversed_params = list(reversed(params))
        cur:Bucket= Bucket(bucket_size_mb)
        cur_bytes = 0
        for p in reversed_params:
            p_bytes = p.numel() * p.element_size()
            if cur.param_list and cur_bytes + p_bytes > cur.bucket_size_bytes:
                self._buckets.append(cur)
                cur = Bucket(bucket_size_mb)
                cur_bytes = 0
            cur.append_param(p)
            cur_bytes += p_bytes
            self._param_to_bucket[id(p)] = cur
        if cur.param_list:
            self._buckets.append(cur)

        with torch.no_grad():
            for param in self.module.parameters():
                dist.broadcast(param, src=0)
            for param in self.module.parameters():
                if not param.requires_grad:
                    continue
                handle = param.register_post_accumulate_grad_hook(
                    self._make_post_accumulate_hook()
                )
    def reset_state_of_bucket(self) -> None:
        for bucket in self._buckets:
            bucket.pending = len(bucket.param_list)
            bucket.ready_param_ids.clear()
            bucket.flat_grads = None
            bucket.work = None
    def _make_post_accumulate_hook(self):
        def hook(param: torch.Tensor):
            if param.grad is None:
                return
            bucket = self._param_to_bucket[id(param)]
            bucket.ready_param_ids.add(id(param))
            bucket.pending -= 1
            if bucket.pending == 0:
                grads: list[torch.Tensor] = []
                for p in bucket.param_list:
                    assert p.grad is not None
                    grads.append(p.grad)
                bucket.flat_grads = _flatten_dense_tensors(grads)
                bucket.work = dist.all_reduce(
                    bucket.flat_grads,
                    op=dist.ReduceOp.SUM,
                    async_op=True,
                )

        return hook

    def forward(self,*input,**kwargs):
        return self.module(*input,**kwargs)
    
    @torch.no_grad()
    def finish_gradient_synchronization(self):
        for bucket in self._buckets:
            if bucket.work is None:
                continue

            bucket.work.wait()
            assert bucket.flat_grads is not None
            bucket.flat_grads.div_(self.world_size)

            grads: list[torch.Tensor] = []
            for p in bucket.param_list:
                assert p.grad is not None
                grads.append(p.grad)

            synced_grads = _unflatten_dense_tensors(bucket.flat_grads,  grads)

            for p, synced_grad in zip(bucket.param_list, synced_grads):
                assert p.grad is not None
                p.grad.copy_(synced_grad)