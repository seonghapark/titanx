"""Reproduce the job-8935726 failure on CPU/gloo and prove src_data_rank=None fixes it.

Builds a 2D mesh (dp_shard=2, tp=2) exactly as FSDP2+TP composition does, makes a
parameter whose placements are (_StridedShard, Shard) -- the layout that hit the
bare AssertionError -- and calls distribute_tensor the old way and the new way.

Run:  mpiexec -n 4 python test_strided.py      (or torchrun --nproc_per_node=4)
"""
import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor
from torch.distributed.tensor.placement_types import _StridedShard


def main():
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp_shard", "tp"))

    # A row-parallel weight, sharded on dim 0 by TP and then by FSDP -- the
    # composition that produces _StridedShard. Rows deliberately NOT divisible
    # by 4, so padding matters, which is what the scatter path asserted on.
    full = torch.arange(10 * 3, dtype=torch.float32).reshape(10, 3)
    placements = (_StridedShard(0, split_factor=2), Shard(0))

    for label, kwargs in (("src_data_rank=0 (old)", {}),
                          ("src_data_rank=None (new)", {"src_data_rank": None})):
        try:
            dt = distribute_tensor(full.clone(), mesh, placements, **kwargs)
            local = dt.to_local()
            ok = isinstance(dt, DTensor)
            msg = "OK  local%s sum=%.1f dtensor=%s" % (
                tuple(local.shape), float(local.sum()), ok)
        except BaseException as exc:  # noqa: BLE001 - bare AssertionError is the bug
            msg = "FAIL %s: %s" % (type(exc).__name__, exc or "<no message>")
        print("[rank%d] %-26s %s" % (rank, label, msg), flush=True)
        dist.barrier()

    # Round-trip: the new path must reconstruct the original tensor exactly.
    if rank == 0:
        print("--- full_tensor round-trip ---", flush=True)
    dist.barrier()
    dt = distribute_tensor(full.clone(), mesh, placements, src_data_rank=None)
    back = dt.full_tensor()
    if rank == 0:
        print("round-trip exact: %s  (max abs diff %.3g)" % (
            torch.equal(back, full), float((back - full).abs().max())), flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
