"""End-to-end rehearsal of the hf_load fix on CPU/gloo, minus the real model.

Bug #2 was `aten.copy_.default got mixed torch.Tensor and DTensor` inside
load_state_dict. Bug #3 was the _StridedShard AssertionError in the fix for #2.
This runs the exact resharding loop from checkpoint.py against a module whose
parameters are DTensors with FSDP2+TP-style placements, then verifies the
parameters hold the intended VALUES -- not merely that nothing raised.

Run:  python3 -m torch.distributed.run --nproc_per_node=4 test_loadsd.py
"""
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor
from torch.distributed.tensor.placement_types import _StridedShard


def main():
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp_shard", "tp"))

    # Reference weights, as from_hf() would hand them over: plain full tensors,
    # identical on every rank. Odd row counts so padding is exercised.
    truth = {
        "wq.weight": torch.randn(10, 6, generator=torch.Generator().manual_seed(0)),
        "wk.weight": torch.randn(6, 6, generator=torch.Generator().manual_seed(1)),
        "tok.weight": torch.randn(14, 6, generator=torch.Generator().manual_seed(2)),
    }
    placements = {
        "wq.weight": (_StridedShard(0, split_factor=2), Shard(0)),
        "wk.weight": (_StridedShard(0, split_factor=2), Shard(0)),
        "tok.weight": (Shard(0), Shard(0)),
    }

    mod = nn.Module()
    for name, ref in truth.items():
        attr = name.split(".")[0]
        sub = nn.Module()
        # Start from zeros so a no-op load is detectable, unlike random init.
        sub.weight = nn.Parameter(
            distribute_tensor(torch.zeros_like(ref), mesh, placements[name],
                              src_data_rank=None))
        setattr(mod, attr, sub)

    # ---- the loop as it stands in checkpoint.py -------------------------
    state_dict_native = {k: v.clone() for k, v in truth.items()}
    model_sd = mod.state_dict()
    for key, loaded in state_dict_native.items():
        target = model_sd.get(key)
        if isinstance(target, DTensor) and not isinstance(loaded, DTensor):
            state_dict_native[key] = distribute_tensor(
                loaded.to(target.dtype), target.device_mesh, target.placements,
                src_data_rank=None)

    matched = sum(1 for key in state_dict_native if key in model_sd)
    try:
        mod.load_state_dict(state_dict_native, strict=False)
        loaded_ok = "OK"
    except BaseException as exc:  # noqa: BLE001
        loaded_ok = "FAIL %s: %s" % (type(exc).__name__, exc or "<no message>")
    print("[rank%d] matched %d/%d  load_state_dict %s" % (
        rank, matched, len(model_sd), loaded_ok), flush=True)
    dist.barrier()

    # ---- did the VALUES actually arrive? --------------------------------
    worst = 0.0
    for name, ref in truth.items():
        got = getattr(mod, name.split(".")[0]).weight
        full = got.full_tensor() if isinstance(got, DTensor) else got
        worst = max(worst, float((full - ref).abs().max()))
    if rank == 0:
        print("max abs deviation from reference weights: %.3g  -> %s" % (
            worst, "WEIGHTS ARRIVED" if worst == 0.0 else "MISMATCH"), flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
