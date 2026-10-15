"""Does model.state_dict() under FSDP2 hand to_hf() SHARDED DTensors?

Reading checkpoint.py says yes:
    checkpoint.py:952  states = self.states[MODEL].state_dict()
    checkpoint.py:106  {k: v for model in self.model for k, v in model.state_dict().items()}
    checkpoint.py:496  state_dict = self.sd_adapter.to_hf(state_dict)
-- no gather, no full_state_dict option, no distributed_state_dict call in
between. But reading is not proof. This runs real FSDP2 (fully_shard) on gloo
and inspects what actually comes out.

Three things are checked, in order of how damning they are:
  1. placement of wq/wk in state_dict()          -> Shard(0)?
  2. result of _permute() on that DTensor        -> matches unsharded?
  3. what each rank would hand to dcp.save()     -> a plain local tensor?

Run: torchrun --nproc_per_node=2 test_fsdp_save_path.py
"""
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

N_HEADS, N_KV, HEAD_DIM, DIM = 16, 4, 128, 2048


def permute(w, n_heads_arg, dim1=None, dim2=None):
    """Verbatim body of Llama3StateDictAdapter._permute."""
    if isinstance(w, DTensor):
        w = w.to_local()
    if dim1 is None:
        dim1 = w.shape[0]
    if dim2 is None:
        dim2 = w.shape[1]
    return (
        w.view(n_heads_arg, dim1 // n_heads_arg // 2, 2, dim2)
        .transpose(1, 2)
        .reshape(dim1, dim2)
        .clone()
    )


class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = nn.Linear(DIM, N_HEADS * HEAD_DIM, bias=False)
        self.wk = nn.Linear(DIM, N_KV * HEAD_DIM, bias=False)


dist.init_process_group("gloo")
rank, world = dist.get_rank(), dist.get_world_size()
torch.manual_seed(1234)  # identical init on every rank

m = Attn()
# Keep a pre-shard copy of the true global weights for the correctness check.
gold_q = m.wq.weight.detach().clone()
gold_k = m.wk.weight.detach().clone()
ref_q = permute(gold_q, N_HEADS)
ref_k = permute(gold_k, N_KV, N_KV * HEAD_DIM, DIM)

fully_shard(m)
sd = m.state_dict()  # exactly what ModelWrapper._get_state_dict() returns

if rank == 0:
    print("=" * 72)
    print("1. WHAT model.state_dict() RETURNS UNDER FSDP2")
    print("=" * 72)
for name in ("wq.weight", "wk.weight"):
    v = sd[name]
    if rank == 0:
        isdt = isinstance(v, DTensor)
        print(f"  {name:12s} DTensor={isdt}", end="")
        if isdt:
            print(f"  global={tuple(v.shape)}  placements={v.placements}"
                  f"  local={tuple(v.to_local().shape)}")
        else:
            print(f"  shape={tuple(v.shape)}")

dist.barrier()

if rank == 0:
    print()
    print("=" * 72)
    print("2. WHAT _permute() PRODUCES ON THAT STATE DICT")
    print("=" * 72)

results = {}
for name, nh, d1, ref in (
    ("wq.weight", N_HEADS, None, ref_q),
    ("wk.weight", N_KV, N_KV * HEAD_DIM, ref_k),
):
    try:
        out = permute(sd[name], nh, d1, DIM) if d1 else permute(sd[name], nh)
        results[name] = ("ok", out)
    except Exception as exc:
        results[name] = ("raise", f"{type(exc).__name__}: {exc}")

# Gather each rank's piece on rank 0 and compare against the true permute.
for name, nh, ref in (("wq.weight", N_HEADS, ref_q), ("wk.weight", N_KV, ref_k)):
    status, payload = results[name]
    gathered = [None] * world
    dist.all_gather_object(gathered, (status, payload if status == "ok" else payload))
    if rank == 0:
        if any(s != "ok" for s, _ in gathered):
            for r, (s, p) in enumerate(gathered):
                if s != "ok":
                    print(f"  {name}: rank {r} RAISED {p}")
            continue
        cat = torch.cat([p for _, p in gathered], dim=0)
        if cat.shape != ref.shape:
            print(f"  {name}: concatenated {tuple(cat.shape)} != expected {tuple(ref.shape)}")
            continue
        nbad = int((cat != ref).any(dim=1).sum())
        verdict = "matches unsharded permute" if nbad == 0 else \
                  f"*** CORRUPT: {nbad}/{ref.shape[0]} rows differ ***"
        print(f"  {name}: {verdict}")

dist.barrier()
if rank == 0:
    print()
    print("=" * 72)
    print("3. WHAT GOES INTO dcp.save()")
    print("=" * 72)
    s, p = results["wq.weight"]
    if s == "ok":
        print(f"  _permute returned a {type(p).__name__} of shape {tuple(p.shape)}")
        print(f"  (the global weight is {tuple(ref_q.shape)})")
        print("  -> the DTensor's sharding metadata is GONE; each rank hands DCP a")
        print("     plain tensor holding only its own rows.")
    print()

dist.destroy_process_group()
