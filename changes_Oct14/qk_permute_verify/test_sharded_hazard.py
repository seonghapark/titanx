"""Is the to_local() path in _permute/_reverse_permute actually safe?

The functions do:
    if isinstance(w, DTensor): w = w.to_local()
    if dim1 is None: dim1 = w.shape[0]        # <-- LOCAL row count
    w.view(n_heads_arg, dim1 // n_heads_arg // 2, 2, dim2)
                ^^^^^^^^^^^ still GLOBAL

On a row-sharded (Shard(0)) tensor the local shard holds dim1/world rows while
n_heads_arg stays global, so the view regroups rows across head boundaries.
That is silent corruption, not an error -- unless the shapes happen not to
divide, in which case it raises instead.

This asks two things:
  1. Is the hazard real when a row-sharded DTensor is passed? (yes/no + how bad)
  2. Do the PRODUCTION call paths ever pass one?

(2) decides whether this is a live bug or a latent trap.
"""
import sys

import torch

N_HEADS, N_KV, HEAD_DIM, DIM = 16, 4, 128, 2048
Q_ROWS = N_HEADS * HEAD_DIM   # 2048
K_ROWS = N_KV * HEAD_DIM      # 512


def permute_local(w, n_heads_arg, dim1=None, dim2=None):
    """_permute's body, post-to_local (dim1 defaults to the LOCAL shape)."""
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


print("=" * 72)
print("1. HAZARD: row-sharded input, dim1 from local shape, n_heads global")
print("=" * 72)
full = torch.randn(Q_ROWS, DIM)
ref = permute_local(full, N_HEADS, Q_ROWS, DIM)

for world in (2, 4, 8, 12, 16, 48):
    if Q_ROWS % world:
        print(f"  world={world:2d}: {Q_ROWS} rows not divisible -- shard would be ragged")
        continue
    rows = Q_ROWS // world
    try:
        out = torch.cat(
            [permute_local(full[r * rows:(r + 1) * rows], N_HEADS) for r in range(world)]
        )
        nbad = int((out != ref).any(dim=1).sum())
        verdict = "IDENTICAL" if nbad == 0 else f"CORRUPT ({nbad}/{Q_ROWS} rows wrong)"
        print(f"  world={world:2d} local_rows={rows:4d}: {verdict}")
    except Exception as exc:
        print(f"  world={world:2d} local_rows={rows:4d}: RAISES {type(exc).__name__}")

print()
print("  Note: the K projection has only 512 rows (GQA, n_kv=4), so at world>=8")
print("  a row-sharded K shard has fewer rows than n_kv*2 and the view cannot")
print("  even be formed -- it would raise rather than corrupt.")

print()
print("=" * 72)
print("2. REACHABILITY: do the production paths pass a sharded DTensor?")
print("=" * 72)
print("""
  from_hf()  (LOAD, the path the FSDP fix was written for)
    checkpoint.py:589-593 builds `hf_state_dict = {}` and fills it via
    dcp.load(), whose own comment states the tensors arrive as REGULAR
    tensors, not DTensors, specifically "to avoid DTensor sharding pattern
    mismatches". Sharding is applied afterwards by
    model.load_state_dict(state_dict_native).
    => isinstance(w, DTensor) is FALSE here; to_local() never fires.

  to_hf()  (SAVE)
    receives the model state dict, where weights ARE DTensors under FSDP.
    This is where to_local() actually fires.
""")

print("=" * 72)
print("3. WHAT to_local() RETURNS ON THE SAVE PATH")
print("=" * 72)
print("""
  Decisive question: under this config is the Q/K weight REPLICATED or
  SHARDED across the mesh?

  AGPT runs tensor_parallel_degree=1 / pipeline=1 / context=1 (see the launch
  banner: "Tensor Parallelism = 1"), so the only mesh dim is dp_shard (FSDP2).
  FSDP2 shards parameters on dim 0 -- so under FSDP2 these DO arrive as
  Shard(0) DTensors, and to_local() yields a row-slice.

  That means the hazard in section 1 is NOT hypothetical for to_hf(); it is
  governed entirely by whether to_hf() is called on sharded parameters or on
  an already-gathered full state dict.
""")
sys.exit(0)
