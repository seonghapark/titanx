"""What does a real FSDP2 shard of AGPT-2B's wq/wk look like, and does
_permute survive it?

AGPT-2B layer 0: wq is (2048, 2048), wk is (512, 2048) with n_kv=4.
Under FSDP2 (dp_shard only, TP=1) parameters are Shard(0): each rank holds a
contiguous row slice.

For _permute's view to be even FORMABLE post-to_local, it needs
    local_rows == n_heads_arg * (dim1 // n_heads_arg // 2) * 2
with dim1 = local_rows and n_heads_arg GLOBAL. That is
    local_rows divisible by 2*n_heads_arg.
And for it to be CORRECT it additionally needs each rank's slice to contain
whole heads AND the grouping to match -- which it does not, because the head
index inside the view is re-derived from the local row count.
"""
import torch

CASES = [
    ("wq", 16, 2048, 2048),   # name, n_heads_arg, rows, cols
    ("wk",  4,  512, 2048),
]
WORLDS = [1, 2, 4, 8, 12, 16, 24, 48]


def permute_local(w, nh):
    d1, d2 = w.shape[0], w.shape[1]
    return (
        w.view(nh, d1 // nh // 2, 2, d2).transpose(1, 2).reshape(d1, d2).clone()
    )


for name, nh, rows, cols in CASES:
    print(f"\n{'=' * 70}\n{name}: global shape ({rows}, {cols}), n_heads_arg={nh}\n{'=' * 70}")
    full = torch.randn(rows, cols)
    ref = permute_local(full, nh)  # correct result, computed on the full tensor
    print(f"{'world':>6} {'local_rows':>11} {'view formable':>14}  result")
    for world in WORLDS:
        if rows % world:
            print(f"{world:>6} {'ragged':>11} {'-':>14}  uneven shard (torchtitan pads/uneven-shards)")
            continue
        lr = rows // world
        formable = (lr % (2 * nh)) == 0
        if not formable:
            print(f"{world:>6} {lr:>11} {'NO':>14}  view() would RAISE")
            continue
        out = torch.cat([permute_local(full[r * lr:(r + 1) * lr], nh) for r in range(world)])
        nbad = int((out != ref).any(dim=1).sum())
        tag = "IDENTICAL to unsharded" if nbad == 0 else f"WRONG ({nbad}/{rows} rows)"
        print(f"{world:>6} {lr:>11} {'yes':>14}  {tag}")

print(f"""
{'=' * 70}
READ THIS CAREFULLY
{'=' * 70}
world=1 is the only configuration that is correct by construction: the
"shard" is the whole tensor, so dim1 (local) == dim1 (global).

Every world>1 row above where the view is formable produces a DIFFERENT
tensor than permuting the full weight. The permutation regroups rows within
each shard as if that shard were the entire weight.
""")
