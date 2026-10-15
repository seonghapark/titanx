"""Q/K permutation correctness, on CPU, no XPU and no distributed runtime.

Three separate questions, deliberately not conflated:

  1. Is _reverse_permute the exact inverse of _permute? (round-trip identity)
  2. Does the permutation implement the layout change it CLAIMS -- HF's
     half-split rotate_half pairing <-> Meta's interleaved pairing? A function
     can be perfectly self-inverse and still permute the wrong rows, so the
     round-trip test alone proves nothing about correctness.
  3. Does to_local() inside these functions stay correct when the tensor is
     row-sharded, i.e. when dim1 (derived from the LOCAL shape) no longer
     matches n_heads_arg (which stays GLOBAL)?

(3) is the one that matters for the FSDP fix, and it is the one a
single-process test would silently pass.
"""
import sys

import torch

# The adapter's two functions, lifted verbatim so this test needs no
# torchtitan import (and so it cannot be fooled by an import resolving to the
# wheel instead of the repo).


def permute(w, n_heads_arg, dim1=None, dim2=None):
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


def reverse_permute(w, n_heads_arg, dim1=None, dim2=None):
    if dim1 is None:
        dim1 = w.shape[0]
    if dim2 is None:
        dim2 = w.shape[1]
    return (
        w.view(n_heads_arg, 2, dim1 // n_heads_arg // 2, dim2)
        .transpose(1, 2)
        .reshape(dim1, dim2)
    )


def banner(s):
    print(f"\n{'=' * 70}\n{s}\n{'=' * 70}")


failures = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        failures.append(name)


# AGPT-2B attention geometry, read off the real config.
N_HEADS, N_KV, HEAD_DIM, DIM = 16, 16, 128, 2048

banner("1. ROUND TRIP: reverse_permute(permute(w)) == w")
for name, nh, d1 in [("wq", N_HEADS, N_HEADS * HEAD_DIM), ("wk", N_KV, N_KV * HEAD_DIM)]:
    w = torch.randn(d1, DIM, dtype=torch.float32)
    back = reverse_permute(permute(w, nh, d1, DIM), nh, d1, DIM)
    check(f"{name} round-trip exact", torch.equal(back, w),
          f"max|d|={(back - w).abs().max():.3e}")

banner("2. SEMANTICS: does it really map rotate_half <-> interleaved?")
# Build a weight whose every row is tagged with its own index, so the
# permutation is directly readable instead of inferred from a norm.
d1 = N_HEADS * HEAD_DIM
tag = torch.arange(d1, dtype=torch.float32).unsqueeze(1).repeat(1, DIM)
p = permute(tag, N_HEADS, d1, DIM)
src = p[:, 0].to(torch.int64)  # src[i] = which original row landed at row i

# Direction matters and is easy to get backwards. _permute is what to_hf()
# applies, so it must go Meta (interleaved) -> HF (half-split):
#   HF layout:   within a head, rows [0, h/2) pair with rows [h/2, h).
#   Meta layout: rows 2i and 2i+1 pair.
# Output row base+j (j < h/2) must therefore collect Meta's even member 2j,
# and output row base+h/2+j must collect Meta's odd member 2j+1.
# _reverse_permute (used by from_hf) is the opposite direction, and the
# round-trip in section 1 already pins the two together.
half = HEAD_DIM // 2
expected = torch.empty(d1, dtype=torch.int64)
for head in range(N_HEADS):
    base = head * HEAD_DIM
    for j in range(half):
        expected[base + j] = base + 2 * j             # first half <- even rows
        expected[base + half + j] = base + 2 * j + 1  # second half <- odd rows
check("_permute realizes the interleaved(Meta)->half-split(HF) map",
      torch.equal(src, expected),
      f"first 8 rows map {src[:8].tolist()}")

# A pure sanity bound: permutation must be a bijection over rows.
check("permutation is a bijection (no row lost or duplicated)",
      torch.equal(torch.sort(src).values, torch.arange(d1)))

banner("3. SHARDED: does to_local() stay correct when rows are split?")
# NOTE: these are EXPECTED-BAD. They document a real defect in _permute, not a
# broken test, so they are reported separately and do not set the exit code.
# See FINDINGS.md and test_fsdp_save_path.py (real FSDP2, same result).
# This models exactly what the FSDP fix does: the function receives only the
# LOCAL shard, recomputes dim1 from the local shape, but keeps n_heads_arg
# global. Simulate by slicing rows across N ranks and permuting each shard
# independently, then comparing against permuting the full tensor.
w_full = torch.randn(d1, DIM, dtype=torch.float32)
ref = permute(w_full, N_HEADS, d1, DIM)

for world in (2, 4, 8, 16, 48):
    if d1 % world:
        print(f"  [SKIP] world={world}: {d1} rows not divisible")
        continue
    rows = d1 // world
    try:
        shards = []
        for r in range(world):
            local = w_full[r * rows:(r + 1) * rows]
            # dim1 defaults to the LOCAL shape; n_heads_arg stays GLOBAL.
            shards.append(permute(local, N_HEADS))
        got = torch.cat(shards, dim=0)
        nbad = int((got != ref).any(dim=1).sum())
        tag = "matches unsharded" if nbad == 0 else f"CORRUPT {nbad}/{d1} rows"
        print(f"  [defect] world={world:2d} ({rows:4d} rows/shard): {tag}")
    except Exception as exc:
        print(f"  [defect] world={world:2d} ({rows:4d} rows/shard): "
              f"RAISES {type(exc).__name__}")

banner("VERDICT")
if failures:
    print(f"  {len(failures)} FAILED:")
    for f in failures:
        print(f"    - {f}")
    sys.exit(1)
print("  all checks passed")
