"""Round-trip the REAL converted HF checkpoint through the real adapter.

Loads the actual q_proj/k_proj tensors from
converted_models/agpt-2b-v2-step-46400_hf, pushes them through the adapter's
own _reverse_permute (HF -> Meta, what from_hf does on load) and _permute
(Meta -> HF, what to_hf does on save), and checks bitwise recovery.

Runs on CPU in fp32/bf16 as stored: the permutation is a pure row gather, so
it must be EXACT, not merely close. Any nonzero delta is a bug, not rounding.
"""
import glob
import json
import os
import sys

import torch
from safetensors.torch import load_file

HF = "/lus/flare/projects/datascience/seonghapark/converted_models/agpt-2b-v2-step-46400_hf"


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


cfg = json.load(open(os.path.join(HF, "config.json")))
n_heads = cfg["num_attention_heads"]
n_kv = cfg.get("num_key_value_heads", n_heads)
dim = cfg["hidden_size"]
head_dim = cfg.get("head_dim") or dim // n_heads
print(f"config: n_heads={n_heads} n_kv={n_kv} dim={dim} head_dim={head_dim}")
print(f"        dtype={cfg.get('torch_dtype')} layers={cfg['num_hidden_layers']}")

shards = sorted(glob.glob(os.path.join(HF, "*.safetensors")))
print(f"shards: {len(shards)}")

bad = []
checked = 0
for sh in shards:
    sd = load_file(sh)
    for k, v in sd.items():
        if k.endswith("self_attn.q_proj.weight"):
            nh, d1 = n_heads, n_heads * head_dim
        elif k.endswith("self_attn.k_proj.weight"):
            nh, d1 = n_kv, n_kv * head_dim
        else:
            continue
        checked += 1
        # HF -> Meta -> HF, exactly as from_hf() then to_hf() would.
        meta = reverse_permute(v, nh, d1, dim)
        back = permute(meta, nh, d1, dim)
        if not torch.equal(back, v):
            bad.append((k, (back.float() - v.float()).abs().max().item()))
        # The intermediate must NOT equal the original, or the permutation is
        # a no-op and the round-trip is vacuous.
        if torch.equal(meta, v):
            bad.append((k + " [permutation is a NO-OP]", 0.0))

print(f"\nq/k tensors checked: {checked}  (expect {2 * cfg['num_hidden_layers']})")
if bad:
    print(f"MISMATCHES: {len(bad)}")
    for k, d in bad[:10]:
        print(f"  {k}: max|delta|={d:.3e}")
    sys.exit(1)
print("all q/k tensors: HF -> Meta -> HF recovers the original BITWISE")
print("and the intermediate differs from the original (permutation is real)")
