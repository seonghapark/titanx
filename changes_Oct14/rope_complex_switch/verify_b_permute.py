"""B: Verify the Q/K permutation on the REAL stored tensors.

Stronger than a synthetic round trip: this checks that the bytes actually on
disk in the converted HF file are the permutation of the bytes in the source
DCP checkpoint -- i.e. that the conversion that ran was correct -- and that
from_hf's reverse permute recovers the DCP values exactly.

Run under the PRETRAINING flavor agpt_2b, which is the flavor the checkpoint
was produced under and the one the conversion used.
"""
import re
import sys

sys.path.insert(0, "/lus/flare/projects/datascience/seonghapark/xpu_launcher/xpu_torchtitan/torchtitan_repo")
sys.path.insert(0, "/lus/flare/projects/datascience/seonghapark/titanx/rope_complex_switch")

import torch
from safetensors.torch import load_file

from dcp_model_only import load_model_only

BASE = "/lus/flare/projects/datascience/seonghapark"
HF = f"{BASE}/converted_models/agpt-2b-v2-step-46400_hf"

torch.set_grad_enabled(False)

print("=== loading DCP model tensors ===", flush=True)
native = load_model_only()
print(f"  {len(native)} tensors", flush=True)

print("\n=== building adapter (flavor agpt_2b) ===", flush=True)
from torchtitan.config.manager import ConfigManager

argv = sys.argv
args = ["--module", "agpt", "--config", "agpt_2b", "--hf_assets_path", HF]
sys.argv = [argv[0], *args]
try:
    cfg = ConfigManager().parse_args(args)
finally:
    sys.argv = argv
spec = cfg.model_spec
adapter = spec.state_dict_adapter(spec.model, HF)

attn = adapter.model_config.layers[0].attention
n_heads = attn.n_heads
n_kv = attn.n_kv_heads if attn.n_kv_heads is not None else n_heads
dim = adapter.model_config.dim
head_dim = attn.head_dim or dim // n_heads
kv_dim = head_dim * n_kv
nlayers = len(adapter.model_config.layers)
print(f"  rope={type(attn.rope).__qualname__} "
      f"needs_qk_permute={adapter._needs_qk_permute()}")
print(f"  n_heads={n_heads} n_kv={n_kv} dim={dim} head_dim={head_dim} "
      f"layers={nlayers}")

print("\n=== loading converted HF safetensors ===", flush=True)
import json

idx = json.load(open(f"{HF}/model.safetensors.index.json"))
hf = {}
for shard in sorted(set(idx["weight_map"].values())):
    hf.update(load_file(f"{HF}/{shard}"))
print(f"  {len(hf)} tensors, dtype={next(iter(hf.values())).dtype}")

# ------------------------------------------------------------------ Q/K
print("\n=== B1. Q/K: disk == _permute(dcp), and _reverse_permute recovers ===")
print(f"{'tensor':>26} | {'to_hf==disk':>11} | {'from_hf==dcp':>12} | {'reordered':>9}")
print("-" * 70)
bad = []
for L in range(nlayers):
    for which, nh, extra in (("q", n_heads, ()), ("k", n_kv, (kv_dim, dim))):
        nat_key = f"layers.{L}.attention.qkv_linear.w{which}.weight"
        hf_key = f"model.layers.{L}.self_attn.{which}_proj.weight"
        if nat_key not in native or hf_key not in hf:
            bad.append(f"{nat_key}: key missing")
            continue
        w_dcp = native[nat_key].to(torch.float32)
        w_hf = hf[hf_key].to(torch.float32)
        fwd_ok = torch.equal(adapter._permute(w_dcp, nh, *extra), w_hf)
        rev_ok = torch.equal(adapter._reverse_permute(w_hf, nh, *extra), w_dcp)
        moved = not torch.equal(w_dcp, w_hf)
        if not (fwd_ok and rev_ok and moved):
            bad.append(f"L{L}.{which}: fwd={fwd_ok} rev={rev_ok} moved={moved}")
        if L < 2 or not (fwd_ok and rev_ok and moved):
            print(f"{f'L{L}.{which}_proj':>26} | {str(fwd_ok):>11} | "
                  f"{str(rev_ok):>12} | {str(moved):>9}")
print(f"  ... {nlayers} layers x (q, k) = {nlayers * 2} tensors checked")

# ----------------------------------------------------------- everything else
print("\n=== B2. non-permuted tensors: DCP vs HF bitwise ===")
to_hf_map = {v: k for k, v in adapter.from_hf_map.items() if v is not None}
checked = mismatched = 0
unmapped = []
for nat_key, t in native.items():
    abstract = re.sub(r"(\d+)", "{}", nat_key, count=1)
    hf_abstract = to_hf_map.get(nat_key) or to_hf_map.get(abstract)
    if hf_abstract is None:
        unmapped.append(nat_key)
        continue
    nums = re.findall(r"\d+", nat_key)
    hf_key = hf_abstract.replace("{}", nums[0], 1) if "{}" in hf_abstract else hf_abstract
    if hf_key not in hf:
        unmapped.append(f"{nat_key} -> {hf_key} (absent in HF)")
        continue
    if ".q_proj." in hf_key or ".k_proj." in hf_key:
        continue
    checked += 1
    if not torch.equal(t.to(torch.float32), hf[hf_key].to(torch.float32)):
        mismatched += 1
        if mismatched <= 5:
            d = (t.to(torch.float32) - hf[hf_key].to(torch.float32)).abs().max()
            print(f"  DIFF {hf_key}  max|d|={d:.3e}")
print(f"  {checked} non-QK tensors compared, {mismatched} differ")
if unmapped:
    print(f"  {len(unmapped)} unmapped: {unmapped[:5]}")

# ------------------------------------------------- the permutation's shape
print("\n=== B3. what the permutation actually does (layer 0, head 0) ===")
w = native["layers.0.attention.qkv_linear.wq.weight"].to(torch.float32)
p = adapter._permute(w, n_heads)
head0_src = w[:head_dim]
head0_dst = p[:head_dim]
mapping = []
for i in range(min(8, head_dim)):
    j = (head0_src == head0_dst[i]).all(dim=1).nonzero()
    mapping.append(f"{i}<-{j[0].item() if len(j) else '?'}")
print(f"  head_dim={head_dim}; dst row <- src row: {', '.join(mapping)} ...")
print("  (interleaved->rotate_half: dst i <- src 2i for i<d/2, else 2(i-d/2)+1)")

ok = not bad and mismatched == 0
print("\nRESULT:", "PERMUTATION VERIFIED ON REAL WEIGHTS" if ok
      else f"PROBLEMS: bad={bad[:5]} mismatched={mismatched}")
