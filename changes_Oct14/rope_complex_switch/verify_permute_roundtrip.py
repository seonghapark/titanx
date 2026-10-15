"""Confirm the now-enabled Q/K permutation is a clean involution-pair, i.e.
from_hf(to_hf(w)) == w for the yarn flavor. If this did NOT hold, every
HF<->DCP round trip under the new flavor would degrade weights.
"""
import torch
from torchtitan.models.agpt import config_registry as reg
from torchtitan.models.common.rope import ComplexRoPE

cfg = reg.agpt_2b_yarn()
spec = cfg.model_spec
ad = spec.state_dict_adapter(spec.model, None)
rope = spec.model.layers[0].attention.rope

print(f"flavor rope       = {type(rope).__qualname__}")
print(f"needs_qk_permute  = {ad._needs_qk_permute()}")
assert issubclass(type(rope), ComplexRoPE.Config)
assert ad._needs_qk_permute()

# Mirror exactly how to_hf/from_hf reach these values and call _permute.
m = spec.model
attn = m.layers[0].attention
n_heads = attn.n_heads
n_kv = attn.n_kv_heads if attn.n_kv_heads is not None else n_heads
dim = m.dim
head_dim = attn.head_dim or dim // n_heads
kv_dim = head_dim * n_kv

torch.manual_seed(0)
wq = torch.randn(n_heads * head_dim, dim)
wk = torch.randn(kv_dim, dim)

print(f"n_heads={n_heads} n_kv_heads={n_kv} dim={dim} head_dim={head_dim}")

ok = True
# q_proj uses the default dims; k_proj passes kv_dim explicitly, as to_hf does.
for name, w, nh, extra in (("q_proj", wq, n_heads, ()),
                           ("k_proj", wk, n_kv, (kv_dim, dim))):
    fwd = ad._permute(w, nh, *extra)
    back = ad._reverse_permute(fwd, nh, *extra)
    exact = torch.equal(back, w)
    moved = not torch.equal(fwd, w)
    ok &= exact and moved
    print(f"{name}: permute changed weights={moved}  reverse_permute restores exactly={exact}")

print("\nRESULT:", "ROUND TRIP EXACT" if ok else "ROUND TRIP BROKEN")
