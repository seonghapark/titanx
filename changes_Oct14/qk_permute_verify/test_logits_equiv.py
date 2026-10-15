"""End-to-end: do HF weights and adapter-converted weights give the same logits?

The round-trip test proves the permutation is self-consistent. It cannot
prove the permutation is the RIGHT one -- a wrong-but-invertible permutation
round-trips perfectly and still corrupts attention. Only running attention
under each convention can tell them apart.

Method, holding precision and execution conditions fixed (CPU, float32,
deterministic, identical inputs):

  branch A  HF weights  + HF rotate_half RoPE   (half-split pairing)
  branch B  permuted wt + interleaved complex RoPE (Meta pairing)

If the permutation is correct these are the same linear map composed with the
same rotation, so Q*K^T -- and the attention output -- must agree to fp32
noise. A control run with the permutation deliberately skipped must DISAGREE,
otherwise the test is insensitive and proves nothing.
"""
import json
import os
import sys

import torch

HF = "/lus/flare/projects/datascience/seonghapark/converted_models/agpt-2b-v2-step-46400_hf"
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)


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


# ---- the two RoPE conventions -------------------------------------------
def rope_hf(x, cos, sin):
    """HF rotate_half: pairs (i, i+half) within each head."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rot = torch.cat((-x2, x1), dim=-1)
    return x * cos + rot * sin


def rope_complex(x, freqs):
    """Meta/ComplexRoPE: pairs (2i, 2i+1), applied via view_as_complex."""
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    out = torch.view_as_real(xc * freqs).flatten(-2)
    return out.type_as(x)


cfg = json.load(open(os.path.join(HF, "config.json")))
n_heads = cfg["num_attention_heads"]
n_kv = cfg.get("num_key_value_heads", n_heads)
dim = cfg["hidden_size"]
head_dim = cfg.get("head_dim") or dim // n_heads
theta = cfg.get("rope_theta", 10000.0)
print(f"n_heads={n_heads} n_kv={n_kv} dim={dim} head_dim={head_dim} theta={theta}")

from safetensors.torch import load_file  # noqa: E402

import glob  # noqa: E402

sd = {}
for sh in sorted(glob.glob(os.path.join(HF, "*.safetensors"))):
    sd.update(load_file(sh))

wq = sd["model.layers.0.self_attn.q_proj.weight"].float()
wk = sd["model.layers.0.self_attn.k_proj.weight"].float()
print(f"wq {tuple(wq.shape)}  wk {tuple(wk.shape)}")

SEQ = 8
x = torch.randn(SEQ, dim, dtype=torch.float32)

# Rotation tables. Both conventions must describe the SAME rotation by angle
# m*theta_i for frequency i -- only the component pairing differs.
inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
pos = torch.arange(SEQ).float()
ang = torch.outer(pos, inv)                      # (SEQ, head_dim/2)
cos_h = torch.cat([ang.cos(), ang.cos()], -1)    # half-split duplication
sin_h = torch.cat([ang.sin(), ang.sin()], -1)
freqs = torch.polar(torch.ones_like(ang), ang)   # interleaved, complex


def project(w, nh):
    return (x @ w.T).view(SEQ, nh, head_dim)


def run(permuted: bool):
    if permuted:
        q = project(reverse_permute(wq, n_heads, n_heads * head_dim, dim), n_heads)
        k = project(reverse_permute(wk, n_kv, n_kv * head_dim, dim), n_kv)
        q = rope_complex(q, freqs.unsqueeze(1))
        k = rope_complex(k, freqs.unsqueeze(1))
    else:
        q = project(wq, n_heads)
        k = project(wk, n_kv)
        q = rope_hf(q, cos_h.unsqueeze(1), sin_h.unsqueeze(1))
        k = rope_hf(k, cos_h.unsqueeze(1), sin_h.unsqueeze(1))
    # GQA: repeat kv heads to match q heads, then attention scores.
    k = k.repeat_interleave(n_heads // n_kv, dim=1)
    return torch.einsum("qhd,khd->hqk", q, k) / (head_dim ** 0.5)


a = run(permuted=False)   # HF weights, HF RoPE
b = run(permuted=True)    # permuted weights, complex RoPE

d = (a - b).abs()
scale = a.abs().max().clamp(min=1e-12)
print(f"\nbranch A (HF wt + rotate_half)    scores range [{a.min():+.4f}, {a.max():+.4f}]")
print(f"branch B (perm wt + complex RoPE) scores range [{b.min():+.4f}, {b.max():+.4f}]")
print(f"max|A-B| = {d.max():.3e}   relative = {(d.max() / scale):.3e}")

TOL = 1e-4
ok = (d.max() / scale).item() < TOL

# Control: skip the permutation but still use complex RoPE. This MUST differ,
# or the comparison above is not actually sensitive to the permutation.
qc = project(wq, n_heads)
kc = project(wk, n_kv)
qc = rope_complex(qc, freqs.unsqueeze(1))
kc = rope_complex(kc, freqs.unsqueeze(1))
kc = kc.repeat_interleave(n_heads // n_kv, dim=1)
c = torch.einsum("qhd,khd->hqk", qc, kc) / (head_dim ** 0.5)
ctrl = (a - c).abs().max() / scale
print(f"control (NO permute + complex)   max|A-C| = {ctrl:.3e}  <- must be LARGE")

print()
print(f"  [{'PASS' if ok else 'FAIL'}] permuted branch matches HF within {TOL:.0e} relative")
print(f"  [{'PASS' if ctrl > 1e-2 else 'FAIL'}] test is sensitive (unpermuted control diverges)")
sys.exit(0 if (ok and ctrl > 1e-2) else 1)
