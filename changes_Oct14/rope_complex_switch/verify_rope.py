"""Verify ComplexRoPE(permuted W) == CosSinRoPE(HF W) for YaRN params.

If this holds, switching the agpt YaRN flavors from cos_sin to complex is
weight-layout-safe PROVIDED the state_dict_adapter permutation is applied,
which _needs_qk_permute() already keys off the RoPE class.
"""
import math, sys
import torch
from torchtitan.models.common.rope import ComplexRoPE, CosSinRoPE

torch.manual_seed(0)

HEAD_DIM, SEQ, BATCH, NH = 128, 64, 2, 4
COMMON = dict(dim=HEAD_DIM, max_seq_len=SEQ, theta=50000.0)

def hf_to_meta_permute(w, n_heads, head_dim):
    """HF half-split -> Meta interleaved, the adapter's _permute."""
    d0 = w.shape[0]
    return (w.view(n_heads, 2, head_dim // 2, *w.shape[1:])
             .transpose(1, 2).reshape(d0, *w.shape[1:]))

def report(tag, yarn_kwargs):
    cs = CosSinRoPE(CosSinRoPE.Config(**COMMON, **yarn_kwargs))
    cx = ComplexRoPE(ComplexRoPE.Config(**COMMON, **yarn_kwargs))

    # A "weight row vector" per head: simulate q = W x for one token.
    q_hf = torch.randn(BATCH, SEQ, NH, HEAD_DIM, dtype=torch.float32)
    k_hf = torch.randn(BATCH, SEQ, NH, HEAD_DIM, dtype=torch.float32)

    # cos_sin consumes HF layout directly
    q_cs, k_cs = cs(q_hf.clone(), k_hf.clone())

    # complex consumes Meta-interleaved layout: permute the feature axis
    def perm_feat(t):
        b, s, n, d = t.shape
        return (t.reshape(b, s, n, 2, d // 2).transpose(-1, -2).reshape(b, s, n, d))
    def unperm_feat(t):
        b, s, n, d = t.shape
        return (t.reshape(b, s, n, d // 2, 2).transpose(-1, -2).reshape(b, s, n, d))

    q_cx, k_cx = cx(perm_feat(q_hf.clone()), perm_feat(k_hf.clone()))
    q_cx, k_cx = unperm_feat(q_cx), unperm_feat(k_cx)

    dq = (q_cs - q_cx).abs().max().item()
    dk = (k_cs - k_cx).abs().max().item()
    print(f"{tag:28s} max|dq|={dq:.3e} max|dk|={dk:.3e}  ->", "MATCH" if max(dq,dk) < 2e-4 else "MISMATCH")
    return max(dq, dk) < 2e-4

ok = True
ok &= report("no scaling", dict(scaling="none"))
# YaRN at factor 1.0 (training at/below pretrain len)
ok &= report("yarn factor=1.0", dict(scaling="yarn", rope_factor=1.0,
             original_seq_len=8192, beta_fast=1.0, beta_slow=32.0))
print()
print("=== mscale parity (the stated reason for cos_sin) ===")
for factor in (1.0, 2.0, 4.0, 32.0):
    cfg = dict(dim=HEAD_DIM, max_seq_len=262144, theta=50000.0, scaling="yarn",
               rope_factor=factor, original_seq_len=8192, beta_fast=1.0, beta_slow=32.0)
    cx = ComplexRoPE(ComplexRoPE.Config(**cfg))
    expect = 1.0 if factor <= 1.0 else 0.1*math.log(factor)+1.0
    got = cx._get_mscale()
    print(f"  factor={factor:6.1f}  ComplexRoPE mscale={got:.6f}  expected={expect:.6f}",
          "OK" if abs(got-expect) < 1e-9 else "FAIL")
    ok &= abs(got-expect) < 1e-9

print()
print("RESULT:", "ALL CHECKS PASS" if ok else "FAILURES PRESENT")
sys.exit(0 if ok else 1)
