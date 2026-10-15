"""The real risk: do the two backends agree when YaRN interpolation is ACTIVE
(rope_factor > 1)? They compute the correction range with different formulas.
"""
import math, torch
from torchtitan.models.common.rope import ComplexRoPE, CosSinRoPE

torch.manual_seed(0)
HEAD_DIM, SEQ, BATCH, NH = 128, 64, 2, 4

def perm_feat(t):
    b,s,n,d = t.shape
    return t.reshape(b,s,n,2,d//2).transpose(-1,-2).reshape(b,s,n,d)
def unperm_feat(t):
    b,s,n,d = t.shape
    return t.reshape(b,s,n,d//2,2).transpose(-1,-2).reshape(b,s,n,d)

print(f"{'config':>34} | {'max|dq|':>10} | verdict")
print("-"*62)
worst = 0.0
for factor, orig in [(2.0, 8192), (4.0, 8192), (32.0, 8192)]:
    cfg = dict(dim=HEAD_DIM, max_seq_len=int(orig*factor), theta=50000.0,
               scaling="yarn", rope_factor=factor, original_seq_len=orig,
               beta_fast=1.0, beta_slow=32.0)
    try:
        cs = CosSinRoPE(CosSinRoPE.Config(**cfg))
        cx = ComplexRoPE(ComplexRoPE.Config(**cfg))
    except Exception as e:
        print(f"{'factor=%s'%factor:>34} | {'-':>10} | CONSTRUCT FAIL: {type(e).__name__}: {e}")
        continue
    q = torch.randn(BATCH, SEQ, NH, HEAD_DIM)
    k = torch.randn(BATCH, SEQ, NH, HEAD_DIM)
    q_cs, k_cs = cs(q.clone(), k.clone())
    q_cx, k_cx = cx(perm_feat(q.clone()), perm_feat(k.clone()))
    q_cx, k_cx = unperm_feat(q_cx), unperm_feat(k_cx)
    d = max((q_cs-q_cx).abs().max().item(), (k_cs-k_cx).abs().max().item())
    worst = max(worst, d)
    print(f"{'factor=%-5s orig=%s'%(factor,orig):>34} | {d:10.3e} | {'MATCH' if d<2e-4 else 'DIFFER'}")

print()
print("Interpretation: a DIFFER here means the two backends implement YaRN")
print("frequency interpolation differently, so switching flavors changes the")
print("model's positional math (not just weight layout).")
print(f"worst={worst:.3e}")
