"""Does permute(W) + HF rotate_half == W + ComplexRoPE interleaved?

This is the mathematical claim the whole conversion rests on, tested on
synthetic weights with no checkpoint involved. If it fails here, the
conversion's permutation is wrong in principle. If it passes here but the real
models diverge, the fault is elsewhere (how I drive the model, or a non-RoPE
difference).

Pure torch + the repo's ComplexRoPE; no HF model load, so it runs in seconds.
"""
import sys, math
sys.path.insert(0, "/lus/flare/projects/datascience/seonghapark/xpu_launcher/xpu_torchtitan/torchtitan_repo")
import torch

torch.manual_seed(0)
torch.set_grad_enabled(False)

D, SEQ, THETA = 128, 16, 50000.0

# ---------------------------------------------------------------- HF side
def hf_rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def hf_apply(x, cos, sin):
    return x * cos + hf_rotate_half(x) * sin

inv_freq = 1.0 / (THETA ** (torch.arange(0, D, 2).double() / D))
t = torch.arange(SEQ).double()
freqs = torch.outer(t, inv_freq)                      # (SEQ, D/2)
emb = torch.cat((freqs, freqs), dim=-1)               # (SEQ, D)  HF duplicates
cos, sin = emb.cos(), emb.sin()

# ---------------------------------------------------------------- TT side
from torchtitan.models.common.rope import ComplexRoPE
rope = ComplexRoPE.Config(dim=D, max_seq_len=SEQ, theta=THETA,
                          scaling="none").build()
rope.init_states(buffer_device=torch.device("cpu"))

# ---------------------------------------------------------------- permutation
# dcp_to_hf's _permute: interleaved -> rotate_half layout, per head.
def permute(w, n_heads, d):
    # w: (n_heads*d, in)
    return (w.view(n_heads, d // 2, 2, -1)
             .transpose(1, 2)
             .reshape(n_heads * d, -1))

# ---------------------------------------------------------------- experiment
# One head. x is the hidden vector; W the projection.
IN = 64
W = torch.randn(D, IN).double()
x = torch.randn(1, SEQ, IN).double()

# TT path: project with W (Meta layout), rotate with ComplexRoPE (interleaved).
q_tt = (x @ W.T).view(1, SEQ, 1, D).float()
k_tt = q_tt.clone()
q_rot_tt, _ = rope(q_tt, k_tt)
q_rot_tt = q_rot_tt.double().view(SEQ, D)

# HF path: project with permute(W) (HF layout), rotate with rotate_half.
Wp = permute(W, 1, D)
q_hf = (x @ Wp.T).view(SEQ, D)
q_rot_hf = hf_apply(q_hf, cos, sin)

# To compare, map the TT result into HF row order with the same permutation.
q_rot_tt_as_hf = permute(q_rot_tt.T.contiguous(), 1, D).T.contiguous()

diff = (q_rot_tt_as_hf - q_rot_hf).abs().max().item()
scale = q_rot_hf.abs().max().item()
print(f"max |permute(ComplexRoPE(Wx)) - rotate_half(permute(W)x)| = {diff:.3e}")
print(f"  (signal magnitude {scale:.3f}, relative {diff/scale:.3e})")
print("VERDICT:", "EQUIVALENT" if diff / scale < 1e-6
      else "NOT EQUIVALENT -- the permutation does not match the rotation")

# Sanity: the permutation must actually do something.
print(f"\nsanity: permute changes W?        {not torch.equal(Wp, W)}")
print(f"sanity: rope changes q?           {not torch.allclose(q_rot_tt.float(), q_tt.view(SEQ, D))}")

# ---------------------------------------------------------------- GQA head count
# The real k_proj has n_kv_heads=4, not n_heads=16. _permute must be applied
# with the tensor's OWN head count; using n_heads for k_proj would reshape
# (4*128, in) as if it were 16 heads and scramble rows. Part B could not catch
# this: it compared the adapter against a converter that uses the same adapter.
print("\n=== GQA: does the head count used for k_proj matter? ===")
NK, NQ = 4, 16
Wk = torch.randn(NK * D, IN).double()
right = permute(Wk, NK, D)
wrong = permute(Wk, NQ, D)
print(f"  permute(Wk, n_kv=4) == permute(Wk, n_heads=16)? "
      f"{torch.equal(right, wrong)}")
print(f"  max difference between the two: {(right - wrong).abs().max().item():.3e}")
print("  -> head count is load-bearing; it must be the tensor's own."
      if not torch.equal(right, wrong) else
      "  -> indistinguishable at this shape")
