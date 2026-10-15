"""Layer-0 internals: where exactly inside the first attention do TT and HF part?

diag_layerwise.py showed embeddings bitwise identical and layer-0 attention
output differing by 6e-2 absolute -- but it never printed the signal magnitude,
so that number alone does not say whether it is a real error or rounding.

This captures, at layer 0, with relative scale:
  q/k/v projections (before RoPE)  -- tests the permutation on live activations
  q/k after RoPE                   -- tests the rotation
  attention output                 -- tests GQA/masking/softmax

Q and K live in different row orders in the two stacks (that is the whole point
of the permutation), so they are compared after mapping TT's rows into HF order.
"""
import sys, dataclasses
sys.path.insert(0, "/lus/flare/projects/datascience/seonghapark/xpu_launcher/xpu_torchtitan/torchtitan_repo")
sys.path.insert(0, "/lus/flare/projects/datascience/seonghapark/titanx/rope_complex_switch")
import torch

from dcp_model_only import load_model_only

BASE = "/lus/flare/projects/datascience/seonghapark"
HF = f"{BASE}/converted_models/agpt-2b-v2-step-46400_hf"
SEQ, BATCH, SEED = 16, 1, 1234
torch.manual_seed(SEED); torch.set_grad_enabled(False)

NH, NKV, HD = 16, 4, 128


def rel(a, b, label):
    a, b = a.float(), b.float()
    m = max(a.abs().max().item(), 1e-12)
    dd = (a - b).abs().max().item()
    print(f"  {label:<34} max|diff|={dd:>10.3e}  scale={m:>8.3f}  rel={dd/m:>10.3e}")
    return dd / m


def interleaved_to_hf(x_BLNH):
    """TT (interleaved pairs 2i,2i+1) -> HF (pairs i, i+d/2), per head."""
    B, L, N, H = x_BLNH.shape
    return torch.cat([x_BLNH[..., 0::2], x_BLNH[..., 1::2]], dim=-1)


# ---------------------------------------------------------------- build TT
from torchtitan.config.manager import ConfigManager
cfg = ConfigManager().parse_args(["--module", "agpt", "--config", "agpt_2b",
                                  "--hf_assets_path", HF])
mc = cfg.model_spec.model
for layer in mc.layers:
    if getattr(layer, "attention", None) is not None:
        layer.attention.rope = dataclasses.replace(layer.attention.rope,
                                                   max_seq_len=SEQ)
native = {k: v.to(torch.float32) for k, v in load_model_only().items()}
with torch.device("cpu"):
    tt = mc.build()
tt.init_states(buffer_device=torch.device("cpu"))
tt.load_state_dict(native, strict=False)
tt = tt.eval()

from transformers import LlamaForCausalLM
hf = LlamaForCausalLM.from_pretrained(HF, dtype=torch.float32,
                                      attn_implementation="eager").eval()

g = torch.Generator().manual_seed(SEED)
tokens = torch.randint(0, mc.vocab_size, (BATCH, SEQ), generator=g)

# ---------------------------------------------------------------- weights
tt_l0 = list(tt.layers.values())[0]
hf_l0 = hf.model.layers[0]
print("=== layer-0 weights as loaded in memory ===")
# TT packs qkv in one linear on some builds; find the real projection weights.
tt_sd = {k: v for k, v in tt.state_dict().items() if k.startswith("layers.0.")}
for k in sorted(tt_sd):
    if any(t in k for t in ("wq", "wk", "q_proj", "k_proj")):
        print(f"  TT {k:<46} {tuple(tt_sd[k].shape)}")
print(f"  HF q_proj {tuple(hf_l0.self_attn.q_proj.weight.shape)}  "
      f"k_proj {tuple(hf_l0.self_attn.k_proj.weight.shape)}")

# ---------------------------------------------------------------- capture
cap = {}
tt_l0.attention_norm.register_forward_hook(
    lambda m, i, o: cap.__setitem__("tt_norm", o.detach()))
hf_l0.input_layernorm.register_forward_hook(
    lambda m, i, o: cap.__setitem__("hf_norm", o.detach()))
tt_l0.attention.register_forward_hook(
    lambda m, i, o: cap.__setitem__("tt_attn", o.detach()))
hf_l0.self_attn.register_forward_hook(
    lambda m, i, o: cap.__setitem__(
        "hf_attn", (o[0] if isinstance(o, tuple) else o).detach()))
hf_l0.self_attn.q_proj.register_forward_hook(
    lambda m, i, o: cap.__setitem__("hf_q", o.detach()))
hf_l0.self_attn.k_proj.register_forward_hook(
    lambda m, i, o: cap.__setitem__("hf_k", o.detach()))
hf_l0.self_attn.v_proj.register_forward_hook(
    lambda m, i, o: cap.__setitem__("hf_v", o.detach()))

# TT's rope is called inside Attention.forward; wrap it to grab both sides.
orig_rope = tt_l0.attention.rope.forward
def rope_spy(q, k, positions=None):
    cap["tt_q_pre"], cap["tt_k_pre"] = q.detach().clone(), k.detach().clone()
    out = orig_rope(q, k, positions)
    cap["tt_q_post"], cap["tt_k_post"] = out[0].detach(), out[1].detach()
    return out
tt_l0.attention.rope.forward = rope_spy

print("\nrunning both forwards...", flush=True)
lt = tt(tokens)
lh = hf(tokens).logits

# ---------------------------------------------------------------- compare
print("\n=== layer 0, relative differences ===")
rel(cap["tt_norm"], cap["hf_norm"], "input layernorm out")

B, L = BATCH, SEQ
hf_q = cap["hf_q"].view(B, L, NH, HD)
hf_k = cap["hf_k"].view(B, L, NKV, HD)
rel(interleaved_to_hf(cap["tt_q_pre"]), hf_q, "q_proj (TT rows -> HF order)")
rel(interleaved_to_hf(cap["tt_k_pre"]), hf_k, "k_proj (TT rows -> HF order)")
print("    ^ if these match, the stored permutation is right on live data")

rel(interleaved_to_hf(cap["tt_q_post"]), hf_q, "q after TT RoPE vs HF pre-RoPE")
print("    ^ expected to DIFFER (HF applies its own rotation later)")

rel(cap["tt_attn"], cap["hf_attn"], "attention output")
rel(lt, lh, "final logits")
