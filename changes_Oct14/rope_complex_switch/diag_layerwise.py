"""Localize the TT-vs-HF logit divergence found in verify_c_logits.py.

Part B proved the weights are the same model (87/87 non-QK bitwise identical,
Q/K permutation exact against disk both directions). So the divergence is in the
computation. This walks the two stacks layer by layer and reports where they
first separate.

Small shapes: HF eager attention at seq 128 took 257s; seq 32 batch 1 is plenty
to localize a structural difference.
"""
import sys, time, dataclasses
sys.path.insert(0, "/lus/flare/projects/datascience/seonghapark/xpu_launcher/xpu_torchtitan/torchtitan_repo")
sys.path.insert(0, "/lus/flare/projects/datascience/seonghapark/titanx/rope_complex_switch")
import torch

from dcp_model_only import load_model_only

BASE = "/lus/flare/projects/datascience/seonghapark"
HF = f"{BASE}/converted_models/agpt-2b-v2-step-46400_hf"

SEQ, BATCH, SEED = 32, 1, 1234
torch.manual_seed(SEED)
torch.set_grad_enabled(False)
torch.set_default_dtype(torch.float32)


def d(a, b):
    return (a.float() - b.float()).abs().max().item()


# ---------------------------------------------------------------- TT side
from torchtitan.config.manager import ConfigManager
cfg = ConfigManager().parse_args(["--module", "agpt", "--config", "agpt_2b",
                                  "--hf_assets_path", HF])
mc = cfg.model_spec.model
for layer in mc.layers:
    if getattr(layer, "attention", None) is not None:
        layer.attention.rope = dataclasses.replace(layer.attention.rope,
                                                   max_seq_len=SEQ)
attn0 = mc.layers[0].attention
print(f"TT attention: inner={type(attn0.inner_attention).__qualname__} "
      f"n_heads={attn0.n_heads} n_kv={attn0.n_kv_heads} head_dim={attn0.head_dim} "
      f"qk_norm={attn0.qk_norm is not None}")
print(f"TT rope: {type(mc.layers[0].attention.rope).__qualname__} "
      f"scaling={mc.layers[0].attention.rope.scaling}")

native = load_model_only()
native = {k: v.to(torch.float32) for k, v in native.items()}
with torch.device("cpu"):
    tt = mc.build()
tt.init_states(buffer_device=torch.device("cpu"))
tt.load_state_dict(native, strict=False)
tt = tt.to(torch.float32).eval()

# ---------------------------------------------------------------- HF side
from transformers import LlamaForCausalLM
hf = LlamaForCausalLM.from_pretrained(HF, dtype=torch.float32,
                                      attn_implementation="eager").eval()

g = torch.Generator().manual_seed(SEED)
tokens = torch.randint(0, mc.vocab_size, (BATCH, SEQ), generator=g)

# ---------------------------------------------------------------- capture
tt_h, hf_h = {}, {}
tt_layers = list(tt.layers.values())
for i, layer in enumerate(tt_layers):
    layer.register_forward_hook(
        lambda m, inp, out, i=i: tt_h.__setitem__(i, out.detach()))
    layer.attention.register_forward_hook(
        lambda m, inp, out, i=i: tt_h.__setitem__(f"attn{i}", out.detach()))
tt.tok_embeddings.register_forward_hook(
    lambda m, inp, out: tt_h.__setitem__("emb", out.detach()))

for i, layer in enumerate(hf.model.layers):
    layer.register_forward_hook(
        lambda m, inp, out, i=i: hf_h.__setitem__(
            i, (out[0] if isinstance(out, tuple) else out).detach()))
    layer.self_attn.register_forward_hook(
        lambda m, inp, out, i=i: hf_h.__setitem__(
            f"attn{i}", (out[0] if isinstance(out, tuple) else out).detach()))
hf.model.embed_tokens.register_forward_hook(
    lambda m, inp, out: hf_h.__setitem__("emb", out.detach()))

print("\nrunning TT forward...", flush=True)
t0 = time.time(); lt = tt(tokens); print(f"  {time.time()-t0:.1f}s", flush=True)
print("running HF forward...", flush=True)
t0 = time.time(); lh = hf(tokens).logits; print(f"  {time.time()-t0:.1f}s", flush=True)

# ---------------------------------------------------------------- compare
print("\n=== where do they diverge? ===")
print(f"  embeddings            : {d(tt_h['emb'], hf_h['emb']):.3e}")
print(f"  {'layer':>5} | {'attn out':>11} | {'block out':>11}")
print("  " + "-" * 34)
first = None
for i in range(len(tt_layers)):
    da = d(tt_h[f"attn{i}"], hf_h[f"attn{i}"])
    db = d(tt_h[i], hf_h[i])
    print(f"  {i:>5} | {da:>11.3e} | {db:>11.3e}")
    if first is None and da > 1e-4:
        first = i
print(f"\n  final logits          : {d(lt, lh):.3e}")
print(f"  first diverging layer : {first}")
