"""C+D: forward-pass equivalence before vs after conversion.

"Before" = the TorchTitan AgptModel built from the source DCP checkpoint.
"After"  = HF LlamaForCausalLM built from the converted safetensors.

Both run on CPU, in float32, in eval mode, on the *same* token ids, with the
same RoPE math. The flavor is agpt_2b (scaling="none") -- the flavor the
checkpoint was produced under, and the one whose positional math matches a
stock HF load, since config.json declares no rope_scaling.
"""
import sys, time, json
sys.path.insert(0, "/lus/flare/projects/datascience/seonghapark/xpu_launcher/xpu_torchtitan/torchtitan_repo")
sys.path.insert(0, "/lus/flare/projects/datascience/seonghapark/titanx/rope_complex_switch")
import torch

from dcp_model_only import load_model_only

BASE = "/lus/flare/projects/datascience/seonghapark"
HF = f"{BASE}/converted_models/agpt-2b-v2-step-46400_hf"

SEQ = 128
BATCH = 2
SEED = 1234
torch.manual_seed(SEED)
torch.set_grad_enabled(False)
torch.set_default_dtype(torch.float32)

# ---------------------------------------------------------------- config
from torchtitan.config.manager import ConfigManager
import dataclasses
argv = sys.argv
sys.argv = [argv[0], "--module", "agpt", "--config", "agpt_2b",
            "--hf_assets_path", HF]
cfg = ConfigManager().parse_args(["--module", "agpt", "--config", "agpt_2b",
                                  "--hf_assets_path", HF])
sys.argv = argv
mc = cfg.model_spec.model

# Shrink the RoPE cache; identical frequencies, just fewer positions.
for layer in mc.layers:
    if getattr(layer, "attention", None) is not None:
        layer.attention.rope = dataclasses.replace(layer.attention.rope,
                                                   max_seq_len=SEQ)
rope0 = mc.layers[0].attention.rope
attn0 = mc.layers[0].attention
head_dim_c = attn0.head_dim or mc.dim // attn0.n_heads
print(f"flavor agpt_2b | rope={type(rope0).__qualname__} scaling={rope0.scaling} "
      f"theta={rope0.theta} factor={rope0.rope_factor} max_seq_len={rope0.max_seq_len}")

# ---------------------------------------------------------------- "before"
print("\n=== BEFORE: TorchTitan model from DCP ===", flush=True)
t0 = time.time()
native = load_model_only()
print(f"  DCP loaded, {len(native)} model tensors, {time.time()-t0:.1f}s", flush=True)

with torch.device("cpu"):
    tt = mc.build()
tt.init_states(buffer_device=torch.device("cpu"))
native = {k: v.to(torch.float32) for k, v in native.items()}
missing, unexpected = tt.load_state_dict(native, strict=False)
real_missing = [k for k in missing if "freqs" not in k and "rope" not in k.lower()]
print(f"  load_state_dict: {len(real_missing)} missing (non-buffer), "
      f"{len(unexpected)} unexpected")
if real_missing[:5]:
    print(f"    missing: {real_missing[:5]}")
if list(unexpected)[:5]:
    print(f"    unexpected: {list(unexpected)[:5]}")
tt = tt.to(torch.float32).eval()

vocab = mc.vocab_size
g = torch.Generator().manual_seed(SEED)
tokens = torch.randint(0, vocab, (BATCH, SEQ), generator=g)

t0 = time.time()
out_tt = tt(tokens)
logits_tt = (out_tt[0] if isinstance(out_tt, (tuple, list)) else out_tt).float()
print(f"  forward {tuple(logits_tt.shape)} in {time.time()-t0:.1f}s", flush=True)
del tt, native
import gc; gc.collect()

# ---------------------------------------------------------------- "after"
print("\n=== AFTER: HF LlamaForCausalLM from converted safetensors ===", flush=True)
from transformers import LlamaForCausalLM
t0 = time.time()
hf = LlamaForCausalLM.from_pretrained(HF, dtype=torch.float32,
                                      attn_implementation="eager")
hf = hf.to(torch.float32).eval()
print(f"  loaded in {time.time()-t0:.1f}s", flush=True)
hr = hf.config
# transformers 5.x nests RoPE settings under `rope_parameters`; there is no
# top-level `rope_theta`/`rope_scaling` attribute any more.
rope_params = hr.to_dict().get("rope_parameters")
rot = hf.model.rotary_emb
print(f"  HF rope_parameters={rope_params} max_pos={hr.max_position_embeddings}")
print(f"  HF rotary={type(rot).__name__} attention_scaling={rot.attention_scaling}")
assert rope_params and rope_params.get("rope_type") == "default", (
    f"expected an unscaled HF RoPE, got {rope_params}")
assert float(rot.attention_scaling) == 1.0, "HF applied an mscale; TT side has none"
t0 = time.time()
logits_hf = hf(tokens).logits.float()
print(f"  forward {tuple(logits_hf.shape)} in {time.time()-t0:.1f}s", flush=True)

# ------------------------------------------------- RoPE frequency identity
# The logit comparison only means something if both stacks rotate by the same
# angles. ComplexRoPE stores exp(i*theta) directly; HF stores inv_freq. Compare
# the underlying frequencies rather than the cache layouts, which differ by
# construction (interleaved vs rotate_half).
print("\n=== RoPE frequency cross-check (TorchTitan vs HF) ===")
hf_inv = rot.inv_freq.double()
tt_inv = 1.0 / (rope0.theta ** (torch.arange(0, head_dim_c, 2).double() / head_dim_c))
fd = (hf_inv - tt_inv).abs().max().item()
print(f"  head_dim={head_dim_c} theta={rope0.theta}")
print(f"  max |inv_freq_HF - inv_freq_TT| = {fd:.3e}")
freq_ok = fd < 1e-12
print(f"  {'identical' if freq_ok else 'MISMATCH -- positional math differs'}")

# ---------------------------------------------------------------- compare
print("\n=== D: logit comparison (fp32, CPU, eval, identical tokens) ===")
diff = (logits_tt - logits_hf).abs()
scale = logits_tt.abs().max().item()
amax, amean = diff.max().item(), diff.mean().item()
rel = amax / max(scale, 1e-12)
print(f"  logit magnitude (max|before|) : {scale:.4f}")
print(f"  max  abs diff                 : {amax:.3e}")
print(f"  mean abs diff                 : {amean:.3e}")
print(f"  max  rel diff (vs max|logit|) : {rel:.3e}")

cos = torch.nn.functional.cosine_similarity(
    logits_tt.reshape(-1), logits_hf.reshape(-1), dim=0).item()
print(f"  cosine similarity             : {cos:.10f}")

top1 = (logits_tt.argmax(-1) == logits_hf.argmax(-1)).float().mean().item()
print(f"  argmax agreement              : {top1*100:.4f}%")

lp_tt = torch.log_softmax(logits_tt, -1)
lp_hf = torch.log_softmax(logits_hf, -1)
kl = torch.nn.functional.kl_div(lp_hf, lp_tt, log_target=True,
                                reduction="batchmean").item()
print(f"  KL(before||after) per token   : {kl:.3e}")

TOL_REL, TOL_KL = 1e-4, 1e-6
ok = freq_ok and rel < TOL_REL and top1 == 1.0 and abs(kl) < TOL_KL
print(f"\n  tolerance: rel<{TOL_REL:g}, argmax 100%, KL<{TOL_KL:g}, "
      f"identical RoPE frequencies")
print("RESULT:", "CONVERSION LOSSLESS (within fp32 tolerance)" if ok
      else "OUT OF TOLERANCE -- investigate")
