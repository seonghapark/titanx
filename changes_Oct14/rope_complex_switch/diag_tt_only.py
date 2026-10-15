"""Is the conversion semantically sound, judged without HF in the loop?

The conversion's claim is:
    ComplexRoPE model holding W   ==   rotate_half model holding permute(W)

TorchTitan has both rope classes, so that claim can be tested entirely inside
TorchTitan -- where a forward costs 0.1s instead of HF's 213s. agpt_2b is
ComplexRoPE (needs_qk_permute=True); agpt_2b_real is CosSinRoPE, which
implements HF's rotate_half convention and takes HF-layout weights as-is.

If these two agree, the permutation and the rope-class pairing are correct and
any remaining HF divergence is an HF-side or harness issue. If they disagree,
the conversion itself is wrong.
"""
import sys, dataclasses
sys.path.insert(0, "/lus/flare/projects/datascience/seonghapark/xpu_launcher/xpu_torchtitan/torchtitan_repo")
sys.path.insert(0, "/lus/flare/projects/datascience/seonghapark/titanx/rope_complex_switch")
import torch

from dcp_model_only import load_model_only

BASE = "/lus/flare/projects/datascience/seonghapark"
HF = f"{BASE}/converted_models/agpt-2b-v2-step-46400_hf"
SEQ, BATCH, SEED = 64, 1, 1234
torch.manual_seed(SEED); torch.set_grad_enabled(False)

from torchtitan.config.manager import ConfigManager
from torchtitan.models.common.rope import ComplexRoPE, CosSinRoPE


def build(flavor, seq):
    cfg = ConfigManager().parse_args(["--module", "agpt", "--config", flavor,
                                      "--hf_assets_path", HF])
    mc = cfg.model_spec.model
    for layer in mc.layers:
        if getattr(layer, "attention", None) is not None:
            layer.attention.rope = dataclasses.replace(layer.attention.rope,
                                                       max_seq_len=seq)
    with torch.device("cpu"):
        m = mc.build()
    m.init_states(buffer_device=torch.device("cpu"))
    return mc, m.eval()


# ---------------------------------------------------------------- the adapter
mc_c, m_complex = build("agpt_2b", SEQ)
print(f"agpt_2b      rope={type(mc_c.layers[0].attention.rope).__qualname__}")
mc_r, m_cossin = build("agpt_2b_real", SEQ)
print(f"agpt_2b_real rope={type(mc_r.layers[0].attention.rope).__qualname__}")

from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter
import json, os
with open(os.path.join(HF, "config.json")) as f:
    hf_cfg = json.load(f)
ad_c = Llama3StateDictAdapter(mc_c, hf_cfg)
ad_r = Llama3StateDictAdapter(mc_r, hf_cfg)
print(f"needs_qk_permute: complex={ad_c._needs_qk_permute()} "
      f"cos_sin={ad_r._needs_qk_permute()}")

native = {k: v.to(torch.float32) for k, v in load_model_only().items()}

# DCP -> HF layout using the real converter path, then back into the cos_sin
# model, which consumes HF layout directly.
hf_layout = ad_c.to_hf(native)
back_for_cossin = ad_r.from_hf(hf_layout)

a = m_complex.load_state_dict(native, strict=False)
b = m_cossin.load_state_dict(back_for_cossin, strict=False)
print(f"complex load: {len([k for k in a.missing_keys if 'freqs' not in k])} missing, "
      f"{len(a.unexpected_keys)} unexpected")
print(f"cos_sin load: {len([k for k in b.missing_keys if 'freqs' not in k])} missing, "
      f"{len(b.unexpected_keys)} unexpected")

# Did the round trip actually move the Q/K rows, or is this a no-op test?
qk = "layers.0.attention.qkv_linear.wq.weight"
moved = not torch.equal(native[qk], back_for_cossin[qk])
print(f"round trip changed {qk}: {moved}")

g = torch.Generator().manual_seed(SEED)
tokens = torch.randint(0, mc_c.vocab_size, (BATCH, SEQ), generator=g)

lc = m_complex(tokens).float()
lr = m_cossin(tokens).float()

d = (lc - lr).abs().max().item()
s = lc.abs().max().item()
cos = torch.nn.functional.cosine_similarity(lc.reshape(-1), lr.reshape(-1), 0).item()
top1 = (lc.argmax(-1) == lr.argmax(-1)).float().mean().item()
print(f"\n=== ComplexRoPE(W) vs CosSinRoPE(permute(W)), inside TorchTitan ===")
print(f"  max abs diff : {d:.3e}   (logit scale {s:.3f}, rel {d/max(s,1e-12):.3e})")
print(f"  cosine       : {cos:.10f}")
print(f"  argmax agree : {top1*100:.4f}%")
print("VERDICT:", "CONVERSION SEMANTICS OK" if d / max(s, 1e-12) < 1e-4
      else "CONVERSION SEMANTICS BROKEN")
