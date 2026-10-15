"""A: Compare RoPE type / rope_factor / frequency settings / max context across
the three places they are defined: the HF config.json files and the TorchTitan
training flavors."""
import json, sys
sys.path.insert(0, "/lus/flare/projects/datascience/seonghapark/xpu_launcher/xpu_torchtitan/torchtitan_repo")
from torchtitan.models.agpt import config_registry as reg
from torchtitan.models.common.rope import ComplexRoPE, CosSinRoPE

BASE = "/lus/flare/projects/datascience/seonghapark"
hf_base = json.load(open(f"{BASE}/agpt-2b-v2-256n-step-92859-safetensors/config.json"))
hf_conv = json.load(open(f"{BASE}/converted_models/agpt-2b-v2-step-46400_hf/config.json"))

def titan(fn):
    cfg = fn()
    m = cfg.model_spec.model
    a = m.layers[0].attention
    r = a.rope
    return dict(
        flavor=cfg.model_spec.flavor,
        rope_cls=type(r).__qualname__,
        theta=r.theta, scaling=r.scaling,
        rope_factor=getattr(r, "rope_factor", None),
        original_seq_len=getattr(r, "original_seq_len", None),
        max_seq_len=r.max_seq_len,
        beta_fast=getattr(r, "beta_fast", None),
        beta_slow=getattr(r, "beta_slow", None),
        n_heads=a.n_heads, n_kv=a.n_kv_heads, dim=m.dim,
        seq_len_cfg=cfg.training.seq_len,
    )

t_pre = titan(reg.agpt_2b)        # flavor the checkpoint was PRETRAINED under
t_yarn = titan(reg.agpt_2b_yarn)  # flavor used for continued training

print("=== A1. HF config.json: base vs converted ===")
keys = sorted(set(hf_base) | set(hf_conv))
diffs = 0
for k in keys:
    b, c = hf_base.get(k, "<absent>"), hf_conv.get(k, "<absent>")
    if b != c:
        diffs += 1
        print(f"  DIFF {k}: base={b!r} converted={c!r}")
print(f"  -> {diffs} differing key(s)" if diffs else "  -> identical (converter copied assets verbatim)")

print("\n=== A2. RoPE scaling declared in HF config? ===")
for nm, c in (("base", hf_base), ("converted", hf_conv)):
    print(f"  {nm:10s} rope_scaling={c.get('rope_scaling', '<ABSENT>')} "
          f"max_position_embeddings={c.get('max_position_embeddings')} "
          f"rope_theta={c.get('rope_theta')}")

print("\n=== A3. TorchTitan flavors ===")
for nm, t in (("agpt_2b (pretrain)", t_pre), ("agpt_2b_yarn (training)", t_yarn)):
    print(f"  {nm}")
    for k, v in t.items():
        print(f"      {k:18s} {v}")

print("\n=== A4. Cross-checks ===")
def chk(label, ok, detail=""):
    print(f"  [{'OK ' if ok else 'MISMATCH'}] {label}" + (f"  -- {detail}" if detail else ""))
    return ok

allok = True
allok &= chk("theta: HF config vs pretrain flavor",
             float(hf_conv["rope_theta"]) == float(t_pre["theta"]),
             f"{hf_conv['rope_theta']} vs {t_pre['theta']}")
allok &= chk("theta: pretrain vs yarn flavor",
             t_pre["theta"] == t_yarn["theta"], f"{t_pre['theta']} vs {t_yarn['theta']}")
allok &= chk("n_heads / n_kv / dim: HF vs flavor",
             hf_conv["num_attention_heads"] == t_pre["n_heads"]
             and hf_conv["num_key_value_heads"] == t_pre["n_kv"]
             and hf_conv["hidden_size"] == t_pre["dim"])
allok &= chk("vocab_size: HF vs flavor-independent 256128",
             hf_conv["vocab_size"] == 256128)
allok &= chk("pretrain flavor has NO yarn scaling",
             t_pre["scaling"] != "yarn", f"scaling={t_pre['scaling']}")
allok &= chk("yarn flavor IS yarn + ComplexRoPE",
             t_yarn["scaling"] == "yarn" and t_yarn["rope_cls"] == "ComplexRoPE.Config",
             f"{t_yarn['scaling']}/{t_yarn['rope_cls']}")
allok &= chk("both flavors now use the SAME RoPE class (post-change)",
             t_pre["rope_cls"] == t_yarn["rope_cls"],
             f"{t_pre['rope_cls']} vs {t_yarn['rope_cls']}")

print("\n=== A5. Max context length ===")
print(f"  HF max_position_embeddings : {hf_conv['max_position_embeddings']}")
print(f"  pretrain flavor max_seq_len: {t_pre['max_seq_len']}")
print(f"  yarn flavor max_seq_len    : {t_yarn['max_seq_len']}  (architectural ceiling)")
print(f"  yarn flavor training seq_len: {t_yarn['seq_len_cfg']} -> rope_factor "
      f"{max(1.0, t_yarn['seq_len_cfg']/t_yarn['original_seq_len'])}")

print("\nRESULT:", "CONFIG CONSISTENT" if allok else "CONFIG MISMATCH FOUND")
