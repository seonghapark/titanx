"""Why does TT-vs-HF agreement depend on sequence length?

At SEQ=16 the full model agreed to 1.1e-06 relative; at SEQ=32 it diverged and
at SEQ=128 it was catastrophic. A correct conversion cannot be length
dependent, so something in the harness or the attention path is.

Two controls, in order of suspicion:

 1. TT vs TT: the same weights, one model whose RoPE cache was shrunk to
    max_seq_len=SEQ (what every verify/diag script did) and one left large.
    If these differ, the shrinking hack is the bug and every earlier number is
    suspect -- including verify_c's.

 2. TT vs HF across SEQ, with the cache NOT shrunk. Locates the real boundary.

Loads each model once and reuses it, since an HF fp32 CPU forward costs ~200s.
"""
import sys, time, dataclasses
sys.path.insert(0, "/lus/flare/projects/datascience/seonghapark/xpu_launcher/xpu_torchtitan/torchtitan_repo")
sys.path.insert(0, "/lus/flare/projects/datascience/seonghapark/titanx/rope_complex_switch")
import torch

from dcp_model_only import load_model_only

BASE = "/lus/flare/projects/datascience/seonghapark"
HF = f"{BASE}/converted_models/agpt-2b-v2-step-46400_hf"
BIG = 512
SEQS = [16, 24, 32, 64, 128]
SEED = 1234
torch.manual_seed(SEED); torch.set_grad_enabled(False)

from torchtitan.config.manager import ConfigManager


def build(max_seq_len):
    cfg = ConfigManager().parse_args(["--module", "agpt", "--config", "agpt_2b",
                                      "--hf_assets_path", HF])
    mc = cfg.model_spec.model
    for layer in mc.layers:
        if getattr(layer, "attention", None) is not None:
            layer.attention.rope = dataclasses.replace(
                layer.attention.rope, max_seq_len=max_seq_len)
    with torch.device("cpu"):
        m = mc.build()
    m.init_states(buffer_device=torch.device("cpu"))
    return mc, m.eval()


def relmax(a, b):
    a, b = a.float(), b.float()
    s = max(a.abs().max().item(), 1e-12)
    return (a - b).abs().max().item() / s


native = {k: v.to(torch.float32) for k, v in load_model_only().items()}

mc_big, tt_big = build(BIG)
tt_big.load_state_dict(native, strict=False)
vocab = mc_big.vocab_size

print("=== control 1: does shrinking the RoPE cache change TT's own output? ===")
print(f"  {'SEQ':>5} | {'rel diff (big cache vs cache=SEQ)':>34}")
print("  " + "-" * 44)
shrink_bad = False
for S in SEQS:
    _, tt_small = build(S)
    tt_small.load_state_dict(native, strict=False)
    g = torch.Generator().manual_seed(SEED)
    tok = torch.randint(0, vocab, (1, S), generator=g)
    r = relmax(tt_big(tok), tt_small(tok))
    print(f"  {S:>5} | {r:>34.3e}")
    if r > 1e-5:
        shrink_bad = True
    del tt_small
print("  -> shrinking is HARMLESS" if not shrink_bad
      else "  -> SHRINKING THE CACHE CHANGES RESULTS: harness bug, earlier numbers invalid")

print("\n=== control 2: TT (full cache) vs HF, across sequence length ===")
from transformers import LlamaForCausalLM
t0 = time.time()
hf = LlamaForCausalLM.from_pretrained(HF, dtype=torch.float32,
                                      attn_implementation="eager").eval()
print(f"  HF loaded in {time.time()-t0:.0f}s", flush=True)
print(f"  {'SEQ':>5} | {'rel diff':>11} | {'cosine':>12} | {'argmax':>8}")
print("  " + "-" * 46)
for S in SEQS:
    g = torch.Generator().manual_seed(SEED)
    tok = torch.randint(0, vocab, (1, S), generator=g)
    lt = tt_big(tok).float()
    lh = hf(tok).logits.float()
    r = relmax(lt, lh)
    c = torch.nn.functional.cosine_similarity(lt.reshape(-1), lh.reshape(-1), 0).item()
    a = (lt.argmax(-1) == lh.argmax(-1)).float().mean().item() * 100
    print(f"  {S:>5} | {r:>11.3e} | {c:>12.9f} | {a:>7.2f}%", flush=True)
