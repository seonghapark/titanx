"""Verify the trainer banner's effective-mscale now matches what the backend
actually applies. Build each config at its own seq_len (do NOT mutate after
build -- Decoder.update_from_config validates seq_len <= rope.max_seq_len)."""
import copy, math, sys
sys.path.insert(0, "/lus/flare/projects/datascience/seonghapark/xpu_launcher/xpu_torchtitan/torchtitan_repo")
from torchtitan.models.agpt import config_registry as reg
from torchtitan.models.common.rope import ComplexRoPE

def banner_mscale(rope):
    """Mirrors the new trainer.py banner logic exactly."""
    scaling = getattr(rope, "scaling", None)
    factor = getattr(rope, "rope_factor", 1.0)
    extends = getattr(rope, "max_seq_len", 0) > getattr(rope, "original_seq_len", 0)
    if scaling == "yarn" and factor > 1.0 and extends:
        return 0.1 * math.log(factor) + 1.0
    return 1.0

ok = True
for seq in (8192, 16384, 32768, 65536):
    # deepcopy: reg.agpt_2b_yarn() hands back specs that alias the module-level
    # agpt_configs dict, so update_from_config in one iteration would lower
    # max_seq_len for every later one.
    cfg = copy.deepcopy(reg.agpt_2b_yarn())
    cfg.training.seq_len = seq          # before update_from_config, so it is honored
    m = cfg.model_spec.model
    m.update_from_config(config=cfg)
    r = m.layers[0].attention.rope
    runtime = ComplexRoPE(r)._get_mscale()   # what the model really applies
    shown = banner_mscale(r)                 # what the banner now prints
    good = abs(shown - runtime) < 1e-12
    ok &= good
    print(f"seq_len={seq:6d} factor={r.rope_factor:5.2f}  "
          f"banner={shown:.6f}  runtime={runtime:.6f}  "
          f"dead_field={getattr(r,'mscale',None)}  {'OK' if good else 'MISMATCH'}")

print("\nRESULT:", "BANNER MATCHES RUNTIME" if ok else "BANNER STILL WRONG")
