"""Post-change verification: do the YaRN flavors now build ComplexRoPE, and
does the state-dict adapter's permute decision flip as predicted?"""
from torchtitan.models.common.rope import ComplexRoPE, CosSinRoPE
from torchtitan.models.agpt import agpt_configs
from torchtitan.models.agpt import config_registry as reg

print("=== raw flavor specs (agpt_configs) ===")
ok = True
for name in ("2B_yarn", "20B_yarn", "2b_yarn", "20b_yarn"):
    spec = agpt_configs[name]
    cls = type(spec.layers[0].attention.rope)
    good = issubclass(cls, ComplexRoPE.Config)
    ok &= good
    print(f"  {name:10s} -> {cls.__qualname__:26s} {'OK' if good else 'FAIL'}")

print("\n=== registry entry points ===")
for fn in (reg.agpt_2b_yarn, reg.agpt_20b_yarn):
    cfg = fn()
    rope = cfg.model_spec.model.layers[0].attention.rope
    cls = type(rope)
    good = issubclass(cls, ComplexRoPE.Config)
    ok &= good
    print(f"  {fn.__name__:16s} -> {cls.__qualname__:26s} "
          f"scaling={rope.scaling} {'OK' if good else 'FAIL'}")

print("\n=== _real flavors must STAY cos_sin (they are the A/B arm) ===")
for fn in (reg.agpt_2b_real, reg.agpt_20b_real, reg.agpt_80b_real):
    cls = type(fn().model_spec.model.layers[0].attention.rope)
    good = issubclass(cls, CosSinRoPE.Config)
    ok &= good
    print(f"  {fn.__name__:16s} -> {cls.__qualname__:26s} {'OK' if good else 'FAIL'}")

print("\n=== adapter _needs_qk_permute (expected: now True for yarn) ===")
from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter
for fn in (reg.agpt_2b_yarn, reg.agpt_20b_yarn, reg.agpt_2b_real):
    cfg = fn()
    spec = cfg.model_spec
    ad = spec.state_dict_adapter(spec.model, None)
    print(f"  {fn.__name__:16s} needs_qk_permute={ad._needs_qk_permute()}")

print("\nRESULT:", "ALL CHECKS PASS" if ok else "FAILURES PRESENT")
