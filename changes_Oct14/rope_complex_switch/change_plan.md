# Change Plan: YaRN flavors switch from CosSinRoPE to ComplexRoPE

**Date**: 2026-10-12
**Request**: "I want RoPE flavors use ComplexRoPE, not CosSinRoPE. If current YaRN
flavors set to use CosSinRoPE, change them to use ComplexRoPE."
**Status**: implemented and statically verified; hardware smoke train queued
(PBS `8923308`, `xpu_launcher/job_smoke_rope_complex.sh`)

---

## Verdict table

| Question | Answer | Evidence |
|---|---|---|
| Do the YaRN flavors currently use CosSinRoPE? | **Yes** — both | `agpt/__init__.py:646` (`2B_yarn`), `:662` (`20B_yarn`), `rope_backend="cos_sin"` |
| Is the stated reason for cos_sin valid? | **No — the comment is false** | `agpt/__init__.py:629` claims cos_sin is needed "to get mscale attention-temperature correction". ComplexRoPE implements mscale identically. Measured below. |
| Are the two backends numerically equivalent? | **No, not under active YaRN** | Equivalent at `scaling=none` and `yarn, factor=1.0` (4.8e-07). **DIFFER** at factor 2/4/32 (up to 1.8e-02). Measured below. |
| Root cause of the difference | **Integer rounding of the correction range** | ComplexRoPE floors/ceils to ints; CosSinRoPE keeps floats. Same formula otherwise. |
| Does the switch change checkpoint handling? | **Yes** | `_needs_qk_permute()` flips `False -> True` for these flavors |
| Should the `*_real` flavors change too? | **No — leave them** | They exist *specifically* to be the cos_sin A/B arm |

---

## 1. What changes

Two flavors in `torchtitan/models/agpt/__init__.py`:

```python
"2B_yarn":  _build_agpt_config(..., rope_backend="cos_sin", scaling="yarn", ...)   # -> "complex"
"20B_yarn": _build_agpt_config(..., rope_backend="cos_sin", scaling="yarn", ...)   # -> "complex"
```

Aliases `agpt_configs["2b_yarn"]` / `["20b_yarn"]` point at the same objects, so the
registry entry points `agpt_2b_yarn()` (`config_registry.py:342`) and
`agpt_20b_yarn()` (`:362`) inherit the change with no edit of their own.

Plus the stale comment at `__init__.py:629`.

### What does NOT change

`agpt_2b_real` / `agpt_20b_real` / `agpt_80b_real` keep `_set_rope_backend(..., "cos_sin")`.
Their docstring states their purpose: inductor cannot codegen complex ops, so these
flavors exist to A/B test whether a real-valued rotation compiles better on XPU.
Converting them to complex would make them duplicates of their base flavors and
delete the comparison. They are not YaRN flavors and the request named YaRN.
**If the intent was a repo-wide ban on CosSinRoPE, these three should be deleted
rather than converted — flagged for the user, not done unilaterally.**

---

## 2. Measurements behind the verdict

Scripts: `verify_rope.py`, `verify_yarn_scaled.py`, `isolate_diff.py` (this directory).
Run under `module load frameworks/2026.1.0` + venv, from the torchtitan repo root.

### 2a. mscale parity — disproves the in-code rationale

```
factor=   1.0  ComplexRoPE mscale=1.000000  expected=1.000000 OK
factor=   2.0  ComplexRoPE mscale=1.069315  expected=1.069315 OK
factor=   4.0  ComplexRoPE mscale=1.138629  expected=1.138629 OK
factor=  32.0  ComplexRoPE mscale=1.346574  expected=1.346574 OK
```

Both classes compute `0.1*ln(factor) + 1.0`. ComplexRoPE applies it inside
`apply_rotary_emb(..., mscale=mscale)`; CosSinRoPE folds it into the cos/sin cache.
Same value, same effect. **The comment at line 629 is wrong and must be corrected
regardless of whether the flavor switch is kept.**

### 2b. Output equivalence

Comparison applies a feature-axis permutation to emulate the adapter's `_permute`,
so the two layouts (interleaved vs rotate_half) are compared like-for-like.

| config | max abs diff | verdict |
|---|---|---|
| `scaling="none"` | 4.768e-07 | MATCH |
| `yarn, factor=1.0` | 4.768e-07 | MATCH |
| `yarn, factor=2.0, orig=8192` | 8.021e-03 | **DIFFER** |
| `yarn, factor=4.0, orig=8192` | 1.559e-02 | **DIFFER** |
| `yarn, factor=32.0, orig=8192` | 1.786e-02 | **DIFFER** |

### 2c. Root cause — integer vs float correction range

The two correction-range formulas are *algebraically identical*
(`dim / (2·log b)` == `d_half / log b`). They differ in one respect only:
ComplexRoPE wraps each endpoint in `math.floor` / `math.ceil`; CosSinRoPE keeps floats.

```
ComplexRoPE correction range: low=22       high=42        (integers)
CosSinRoPE  correction range: low=21.9290  high=42.4292   (floats)
```

That shifts the NTK-by-parts ramp by a fraction of a dimension, which changes
`inv_freq` only inside the transition band:

```
factor=2.0   inv_freq max rel-diff=2.051e-02   differing dims=21/64
factor=4.0   inv_freq max rel-diff=5.909e-02   differing dims=21/64
factor=32.0  inv_freq max rel-diff=3.936e-01   differing dims=21/64
```

43 of 64 frequency dimensions are bit-identical; the 21 inside the ramp move.

**Interpretation.** This is a rounding discrepancy, not two different algorithms.
Neither is obviously "correct": the integer version matches the reference YaRN /
DeepSeek-V2 implementation (which also floors/ceils), the float version is the
smoother interpolation. So the switch is defensible — but it is **not** a no-op, and
it is **not** safe to describe as numerically neutral. Any YaRN checkpoint trained
under cos_sin has learned against the float ramp.

---

## 3. The real risk: checkpoint weight layout

`torchtitan/models/llama3/state_dict_adapter.py`:

```python
def _rope_cls(self) -> type:
    return type(self.model_config.layers[0].attention.rope)

def _needs_qk_permute(self) -> bool:
    return issubclass(self._rope_cls(), ComplexRoPE.Config)
```

ComplexRoPE uses Meta's **interleaved** layout (rows `2i`, `2i+1` pair); CosSinRoPE
uses HF's **rotate_half** layout (rows `i`, `i + head_dim/2` pair). The adapter keys
the Q/K permutation off the RoPE class, so after this change
`_needs_qk_permute()` returns **True** for `2b_yarn` / `20b_yarn` in both `to_hf`
and `from_hf`.

Consequences:

1. **Loading HF base weights into a YaRN flavor** now permutes Q/K. This is the
   *correct* behaviour for a complex-RoPE model, so training from a stock HF
   checkpoint stays right.
2. **`dcp_to_hf.py` output changes** for these flavors. The converter derives the
   adapter from `--module/--config`, so it is correct by construction *provided the
   flavor passed to it matches the one that produced the checkpoint*.
3. **Already-converted artifacts do not retroactively change**, but a checkpoint
   trained under cos_sin and converted under complex (or vice versa) will load
   cleanly and silently corrupt attention. This is exactly the hazard recorded in
   the existing `dcp-conversion-rope-flavor-trap` note.

**Rule after this change**: a DCP checkpoint must be converted under the same RoPE
class it was trained under. Checkpoints produced before this change were trained
with cos_sin; converting them with the post-change `agpt_2b_yarn` is wrong.

---

## 4. Implementation — DONE

| File | Change |
|---|---|
| `torchtitan/models/agpt/__init__.py` | `2B_yarn`: dropped `rope_backend="cos_sin"` |
| `torchtitan/models/agpt/__init__.py` | `20B_yarn`: dropped `rope_backend="cos_sin"` |
| `torchtitan/models/agpt/__init__.py` | replaced the false mscale comment with the real rationale (rounding difference + permute flip) |
| `xpu_launcher/job_convert_and_smoke.sh` | conversion comment restated generally; notes that pre-2026-10-12 YaRN checkpoints are the remaining hazard |
| `titanx/dcp_to_hf_qsub_results/QSUB_RUN_RESULTS.md` | §9 amendment superseding §3.1/§7 |
| `xpu_launcher/job_smoke_rope_complex.sh` | **new** — smoke job with an up-front RoPE-class assertion |

`"complex"` is already the `_build_agpt_config` default, so the flavor edits drop
the argument rather than restate it. `*_real` flavors untouched (section 1).
No other file in the repo or the launcher references these flavor names in a way
that needed updating (checked by grep).

---

## 5. Verification plan

- [x] mscale parity across factors (2a)
- [x] output equivalence at `none` / `factor=1.0` (2b)
- [x] divergence quantified at factor 2/4/32 and root-caused (2b, 2c)
- [x] confirm both YaRN flavors actually construct a `ComplexRoPE` after the edit
      (`verify_switch.py`: raw specs, both aliases, and both registry entry points)
- [x] confirm `*_real` flavors still build `CosSinRoPE`
- [x] confirm `_needs_qk_permute()` is `True` post-change (and still `False` for `_real`)
- [x] full training-config path: `update_from_config` keeps the class, YaRN sync
      gives `rope_factor=4.0` at `seq_len=32768`, complex cache builds
      (`(32768, 64) complex64`)
- [x] launcher test suite: 40/40 pass
- [x] Q/K permute round-trips exactly under the new flavor
      (`verify_permute_roundtrip.py`): `_reverse_permute(_permute(w)) == w`
      bit-exact for `q_proj` and `k_proj`, and the permute genuinely reorders
      rows rather than being a silent no-op. This matters because the permute is
      newly *enabled* for these flavors — had it not been an exact inverse pair,
      every HF<->DCP round trip would degrade weights.
- [x] 5-step smoke train — **PBS `8923308`, rc 0** (`job_smoke_rope_complex.sh`,
      4 nodes, debug-scaling, `seq_len=16384` → `rope_factor=2.0`). See §6.

### Note on `rope_factor` in practice

`agpt_2b_yarn` trains at `seq_len=32768` → `rope_factor=4.0`, which is inside the
DIFFER regime (section 2b), not at the benign `factor=1.0` point. The change is
therefore live for real runs, not latent.

---

## 6. Hardware smoke train result (PBS 8923308, rc 0)

Precheck on-node: `rope class = ComplexRoPE.Config`, `needs_qk_permute = True`.
Trainer banner: `Model Flavor: 2b_yarn`, `RoPE Backend: ComplexRoPE.Config`,
`rope_factor: 2.0` (from `seq_len=16384 / original 8192`) — i.e. the run was in
the DIFFER regime, not at the benign factor=1.0 point.

Same job, same checkpoint, same data, before vs after the switch:

| step | cos_sin (10/09) | complex (10/12) | delta |
|---|---|---|---|
| 1 | 12.93765 | 12.92994 | -0.00771 |
| 2 | 12.84021 | 12.82589 | -0.01432 |
| 3 | 12.64411 | 12.62204 | -0.02207 |
| 4 | 12.36340 | 12.34369 | -0.01971 |
| 5 | 11.97022 | 11.95746 | -0.01276 |

**Reading.** The newly-enabled Q/K reverse permute did not corrupt the weights.
A wrong permutation does not crash — it degrades silently — so a matching loss
curve is the evidence that matters, and the curves track to within 0.02 nats with
the same monotone descent and `grad_norm ~2.0`. The small consistent improvement
is the expected size for the correction-range rounding difference at factor 2.0.
Throughput unchanged (~83 TFLOPs, ~28% MFU).

**Caveat, pre-existing and not caused by this change.** Both runs start near
12.93, above `ln(256128) = 12.45`. That is already recorded as open question #2
in `../dcp_to_hf_qsub_results/QSUB_RUN_RESULTS.md` (attributed there to fresh
PG19 data at a new seq_len) and is unaffected by the RoPE switch — the 10/09
baseline shows the same value. A perplexity comparison against the base
safetensors is still the test that would settle it. **Do not read this smoke
result as proof the converted checkpoint is numerically faithful; it only shows
the switch changed nothing for the worse.**

### Side finding: the banner's mscale line was lying

`RoPE.Config.mscale` (`common/rope.py:66`, default `0.0`) is **dead** — nothing
in the repo reads it. The trainer banner printed it verbatim, so every run logged
`RoPE mscale: 0.0` while the value actually in force was `1 + 0.1*ln(factor)`
(1.069 for this run), computed independently inside each backend. DeepSeek-V3 has
its own unrelated `mscale` on its *attention* config, which is live; this one is a
vestige.

Fixed in `torchtitan/trainer.py`: the banner now derives and labels the effective
value (`RoPE mscale (effective): ...`) instead of echoing the dead field. The
field itself is left in place — removing it is a separate cleanup, and anything
constructing a RoPE config by copying all dataclass fields (e.g.
`config_registry._set_rope_backend`) would need checking first.

---

## 7. Open question for the user

Checkpoints already trained under `agpt_2b_yarn` / `agpt_20b_yarn` used CosSinRoPE.
After this change they cannot be resumed or converted under the same flavor name
without a layout mismatch. Options: accept (retrain from base), or add a pinned
`*_yarn_cossin` flavor to keep the old path reachable for conversion only.
No action taken on this.
