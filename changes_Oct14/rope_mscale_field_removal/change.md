# Removing the dead `RoPE.Config.mscale` field

**Date**: 2026-10-12
**Request**: "Please investigate whether `RoPE.Config.mscale` is truly unused and can
be safely removed. Check all references, including `_set_rope_backend` and any
serialization or backend compatibility implications. If it is safe to remove, make
that a separate change; otherwise, document why the field must remain. Keep the
current effective-value banner fix unchanged."

**Verdict: the field is genuinely dead. Removed.**

| Question | Answer |
|---|---|
| Is it read anywhere for its value? | No — zero semantic reads |
| Does `_set_rope_backend` depend on it? | No — generic `dataclasses.fields()` copy |
| Does it reach any persisted artifact? | No — absent from the saved config JSON |
| Does upstream have it? | **No** — torchtitan 0.3.0 has no such field |
| Does removing it change numerics? | No — bit-identical to 12 decimals |

This is a **separate change** from the trainer banner fix. The banner fix
(`trainer.py:819-836`, derive and log the effective mscale) is left exactly as it
was; only its comment was reworded, because it referred to the field in the present
tense.

---

## 1. What changed

One field deleted from `torchtitan/models/common/rope.py`, `RoPE.Config`:

```python
-        mscale: float = 0.0
```

replaced by a NOTE comment in its place recording why there is deliberately no such
field, so the next reader does not "restore" it. One comment in `trainer.py` retensed
from "that field ... is vestigial" to "that field existed until 2026-10-12".

Nothing else in the repository was touched.

---

## 2. Evidence that it is dead

### 2a. Static: every `.mscale` in the tree

Four categories, no fifth:

| Site | What it is |
|---|---|
| `common/rope.py` `_get_mscale`, `apply_rotary_emb`, `_precompute_cache` | **derived locals** — both backends compute `1 + 0.1*ln(rope_factor)` themselves |
| `deepseek_v3/model.py:53,86` | a **different, live field**: `Attention.Config.mscale`, not `RoPE.Config.mscale` |
| `deepseek_v3/__init__.py` (`mscale=0.70/1.0`) | kwargs targeting that **Attention** config |
| `trainer.py`, `agpt/*` comments | prose only |

The DeepSeek one is worth being explicit about, because the name collision is the
whole reason this field survived so long. It lives on the attention config and is
read at `deepseek_v3/model.py:86`:

```python
if config.rope.max_seq_len > config.rope.original_seq_len:
    mscale = 0.1 * config.mscale * math.log(config.rope.rope_factor) + 1.0
    self.softmax_scale = self.softmax_scale * mscale * mscale
```

Note `config.mscale`, not `config.rope.mscale`. DeepSeek folds mscale into the
*softmax scale*; the RoPE object never sees it.

### 2b. Runtime: a trap on the attribute

Replacing the field with a property that logs every access, then building all 26
agpt flavors and all 4 DeepSeek flavors, serializing them, switching backends and
running forwards:

- **2429 accesses**, from exactly **4 call sites**
- all 4 are generic field machinery: `dataclasses.fields()` copying in
  `_set_rope_backend`, `Configurable.Config.to_dict()`, `override.derive()`, and
  `copy.deepcopy`
- **0 reads of the value for its meaning**

`_set_rope_backend` (`agpt/config_registry.py:38`) is the specific one the request
asked about. It never names the field:

```python
target_cls = ComplexRoPE.Config if backend == "complex" else CosSinRoPE.Config
for layer in model.layers:
    old_rope = layer.attention.rope
    if old_rope is None:
        continue
    kwargs = {f.name: getattr(old_rope, f.name) for f in fields(old_rope)}
    layer.attention.rope = target_cls(**kwargs)
```

Both source and target are `RoPE.Config` subclasses, so the field set is the same on
both sides whether or not `mscale` is among them. Removing it from both removes it
from the copy. No asymmetry is possible.

### 2c. Serialization

- **Saved config JSON: not present.** `trainer.py`'s `to_dict` reduces `model_spec`
  to `{"name", "flavor"}`, so no RoPE field — mscale or otherwise — reaches a
  persisted artifact. There is no checkpoint, resume, or config-file compatibility
  surface to break.
- **Log-only dicts: present 12×** in `model_config.to_dict()` (the torchft trainer
  log line and the metrics `config_dict=`). Those are emitted, never read back.
  Their mscale entries were noise.

### 2d. The one behavioral coupling, and why it is benign

`_apply_config_overrides` (`agpt/config_registry.py:~230`) raises
`KeyError(f"Unknown config field {key!r} ...")` for fields it cannot find. So a
`TT_CONFIG_JSON` that sets `mscale` on a RoPE config would now fail loudly instead
of being silently absorbed.

That path is reachable only through `TT_CONFIG_JSON` plus a `*_from_json` flavor.
No such JSON files exist in the launcher. And failing loudly is the better outcome:
previously, setting `mscale` in such a file would have been accepted and then
ignored, which is exactly the class of silent-no-op this change exists to remove.

### 2e. Upstream

The installed wheel (torchtitan **0.3.0**,
`venv/lib/python3.12/site-packages/torchtitan/models/common/rope.py`) declares:

```
dim, max_seq_len, theta, scaling, scaling_factor, low_freq_factor,
high_freq_factor, original_max_position_embeddings, rope_factor,
beta_fast, beta_slow, original_seq_len, truncate
```

**No `mscale`.** Removal converges on upstream rather than diverging from it.
Upstream is also explicit about the design intent:

> The YaRN attention "mscale" is intentionally NOT applied here -- the rope stays a
> pure rotation and the model folds mscale into its softmax scale.

(Our fork does apply it inside the rope, which is a separate pre-existing divergence
— see `rope_complex_switch/change_plan.md`. It is still *derived*, not configured.)

---

## 3. Verification

| Check | Result |
|---|---|
| 26 agpt flavors build + serialize | 26/26, 0 failures |
| 4 DeepSeek flavors build + serialize, own mscale intact | 4/4 (0.7 / 1.0 / 1.0 / 0.7) |
| `_set_rope_backend` both directions | OK (`agpt_2b_real`→CosSin, `agpt_2b_yarn`→Complex) |
| `override.derive()` | OK |
| Forward passes | OK |
| **Numerics**: q/k/cache checksums, `agpt_2b` / `agpt_2b_yarn` / `agpt_2b_real`, seq 64 and 256 | **identical to 12 decimals** with and without the field |
| Launcher test suite (final file) | **40/40 passed** |

The numeric check is the decisive one: if any code path had been reading the field,
removing a `0.0` default would have perturbed something. Nothing moved.

---

## 4. Why it was there

Best reconstruction: the field was added alongside the other YaRN knobs
(`rope_factor`, `beta_fast`, `beta_slow`, `original_seq_len`) by analogy with
DeepSeek's config, where an `mscale` knob is real. But in the RoPE backends the
value is a pure function of `rope_factor`, so there was nothing for a knob to do,
and no code was ever written to read it.

Its only observable effect was harmful: the trainer banner printed
`RoPE mscale: 0.0` — a value that was never in force and is not even a legal mscale
(the true value is ≥ 1.0). That is what the banner fix addressed, and this change
removes the thing that made the wrong value available to print.

---

## 5. Files

| File | Change |
|---|---|
| `torchtitan/models/common/rope.py` | field deleted, NOTE comment in its place |
| `torchtitan/trainer.py` | comment retensed; **banner logic unchanged** |

Backups: `/tmp/rope_with_mscale.py` (before), `/tmp/rope_no_mscale.py` (after).

Related: `titanx/rope_complex_switch/change_plan.md` (the ComplexRoPE switch and the
banner fix), memory `rope-config-mscale-field-is-dead`.
