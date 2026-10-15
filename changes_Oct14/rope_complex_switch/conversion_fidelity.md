# Conversion fidelity: step-46400 DCP → HF safetensors

**Date**: 2026-10-12
**Request**: "Compare the checkpoint's RoPE type, `rope_factor`, frequency settings,
and maximum context length against the training configuration. After loading the HF
checkpoint, verify that the permutation and inverse permutation of the Q/K weights
are correct. Run the model on identical inputs before and after conversion and compare
the output logits. Whenever possible, use the same precision and execution conditions,
and verify that the differences fall within an acceptable tolerance."

**Artifacts under test**

| Role | Path |
|---|---|
| Source DCP | `AuroraGPT/foremans/runs/agpt-2b-v2/.../agpt-2b-sophiag-olmo-mix-1124-n512-gbs12288/step-46400` |
| Converted HF | `converted_models/agpt-2b-v2-step-46400_hf` (fp32, 2 shards, 111 tensors) |
| Base/assets HF | `agpt-2b-v2-256n-step-92859-safetensors` (bf16, single file) |

Scripts: `verify_a_config.py`, `verify_b_permute.py`, `verify_c_logits.py`,
`dcp_model_only.py` (shared loader), all in this directory.

---

## Summary

| Part | Question | Status |
|---|---|---|
| A | RoPE type / `rope_factor` / frequencies / max context vs training config | **CONFIG CONSISTENT** |
| B | Q/K permutation and inverse correct on the real stored weights | **VERIFIED** (24/24 Q/K, 87/87 others bitwise) |
| C | Logits identical before vs after conversion | pending |
| D | Matched precision and execution conditions, explicit tolerance | pending |

---

## A. Config and RoPE comparison — CONFIG CONSISTENT

`verify_a_config.py`.

### A1-A2. What the HF checkpoint declares

The converter copies the asset files verbatim, so the converted `config.json` is
**byte-identical** to the base model's. On disk, both files declare:

```json
"rope_theta": 50000.0,
"max_position_embeddings": 131072
```

and **`rope_scaling` is absent** — no scaling of any kind is declared.

Loading it confirms that is how transformers reads it. transformers 5.14.1
normalizes RoPE settings into a nested `rope_parameters` dict (there is no
top-level `config.rope_theta` attribute on the loaded object any more), giving:

```
rope_parameters = {"rope_theta": 50000.0, "rope_type": "default"}
LlamaRotaryEmbedding.attention_scaling = 1.0
```

`rope_type: "default"` and `attention_scaling = 1.0` are the load-time
confirmation that the absent `rope_scaling` really does mean plain, unscaled RoPE.

### A3. What the TorchTitan flavors declare

| | `agpt_2b` (pretrain) | `agpt_2b_yarn` (extension) |
|---|---|---|
| RoPE class | `ComplexRoPE.Config` | `ComplexRoPE.Config` |
| scaling | `none` | `yarn` |
| `rope_factor` | 1.0 | 32.0 (ceiling) → **4.0** at `seq_len=32768` |
| `original_seq_len` | 4096 | 8192 |
| `max_seq_len` | 131072 | 262144 |
| `beta_fast` / `beta_slow` | 32.0 / 1.0 | 1.0 / 32.0 |
| configured `seq_len` | 8192 | 32768 |

Both now build `ComplexRoPE` (the change recorded in `change_plan.md`), so
`_needs_qk_permute()` is `True` for both.

### A4-A5. Cross-checks

- HF `max_position_embeddings` (131072) **equals** the pretrain flavor's
  `max_seq_len` (131072). Consistent.
- HF `rope_theta` (50000.0) **equals** both flavors' `theta` (50000). Consistent.
- The yarn flavor's 262144 is an architectural ceiling, not a claim about this
  checkpoint; `rope_factor` is resynced from `training.seq_len` at build time by
  `AgptModel.Config._sync_yarn_rope_factor`.

### Which flavor this checkpoint belongs to

**`agpt_2b`, scaling `none`.** This matters for everything downstream:

`config.json` declares no scaling, so a stock HF load applies plain RoPE. Running
a TorchTitan model configured with `agpt_2b_yarn` against that HF model would
compare *different positional math* — it would measure the YaRN ramp, not the
conversion. Parts B, C and D are therefore all run under `agpt_2b`, the flavor
the checkpoint was produced under and the one the conversion itself used.

**Conclusion for part A: CONFIG CONSISTENT.** The checkpoint's declared RoPE type,
theta, and maximum context length agree with the pretraining flavor, and no
`rope_factor` or frequency scaling is in force on either side.

---

## B. Q/K permutation on the real stored weights — VERIFIED

`verify_b_permute.py`, run against the real step-46400 DCP tensors and the real
converted safetensors (not synthetic data). Flavor `agpt_2b`, so
`needs_qk_permute=True`; `n_heads=16`, `n_kv=4`, `dim=2048`, `head_dim=128`,
12 layers.

### B1. Forward and inverse, per tensor — 24/24 correct

For every `q_proj` and `k_proj` in all 12 layers, three independent claims hold:

| Claim | Meaning | Result |
|---|---|---|
| `to_hf == disk` | the adapter's permutation of the DCP tensor reproduces **what is actually stored** in the converted safetensors, bit for bit | 24/24 |
| `from_hf == dcp` | `_reverse_permute` applied to the stored HF tensor recovers the **original DCP tensor**, bit for bit | 24/24 |
| `reordered` | the permutation is not a no-op — the result genuinely differs from the input | 24/24 |

The third column matters: without it, a permutation that silently did nothing
would pass the first two. It does not — every Q/K tensor is really reordered.

The first two together are the round-trip: `_reverse_permute(_permute(x)) == x` is
only half the question, since both could be wrong in a compensating way. Checking
each leg against the **bytes on disk** rules that out — the forward leg is pinned
to what the converter wrote, the inverse leg to what the trainer produced.

### B2. Everything else — 87/87 bitwise identical

The 87 non-QK tensors (`v_proj`, `o_proj`, MLP, norms, embeddings, lm_head) are
**bitwise equal** between DCP and HF, 0 differing. So the conversion's only
transformation is the Q/K permutation; nothing else is touched, rescaled, or
re-cast. 24 + 87 = 111, the full tensor set.

### B3. What the permutation is

```
head_dim=128; dst row <- src row: 0<-0, 1<-2, 2<-4, 3<-6, 4<-8, 5<-10, ...
(interleaved -> rotate_half: dst i <- src 2i for i < d/2, else 2(i - d/2) + 1)
```

This is the expected Meta-interleaved → HF-rotate_half mapping, confirming that
`_needs_qk_permute()` returning `True` for `ComplexRoPE` selects the right
transformation for the layout actually stored in the checkpoint.

**Conclusion for part B: PERMUTATION VERIFIED ON REAL WEIGHTS.**

---

## C. Logits before vs after conversion

*(pending — `verify_c_logits.py`)*

---

## D. Precision, execution conditions, and tolerance

*(pending)*

---

## Method note: reading the DCP

The converter's `_load_dcp` uses `_EmptyStateDictLoadPlanner()` with no key filter,
which materializes optimizer state as well. For this checkpoint that is **1110 of
1230 metadata entries** — the model is only 111. With 6144 shard files on Lustre
the unfiltered read did not complete in 50 minutes.

`_EmptyStateDictLoadPlanner(keys=...)` filters at plan time. `dcp_model_only.py`
passes the 111 model keys and caches the result to
`titanx/_scratch_rope/step46400_model_only.pt`, so parts B and C share one read.

This is a property of the verification harness, not a defect in `dcp_to_hf.py`:
the converter runs once as a batch job and the extra read is tolerable there.
It is worth knowing if anyone needs an interactive DCP read of a full
(resumable) checkpoint.
