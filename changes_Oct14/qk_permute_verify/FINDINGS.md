# Q/K permutation verification — AGPT-2B / Llama3StateDictAdapter

Date: 2026-10-13
Checkpoint: `converted_models/agpt-2b-v2-step-46400_hf`
Geometry: n_heads=16, n_kv=4 (GQA), head_dim=128, dim=2048, theta=50000.0, 12 layers
Conditions: CPU, float32, `torch.use_deterministic_algorithms(True)`, `torch.manual_seed(0)`

## Verdict on the question asked

The permutation and inverse permutation are **correct**, and the model produces
**equivalent outputs on identical inputs** before and after conversion.

| check | result |
|---|---|
| `_reverse_permute(_permute(w)) == w` | exact, `max\|d\| = 0` |
| round-trip on all 24 real q/k tensors | **bitwise** identical |
| permutation is non-trivial (not a no-op) | confirmed |
| permutation is a bijection over rows | confirmed |
| `_permute` direction | interleaved (Meta) → half-split (HF), correct for `to_hf()` |
| **attention scores, HF vs converted** | **max\|A−B\| = 1.431e-06, relative = 2.765e-07** |
| sensitivity control (permutation skipped) | `max\|A−C\| = 2.757e-01` |

Tolerance 1e-4 relative. The observed 2.765e-07 is fp32 accumulation noise over
a 2048-wide dot product — about one ulp of the score magnitude. The control
diverges by **six orders of magnitude** more, so the comparison is genuinely
sensitive to the permutation rather than passing vacuously.

The round-trip alone would not have established this: a wrong-but-invertible
permutation round-trips perfectly and still corrupts attention. The equivalence
was therefore tested by running both conventions end to end —
HF weights + `rotate_half` RoPE versus permuted weights + interleaved complex
RoPE — which must agree only if the permutation is the right one.

## Separate defect found: `to_local()` under row sharding

Not part of the question, found while checking execution conditions. Reporting
it rather than acting on it, per the standing instruction to leave
`state_dict_adapter.py` untouched unless a change is demonstrably necessary.

`_permute` / `_reverse_permute` call `w.to_local()` on a DTensor, then derive
`dim1` from the **local** shape while `n_heads_arg` stays **global**:

```python
if isinstance(w, DTensor): w = w.to_local()
if dim1 is None: dim1 = w.shape[0]          # local rows
w.view(n_heads_arg, dim1 // n_heads_arg // 2, 2, dim2)
         ^^^^^^^^^^^ global
```

Each rank re-derives head boundaries from its own slice, so the rows are
regrouped as though the shard were the whole weight.

Confirmed on **real FSDP2** (`fully_shard`, gloo, world=2 — `test_fsdp_save_path.py`):

```
wq.weight  DTensor=True  global=(2048,2048)  placements=(Shard(dim=0),)  local=(1024,2048)
wk.weight  DTensor=True  global=(512,2048)   placements=(Shard(dim=0),)  local=(256,2048)

wq.weight: *** CORRUPT: 1024/2048 rows differ ***
wk.weight: RuntimeError: shape '[4, 64, 2, 2048]' is invalid for input of size 524288
```

So `model.state_dict()` under FSDP2 really does yield `Shard(0)` DTensors, Q is
silently corrupted, and K raises outright. Severity by world size:

| world | wq | wk |
|---|---|---|
| 1 | correct | correct |
| 2 | 1024/2048 rows wrong | raises |
| 4 | 1536/2048 rows wrong | raises |
| 8 | 1792/2048 rows wrong | raises |
| 16 | 1920/2048 rows wrong | raises |
| 12, 24, 48 | ragged shard | ragged shard |

`_permute` also returns a plain tensor, discarding the DTensor sharding
metadata, so each rank would hand DCP a local-sized tensor claiming to be the
whole weight.

### Reachability — why nothing is currently broken

- **`from_hf()` (load) — safe.** `checkpoint.py:589-593` builds `hf_state_dict = {}`
  and lets `dcp.load()` populate it with regular tensors, explicitly "to avoid
  DTensor sharding pattern mismatches". `isinstance(w, DTensor)` is False, so
  `to_local()` never fires. This is the path the FSDP fix was written for, and
  it is correct.
- **`to_hf()` (save) — reachable but not enabled.** Only called from
  `dcp_save(..., to_hf=True)`, gated on `checkpoint.last_save_in_hf`, which
  defaults to `False` and is not set anywhere in the repo or launcher configs.
- **`dcp_to_hf.py` — safe.** Single process, loads from disk, no mesh; tensors
  are regular.

`permute_qk` *is* True for AGPT (`rope_backend` defaults to `"complex"` →
`ComplexRoPE.Config`), so the guard is not what protects us. **The only thing
standing between this and corrupted exports is `last_save_in_hf=False`.**
Turning that flag on at any world size > 1 would produce a silently wrong Q and
a hard failure on K.

A fix would pass the global `dim1` explicitly, or gather before permuting, but
the current call `self._permute(value, n_heads)` omits `dim1` for Q — which is
exactly how the local shape leaks in. Not changed here.

## Files

| file | what it does |
|---|---|
| `test_permute_algebra.py` | round-trip, direction/semantics, bijection; sharding shown as `[defect]`, not a test failure |
| `test_real_weights.py` | round-trips all 24 real q/k tensors from the checkpoint |
| `test_logits_equiv.py` | **the equivalence test** — both conventions end to end, plus sensitivity control |
| `test_shard_geometry.py` | corruption vs world size for the real AGPT-2B shapes |
| `test_sharded_hazard.py` | hazard mechanics and reachability reasoning |
| `test_fsdp_save_path.py` | real FSDP2; run `torchrun --nproc_per_node=2 test_fsdp_save_path.py` |

Run the single-process ones with:
`module load frameworks/2026.1.0; source venv/bin/activate; python3 -I <script>`
(the module load is required or torch fails with `ImportError: libpti_view.so.0`).
