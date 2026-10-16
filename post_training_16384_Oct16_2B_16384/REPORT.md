# AGPT-2B Post-Training @ seq_len 16384 — Run Report

**Date**: 2026-10-16
**Job**: `8935820` (succeeded) — 4 nodes, 48 ranks, Aurora `debug-scaling`
**Log dir**: `/lus/flare/projects/datascience/seonghapark/xpu_launcher/post16384_20261016_155450`
**Output model**: `/lus/flare/projects/datascience/seonghapark/agpt-2b-v2-256n-post-16384`

---

## Executive Summary

**Status**: ✅ **SUCCESS** — 400/400 steps, pretrained weights verified loaded, model exported.

- ✅ 400 steps at seq_len 16384 on 4 nodes / 48 ranks, 30.5 min wall
- ✅ Pretrained weights confirmed loaded (`111/111` params; step-1 loss **2.69** vs random-init 12.45)
- ✅ TP=2, FSDP `dp_shard`=24 — verified in the device-mesh line
- ✅ Exported to HF safetensors; weights verified changed-but-finite vs. the input checkpoint
- ❌ **Resource monitoring collected nothing** — launcher drops the env var (§7). The one unmet requirement.

**Three allocations were burned first** (8935565, 8935642, 8935726) on three distinct bugs in the
HF checkpoint-load path. §6 documents them; the first one is the dangerous one.

---

## 1. Verification That Weights Actually Loaded

This matters more than it sounds. The *first* attempt at this run (job 8935565) logged a successful
checkpoint load, raised nothing, took a plausible 1.12 s — and loaded **zero** weights. It trained a
randomly-initialized model for 250 steps before being killed.

| Check | Value | Interpretation |
|---|---|---|
| `HF checkpoint: matched N/M model parameters` | **111/111** | structural: every param found |
| **step-1 loss** | **2.68685** | semantic: only reachable from trained weights |
| `ln(vocab_size)` = `ln(256128)` | 12.45 | what random init produces |
| job 8935565 step-1 loss (random init) | 13.01 | the failure mode, for contrast |

**The loss value is the real proof.** `111/111` is necessary but not sufficient — it says the keys
matched, not that the numbers arrived. Always check step-1 loss against `ln(vocab_size)` before
trusting a continued-pretraining run.

---

## 2. Parallelism — Multi-Node, TP, FSDP

Scheduler-level (4 distinct physical nodes, 208 CPUs each):

```
exec_host = x4710c6s1b0n0/0*208 + x4710c6s2b0n0/0*208
          + x4710c6s4b0n0/0*208 + x4710c6s6b0n0/0*208
```

Framework-level, logged by all 48 ranks:

```
Building device mesh with parallelism: pp=1, dp_replicate=1, dp_shard=24, cp=1, tp=2, ep=1
```

| Dimension | Degree |
|---|---|
| **Tensor Parallel (TP)** | **2** |
| **FSDP shard (`dp_shard`)** | **24** |
| Data parallel replicate | 1 |
| Pipeline / Context / Expert | 1 / 1 / 1 |

`24 × 2 = 48 ranks = 4 nodes × 12 XPUs` ✓. FSDP2 with `enable_sequence_parallel: true`.
`data_parallel_shard_degree: -1` in config means "infer", and it resolved to 24.

> ⚠️ **TP=2 and local_batch_size=1 were chosen by Claude, not specified by the user.** They
> produced a healthy run, but they are not user-approved values. LBS=1 is the only value measured
> at seq_len 16384.

---

## 3. Hyperparameters

### Model & data
| Parameter | Value |
|---|---|
| Module / config | `agpt` / `agpt_2b_yarn` (flavor `2b_yarn`) |
| Total parameters | **1,986,578,432** |
| Layers | 12 |
| Vocab size | 256,128 |
| RoPE | ComplexRoPE (YaRN flavor) |
| Init checkpoint | `agpt-2b-v2-256n-step-92859-safetensors` |
| Dataset | **PG19**, local: `assets/hf/datasets/pg19/data` |

### Optimizer
| Parameter | Value |
|---|---|
| **Optimizer** | **AdamW** (`implementation: fused`) |
| **Learning rate** | **2.28e-5** |
| Betas | (0.9, 0.95) |
| Eps | 1e-8 |
| Weight decay | 0.1 |
| Param groups | 1, pattern `.*` |

### Schedule & batching
| Parameter | Value |
|---|---|
| **Steps** | **400** |
| **Sequence length** | **16384** |
| Warmup steps | 200 |
| Decay type / ratio | linear / 0.8 |
| min_lr_factor | 0.0 |
| Local batch size | 1 |
| **Global batch size** | **24** |
| Gradient accumulation | 1 |
| Grad clip (`max_norm`) | 1.0 |

### Precision & runtime
| Parameter | Value |
|---|---|
| Param dtype (mixed precision) | bfloat16 |
| Reduce dtype | float32 |
| Base dtype | float32 |
| GC frequency | 50 |
| Checkpoint interval | 100 (`keep_latest_k: 0`) |
| Loss-std termination | enabled, threshold 0.001, window 50 — **did not trigger** |
| max_duration_hours | 4.0 |

### Environment
`CCL_OP_SYNC=1`, `CCL_PROCESS_LAUNCHER=pmix`, `ZE_FLAT_DEVICE_HIERARCHY=FLAT`,
`PYTHONMULTIPROCESSINGMETHOD=spawn`, per-rank `TORCHINDUCTOR_CACHE_DIR` isolation,
guarded `os._exit()` teardown shim. Module `frameworks/2026.1.0`.

---

## 4. Results

### Loss trajectory (full 400 steps in `artifacts/loss_trajectory_400steps.txt`)

```
step   1  loss 2.68685  grad_norm  5.0290
step  10  loss 3.20060  grad_norm  6.2469
step  50  loss 2.82143  grad_norm  1.4540
step 100  loss 2.76540  grad_norm  1.1851
step 200  loss 2.77131  grad_norm  1.6914
step 300  loss 2.81074  grad_norm  2.1425
step 390  loss 2.74124  grad_norm  0.8303
step 400  loss 2.66022  grad_norm  0.7282
```

Final: **loss 2.66022**, grad_norm 0.728, global_max_loss 3.079.

The curve is **flat-ish**, which is correct for post-training an already-converged model — contrast
the random-init run's 13.0 → 10.4 → 8.1 → 6.1 plunge. Grad-norm settling from 5.03 → 0.73 is the
healthier signal here. 400 steps at global batch 24 is a short run; no convergence claim is implied.

### Throughput
| Metric | Value |
|---|---|
| Throughput | ~4,500 tokens/sec |
| TFLOPs | ~61 |
| **MFU** | **~20.5%** |
| Peak memory | 39.51 GiB (61.75%) |
| Max active memory | 30.02 GiB (46.92%) |
| Wall time | 30.5 min (15:57:05 → 16:27:36) |
| Steps completed | **400 / 400**, "Training completed" on 48/48 ranks |

### Shutdown integrity
| Check | Result |
|---|---|
| Guarded `os._exit` | 49 ranks |
| Isolated Inductor cache | 48 ranks |
| `@jit` compile flakes | **0** |
| New core files | **0** (2 before / 2 after) |
| Alloc retries / OOMs | 0 / 0 |

---

## 5. Export Verification

`dcp_to_hf.py` converted the step-400 DCP checkpoint (48 shards, 23 GB) to HF safetensors.

```
[dcp_to_hf] adapter: Llama3StateDictAdapter
[dcp_to_hf] rope: ComplexRoPE.Config
[dcp_to_hf] loaded 115 top-level entries
[dcp_to_hf] ignoring 4 non-model key pattern(s): dataloader, lr_scheduler, optimizer, train_state
[dcp_to_hf] 111 model tensors -> converted to 111 HF tensors
[dcp_to_hf] wrote model-00001-of-00002.safetensors (22 tensors)
[dcp_to_hf] wrote model-00002-of-00002.safetensors (89 tensors)
```

Output (7.5 GB): 2 safetensors shards, `model.safetensors.index.json` (111 tensors, 7.95 GB),
plus `config.json`, `tokenizer.json`, `tokenizer.model`, `tokenizer_config.json`,
`special_tokens_map.json`.

**Diff vs. the input checkpoint** — key sets identical (111 ↔ 111); sampled tensors all moved by a
small, finite, training-sized amount, no NaN/Inf:

| Tensor | Shape | max abs Δ | relative |
|---|---|---|---|
| `lm_head.weight` | (256128, 2048) | 3.379e-03 | 0.41% |
| `model.embed_tokens.weight` | (256128, 2048) | 4.090e-03 | 0.07% |
| `model.layers.6.self_attn.k_proj.weight` | (512, 2048) | 2.769e-03 | 1.35% |
| `model.layers.11.mlp.down_proj.weight` | (2048, 11008) | 1.544e-03 | 0.38% |
| `model.norm.weight` | (2048,) | 1.498e-03 | 0.08% |

This rules out both failure directions: the export is neither a copy of the input (Δ > 0) nor
corrupted (Δ small, finite, no NaN).

> **Not verified**: no inference / perplexity / generation check was run on the exported model.
> Weight-level sanity is not the same as behavioral correctness.

---

## 6. Bugs Found and Fixed (3 failed allocations)

All three were in `hf_load()` in the **vendored** repo
(`xpu_torchtitan/torchtitan_repo/torchtitan/components/checkpoint.py`), which shadows the
installed wheel. Each fix exposed the next bug.

### Bug 1 — `hf_state_dict = {}` silently trained from scratch (job 8935565) ⚠️ **worst**

```python
hf_state_dict = {}                                     # <-- the bug
dcp.load(hf_state_dict, storage_reader=hf_storage_reader)
state_dict_native = self.sd_adapter.from_hf(hf_state_dict)
self.states[MODEL].load_state_dict(state_dict_native)  # no-op
```

`dcp.load()` **plans from the keys already present** in the dict it is given. An empty dict plans
nothing and reads nothing. Measured directly: **0 of 111** tensors loaded.
`_EmptyStateDictLoadPlanner` does not help with the HF reader (also 0).

It logged `Finished loading the checkpoint in 1.12 seconds` on all 48 ranks and raised nothing.
Killed at step ~250 before it could overwrite the user's output path.

**Fix** — build the load template from the checkpoint's own metadata:

```python
hf_storage_reader = self.sd_adapter.get_hf_storage_reader(checkpoint_id, from_quantized)
hf_storage_reader.reset()
metadata = hf_storage_reader.read_metadata()
hf_state_dict = {
    name: torch.empty(tuple(md.size), dtype=md.properties.dtype)
    for name, md in metadata.state_dict_metadata.items()
}
dcp.load(hf_state_dict, storage_reader=hf_storage_reader)
```

Deliberately **not** upstream's `to_hf(state_dict)` approach: `to_hf → _permute` calls `to_local()`
and derives head boundaries from the *local* shape while `n_heads` stays global, corrupting Q and
hard-failing on K at world > 1. Reading shapes from metadata avoids `to_hf()` entirely and leaves
`state_dict_adapter.py` untouched.

### Bug 2 — mixed Tensor/DTensor on copy (job 8935642)

With real tensors finally flowing, all 48 ranks died (5,305 occurrences):

```
aten.copy_.default got mixed torch.Tensor and DTensor,
need to convert all torch.Tensor to DTensor before calling distributed operators
```

`from_hf()` returns plain full tensors; under FSDP2/TP the model's params are DTensors and
`load_state_dict` ends in `param.copy_(loaded)`. **This is the exact "DTensor sharding mismatch" the
original `{}` comment cited** — the empty dict never solved it, it just left nothing to copy.

**Fix** — reshard onto each target param's own mesh/placements before copying.

### Bug 3 — `_StridedShard` AssertionError (job 8935726)

The Bug-2 fix then failed on all 49 ranks with a bare `AssertionError` at
`placement_types.py:993`. Under FSDP2+TP composition the placements are `_StridedShard`, whose
default path scatters from a source rank and asserts all padded chunks share a shape.

**Fix** — `src_data_rank=None`, which is **required, not an optimization**:

```python
model_sd = self.states[MODEL].state_dict()
for key, loaded in state_dict_native.items():
    target = model_sd.get(key)
    if isinstance(target, DTensor) and not isinstance(loaded, DTensor):
        state_dict_native[key] = distribute_tensor(
            loaded.to(target.dtype), target.device_mesh, target.placements,
            src_data_rank=None)
```

`None` means "every rank already holds this data, just split it locally" — exactly true here,
since `dcp.load()` gave every rank the identical full tensor. It takes a split-only branch that
calls the same `_split_tensor`, so the shard layout is identical; only the collective and its
padding assertion are skipped. It also avoids 111 needless collectives.

**Validated on CPU/gloo before spending a 4th allocation** (`artifacts/test_strided.py`,
`artifacts/test_loadsd.py`):

| Check | Result |
|---|---|
| Old call reproduces the failure | bare `AssertionError` on all 4 ranks |
| New call | correct shards; sums total 435 = `sum(0..29)` |
| `full_tensor()` round-trip | exact, max abs diff **0** |
| Reshard + `load_state_dict` into DTensor params | `matched 3/3`, no copy error |
| Params hold the **intended values** | max deviation **0** — "WEIGHTS ARRIVED" |

The modules were initialized to **zeros**, not random, so a silent no-op would have been detectable.

### Guards added
Three independent checks now stand between a bad run and the output path:
1. `hf_load` raises if `matched == 0` and logs `matched N/M model parameters` on success
2. The job script aborts the export if step-1 loss > 11.0 (reads `run.clean.log`, ANSI-stripped)
3. The `copy_` DTensor error is no longer masked by an empty dict

---

## 7. ❌ Unmet Requirement: 15-Second Resource Monitoring

**No resource data was collected for this run.** The banner printed `Resource Monitor = 0` despite
the job exporting `RESOURCE_MONITOR=1`, and no `resource_metrics` directory was created.

**Root cause** — `xpu_torchtitan/run_train_agpt.sh` lines 42-44 run unconditionally at startup:

```bash
RESOURCE_MONITOR=""
RESOURCE_INTERVAL=""
RESOURCE_OUTPUT_DIR=""
```

This **clobbers any inherited environment value** before parsing. Lines 459-461 re-export them only
`[[ -n ... ]]`, so an exported `RESOURCE_MONITOR=1` arrives empty and
`run_train_torchtitan.sh:67` defaults it to `0`. **Only the CLI flags work.**

**Fix applied** to `job_agpt2b_post16384.sh` — pass them as flags:

```bash
./run_train_agpt.sh --model 2b --tp 2 ... multi \
  --resource-monitor --resource-interval 15 --resource-output-dir "$RES_DIR" \
  -- --training.steps 400 ...
```

Flag placement after the mode word is correct: the parser's `single|multi)` branch does not break,
and its hostfile grab skips anything matching `^--`. Confirmed by login-node dry-run:

```
Resource Monitor       = 1
Resource Interval      = 15s
Resource Output Dir    = /tmp/rm_test
```

**Collecting the telemetry requires another allocation.** The trained model is already complete and
exported, so such a run would be purely for the resource profile. Not submitted.

Indirect resource data that *is* available: peak memory 39.51 GiB (61.75%) per rank, max active
30.02 GiB, ~61 TFLOPs, ~20.5% MFU, 0 OOMs, 0 alloc retries.

`agg_resources.py` (in `xpu_launcher/`) is ready to aggregate across all 4 nodes once samples
exist; the in-job report previously used `ls | head -1`, reporting one node out of four.

---

## 8. Reproduction

```bash
qsub -A AuroraGPT -q debug-scaling -d doe -j oe \
     -l filesystems=home:flare -l select=4 -l walltime=01:00:00 \
     /lus/flare/projects/datascience/seonghapark/xpu_launcher/job_agpt2b_post16384.sh
```

Job script snapshot: `artifacts/job_agpt2b_post16384.sh` (includes the §7 fix).

> ⚠️ **PBS spools the job script at submit time.** Editing a queued script does not affect that job.

Underlying command:

```
xpu launch --scheduler auto --hostfile <hosts> -nh 4 -n 48 -ppn 12 -- python titan_train.py \
  --module agpt --config agpt_2b_yarn \
  --hf_assets_path .../agpt-2b-v2-256n-step-92859-safetensors \
  --dataloader.dataset PG19 --dataloader.dataset_path .../pg19/data \
  --training.steps 400 --training.seq_len 16384 --training.local_batch_size 1 \
  --parallelism.tensor_parallel_degree 2 \
  --parallelism.pipeline_parallel_degree 1 --parallelism.context_parallel_degree 1 \
  --optimizer.param-groups.0.optimizer-name AdamW \
  --optimizer.param-groups.0.optimizer-kwargs.lr 2.28e-5 \
  --checkpoint.enable --checkpoint.interval 100 --checkpoint.keep_latest_k 0 \
  --training.enable_loss_std_termination --training.loss_std_threshold=0.001 \
  --training.loss_std_window=50
```

Note optimizer and LR are **indexed param-group flags**, not `--optimizer.lr`.

---

## 9. Artifacts

| File | Contents |
|---|---|
| `artifacts/loss_trajectory_400steps.txt` | all 400 deduped step lines (loss, grad_norm, memory, tps, tflops, mfu) |
| `artifacts/export.log` | full `dcp_to_hf` conversion log |
| `artifacts/pbs_stdout_8935820.dedup.log` | PBS stdout, ANSI-stripped and deduped (6,219 lines) |
| `artifacts/pbs_stdout_8935820.log.gz` | full raw PBS stdout (5.2 MB → 81 KB) |
| `artifacts/job_agpt2b_post16384.sh` | job script as run, plus the §7 resource-flag fix |
| `artifacts/test_strided.py` | CPU repro of Bug 3 + proof `src_data_rank=None` fixes it |
| `artifacts/test_loadsd.py` | CPU rehearsal of the full reshard + `load_state_dict` path |

Live run dir (not copied, 9 MB): `xpu_launcher/post16384_20261016_155450/`
DCP checkpoint (23 GB, 48 shards):
`xpu_launcher/xpu_torchtitan/outputs/agpt_2b_20261016_155453/agpt-2b-AdamW-pg19-n4-gbs24/step-400`

---

## 10. Open Items

1. **Resource telemetry** (§7) — fix applied, needs an allocation to collect.
2. **TP=2 / LBS=1 unapproved** (§2) — Claude's choices, never confirmed by the user.
3. **No behavioral eval** (§5) — weights verified at tensor level only; no perplexity or generation check.
4. **`data_parallel_shard_degree: -1`** resolved to 24 here; it is inferred, so it changes with node count.
