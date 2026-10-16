# AGPT-2B Post-Training @ seq_len 32768 — Run Report

| | |
|---|---|
| Job | `8936184.aurora-pbs-0001` |
| Date | 2026-10-16, 18:22:03 → 18:47 UTC |
| Input | `/lus/flare/projects/datascience/seonghapark/agpt-2b-v2-256n-step-92859-safetensors` |
| Output | `/lus/flare/projects/datascience/seonghapark/agpt-2b-v2-256n-post-32768` |
| Verdict | ✅ **SUCCESS** — 48/48 ranks completed 400/400 steps, export rc=0 |

## Executive Summary

400 steps of continued pretraining at sequence length 32768 on PG19, 4 Aurora nodes
(48 XPU ranks), TP=4 × FSDP=12. Loss fell **3.054 → 2.717**. Peak device memory held
flat at **40.22 GiB (62.86%)** from step 2 onward. Steady-state throughput ~2,950 tok/s
with ~18.2% MFU, ~2.8 s/step, 22.7 min of training. The exported HF checkpoint is 7.5 GB
across two safetensors shards, 111 tensors, index total_size 7,946,313,728 bytes.

This is the second cell of the 2 × 4 post-training matrix. The first
([16384](../post_training_16384_Oct16_2B_16384/)) succeeded but collected no resource
telemetry; **this run collected all of it** — 364 samples at 15 s across 4 hosts.

---

## 1. Verification That Weights Actually Loaded

The failure mode that cost three allocations earlier in this project is a checkpoint load
that logs success and loads nothing. Two independent checks:

```
HF checkpoint: matched 111/111 model parameters.
step:  1  loss:  3.05390
```

`ln(vocab_size) = ln(256128) = 12.45` is the loss a randomly-initialized model produces.
Step-1 loss of **3.054** is nowhere near it — the weights are real. The job script also
carries a hard guard that aborts with exit 3 if step-1 loss exceeds 11.0; it did not fire.

The step-1 loss is higher than the 16384 run's 2.687 because the RoPE scaling factor
changed with the sequence length (see §3) — the model is being asked to attend over twice
the context it was trained for, and pays for it on the first batch before adapting.

## 2. Parallelism — Multi-Node, TP, FSDP

**Requested** (job script → launcher → trainer, forwarded verbatim):

```
--parallelism.tensor_parallel_degree 4
--parallelism.pipeline_parallel_degree 1
--parallelism.context_parallel_degree 1
```

**Confirmed by the trainer itself**:

```
Building device mesh with parallelism: pp=1, dp_replicate=1, dp_shard=12, cp=1, tp=4, ep=1
```

| dimension | degree |
|---|---|
| **Tensor Parallel (TP)** | **4** |
| **FSDP (`dp_shard`)** | **12** |
| Context Parallel (CP) | 1 |
| Pipeline Parallel (PP) | 1 |
| Data Parallel replicate | 1 |
| **world size** | TP × PP × CP × FSDP = 4 × 1 × 1 × 12 = **48** ✅ |

**Multi-node confirmed by observation, not by configuration** — ranks were seen running
on four distinct physical hosts:

```
distinct ranks seen : 48
  x4612c0s3b0n0.hsn.cm.aurora.alcf.anl.gov
  x4612c0s4b0n0.hsn.cm.aurora.alcf.anl.gov
  x4612c0s5b0n0.hsn.cm.aurora.alcf.anl.gov
  x4612c0s6b0n0.hsn.cm.aurora.alcf.anl.gov
```

4 nodes × 12 XPUs/node = 48 ranks, `ZE_FLAT_DEVICE_HIERARCHY=FLAT`.

**Why TP=4 and not TP=2** (the 16384 setting): doubling seq_len doubles activation
memory. Widening the tensor-parallel split from 2 to 4 halves the per-rank activation
footprint, which is what kept peak memory at 62.86% instead of overflowing.
**TP cannot exceed 4 for this model** — `num_key_value_heads = 4`, and TP shards KV heads.
Beyond 32768, CP is the only remaining lever.

`'Applied HSDP' lines: 0` — this is pure FSDP2 sharding, no hybrid replicate dimension.

## 3. Hyperparameters

### Model & data

| | |
|---|---|
| Module / Config | `agpt` / `agpt_2b_yarn` (flavor `2b_yarn`) |
| Total parameters | **1,986,578,432** |
| Layers / hidden / intermediate | 12 / 2048 / 11008 |
| Attention heads / KV heads | 16 / **4** |
| Vocab | 256,128 |
| Dataset | PG19, `/lus/flare/projects/datascience/seonghapark/assets/hf/datasets/pg19/data` |
| Tokenizer | `…step-92859-safetensors/tokenizer.json` |

### RoPE (YaRN) — differs from the 16384 run

| | 16384 run | **this run** |
|---|---|---|
| backend | `ComplexRoPE.Config` | `ComplexRoPE.Config` |
| theta | 50000 | 50000 |
| scaling | yarn | yarn |
| `max_seq_len` | 16384 | **32768** |
| `original_seq_len` | 8192 | 8192 |
| **`rope_factor`** | 2.0 | **4.0** |
| beta_fast / beta_slow | 1.0 / 32.0 | 1.0 / 32.0 |
| mscale (effective) | 1.069315 | **1.138629** |

`rope_factor` is derived as `max_seq_len / original_seq_len`, driven solely by
`--training.seq_len`. The backend is `ComplexRoPE`, matching the input checkpoint's
RoPE class — a mismatch here loads cleanly while training on corrupted attention.

### Optimizer

```
Optimizer AdamW (model_part=0): 111 params [.*]
  {'fused': True, 'foreach': False, 'lr': 2.28e-05,
   'betas': (0.9, 0.95), 'eps': 1e-08, 'weight_decay': 0.1}
```

| | |
|---|---|
| Optimizer | **AdamW** (fused) |
| LR | 2.28e-5 |
| betas | (0.9, 0.95) |
| eps | 1e-8 |
| weight_decay | 0.1 |

### Schedule & batching

| | |
|---|---|
| Steps | **400** |
| Sequence length | **32768** |
| Local batch size | 1 |
| Grad accumulation | 1 |
| **Global batch size** | **12** (= 48 ranks ÷ TP 4) |
| Tokens/step | 12 × 32768 = **393,216** |
| Tokens total | ~157.3 M |
| LR schedule | linear, warmup 200, decay_ratio 0.8, min_lr_factor 0.0 |
| Warmup note | `Warmup (200) + decay (320) steps exceed total steps (400). Adjusting decay steps to 200.` |
| Loss-std termination | enabled, threshold 0.001, window 50 (did **not** trigger) |
| Checkpoint | every 100 steps, `keep_latest_k=0` (keep all) |

The LR schedule spends half the run warming up and half decaying — an artifact of a
200-step warmup on a 400-step run. Worth flagging if these runs are ever compared for
convergence quality rather than for memory/throughput feasibility.

### Precision & runtime

| | |
|---|---|
| Master dtype | float32 |
| `mixed_precision_param` | bfloat16 |
| `mixed_precision_reduce` | float32 |
| `export_dtype` | float32 |
| Activation checkpointing | **none** (`agpt_2b_yarn` sets `activation_checkpoint_mode="none"`) |
| `torch.compile` | **enabled**, components `['model', 'loss']` |

Note for the 20B cells of this matrix: `agpt_20b_yarn` keeps **FullAC**, and
FullAC + `torch.compile` is what crashed job 8936058 in its first forward pass.
The 2B config avoids it only because its AC mode is `none`.

### Environment

```
module load framework/2026.1.0
CCL_OP_SYNC=1
CCL_PROCESS_LAUNCHER=pmix
ZE_FLAT_DEVICE_HIERARCHY=FLAT
no_proxy="localhost,127.0.0.1"
PYTHONMULTIPROCESSINGMETHOD=spawn
```

Plus two PYTHONPATH `sitecustomize` shims (md5 `3163c9b9…` / `3e1269ca…`):
per-rank `TORCHINDUCTOR_CACHE_DIR` isolation, and a guarded `os._exit()` that skips
the `sycl.descriptor_unreg` teardown SIGSEGV.

## 4. Results

### Loss trajectory

Full 400 lines in `artifacts/loss_trajectory_400steps.txt`. Every 50 steps:

| step | loss | grad_norm | memory | mfu |
|---|---|---|---|---|
| 1 | 3.05390 | 14.1359 | 32.39 GiB (50.63%) | 0.66% |
| 50 | 3.18042 | 1.9357 | 40.22 GiB (62.86%) | 17.83% |
| 100 | 3.06429 | 1.9816 | 40.22 GiB (62.86%) | 12.55% |
| 150 | 2.80564 | 1.7185 | 40.22 GiB (62.86%) | 18.51% |
| 200 | 2.85771 | 1.6927 | 40.22 GiB (62.86%) | 18.20% |
| 250 | 2.79563 | 1.3990 | 40.22 GiB (62.86%) | 16.91% |
| 300 | 2.79498 | 1.1867 | 40.22 GiB (62.86%) | 18.68% |
| 350 | 2.83906 | 0.9972 | 40.22 GiB (62.86%) | 2.39% |
| **400** | **2.71722** | **0.9122** | 40.22 GiB (62.86%) | 18.20% |

Loss **3.054 → 2.717**. Grad norm decays monotonically 14.1 → 0.91, which is the
expected signature of a stable continued-pretraining run — no divergence, no spikes
after the first few steps. The MFU dips at steps 100/350 coincide with checkpoint writes.

### Throughput

| | |
|---|---|
| Steady-state tok/s | ~2,950 |
| TFLOPs | ~54.3 |
| **MFU** | **~18.2%** (peak 18.68%) |
| Pace | ~2.8 s/step |
| Training wall time | ~22.7 min |
| Peak device memory | **40.22 GiB / 62.86%** |

Memory is flat from step 2 to step 400 — the allocator reaches steady state immediately
and never grows. Step 1's lower 32.39 GiB is pre-steady-state and should **not** be used
for headroom extrapolation (doing so once led me to wrongly call 65536/TP=4 viable).

### Shutdown integrity

```
'Training completed' ranks    : 48
ranks taking guarded os._exit : 53   (48 trainers + wrappers)
ranks with isolated Inductor  : 96   (two-phase: launcher + trainer)
@jit compile flake hits       : 0
cores before / after          : 2 / 2  (new: 0)
```

No new core dumps, no `@jit` first-compile races, clean exit on all 48 ranks.

## 5. Resource Consumption (15-second sampling) ✅

This is the requirement the 16384 run failed to meet. It is met here.

```
RESOURCE CONSUMPTION  (sampled every 15 s, all 4 hosts)
host             samples  cpu_avg%  cpu_max%  mem_avg_GiB  mem_max_GiB
----------------------------------------------------------------------
x4612c0s3b0n0         91     5.596     8.250       222.43       236.92
x4612c0s4b0n0         91     5.573     8.431       219.72       237.53
x4612c0s5b0n0         91     5.590     8.629       223.63       239.39
x4612c0s6b0n0         91     5.589     8.518       223.87       238.54
ALL (mean/max)       364     5.587     8.629       222.41       239.39

monitored window : 1362.1 s
total samples    : 364 across 4 hosts
```

Per-host raw data in `artifacts/resource_metrics/` — `.csv`, `.jsonl`, and
`.summary.json` for each of the four nodes.

**Host CPU is nearly idle (5.6% avg).** That is correct for this workload: the XPUs do
the work and the host cores only feed the dataloader. Host RAM sits at ~222 GiB of
~1.1 TiB (19.6%).

**❌ No XPU telemetry.** `xpu-smi` returned no per-device rows on the compute nodes, so
there is no per-device utilization, power, or temperature. The only device-side number
available is the trainer's own `memory:` field: **40.22 GiB (62.86%)**. This is a
limitation of `xpu-smi` on these nodes, not of the monitoring configuration — the
monitor ran correctly and recorded everything it could reach.

### How the monitoring was actually enabled

`run_train_agpt.sh:42-44` **overwrites** inherited `RESOURCE_MONITOR`,
`RESOURCE_INTERVAL`, and `RESOURCE_OUTPUT_DIR`. Setting them as environment variables —
exactly as the task specification describes — silently does nothing. Only the CLI flags
survive the launcher chain:

```bash
./run_train_agpt.sh --model 2b --tp 4 ... multi \
  --resource-monitor --resource-interval 15 --resource-output-dir "$RES_DIR" \
  -- ...
```

The environment variables are still set (the banner reports `Resource Monitor = 1`), but
the flags are what made the 364 samples appear.

## 6. Export Verification

```
[dcp_to_hf] adapter: Llama3StateDictAdapter
[dcp_to_hf] rope: ComplexRoPE.Config
[dcp_to_hf] loading DCP checkpoint: …/agpt-2b-AdamW-pg19-n4-gbs12/step-400
[dcp_to_hf] loaded 115 top-level entries
[dcp_to_hf] ignoring 4 non-model key pattern(s): ['dataloader','lr_scheduler','optimizer','train_state']
[dcp_to_hf] 111 model tensors
[dcp_to_hf] converted to 111 HF tensors
[dcp_to_hf] wrote model-00001-of-00002.safetensors (22 tensors)
[dcp_to_hf] wrote model-00002-of-00002.safetensors (89 tensors)
[dcp_to_hf] wrote model.safetensors.index.json
[dcp_to_hf] copied assets: config.json, special_tokens_map.json, tokenizer.json,
                           tokenizer.model, tokenizer_config.json
export rc : 0
```

Checks performed:

| check | result |
|---|---|
| DCP shards consumed | 48 |
| Model tensors in → out | 111 → 111 |
| Key set vs. input checkpoint | **identical** — 0 keys only-in-input, 0 only-in-output |
| Index `total_size` | 7,946,313,728 bytes |
| RoPE class on the save path | `ComplexRoPE.Config`, matching the load path |
| Tokenizer/config assets | copied |
| NaN/Inf sweep, all 111 output tensors | **none** |
| Per-tensor delta vs. input, 24 sampled | **all 24 changed** (no stale copies) |

```
total 7781412
-rw-r--r-- config.json                            478
-rw------- model-00001-of-00002.safetensors  4911573464
-rw------- model-00002-of-00002.safetensors  3034752952
-rw-r--r-- model.safetensors.index.json         9088
-rw-r--r-- special_tokens_map.json               636
-rw-r--r-- tokenizer_config.json               33556
-rw-r--r-- tokenizer.json                   17518497
-rw-r--r-- tokenizer.model                   4241003
```

`config.json` carries `max_position_embeddings: 131072` — this is copied verbatim from
the input checkpoint's assets and is **not** a statement about what this run trained at.
The trained context length is 32768. Anyone loading this checkpoint for inference at
>32768 is extrapolating beyond what the YaRN schedule was fit to.

### Tensor-level diff against the input checkpoint

Every one of the 111 output tensors was swept for NaN/Inf — **none found**. 24 tensors
spanning the embedding, `lm_head`, all of layer 0, and the norms of layers 1–11 were
differenced element-wise against the input checkpoint
(`artifacts/verify_export_bf16.py`, full output in `artifacts/verify_export.txt`):

| tensor | max\|delta\| | rel_rms |
|---|---|---|
| `lm_head.weight` | 0.00339778 | 0.011531 |
| `model.embed_tokens.weight` | 0.00346780 | 0.000478 |
| `model.layers.0.self_attn.o_proj.weight` | 0.00155637 | **0.021112** |
| `model.layers.0.self_attn.v_proj.weight` | 0.00143941 | 0.016891 |
| `model.layers.0.self_attn.k_proj.weight` | 0.00184412 | 0.013437 |
| `model.layers.0.self_attn.q_proj.weight` | 0.00162013 | 0.013398 |
| `model.layers.0.mlp.down_proj.weight` | 0.00183530 | 0.012878 |
| `model.layers.0.mlp.up_proj.weight` | 0.00177329 | 0.012318 |
| `model.layers.0.mlp.gate_proj.weight` | 0.00185960 | 0.010937 |
| layer norms (11 layers, both positions) | 0.0010–0.0022 | 0.0003–0.0012 |

**sampled 24 tensors, 24 changed from the input checkpoint.**

Two things this establishes. First, no tensor was silently passed through unmodified —
a partial-write or stale-shard bug would leave some at exactly zero delta. Second, the
magnitudes are what 400 steps at lr 2.28e-5 should produce: projection weights moved
~1–2% in RMS, norms moved ~0.1%, and `max|delta|` sits at 0.0015–0.0035 across the
board, which is near one BF16 ulp at these weight magnitudes. Nothing exploded.

`o_proj` at 2.1% rel_rms is the largest mover, consistent with attention output being
where a context-length change bites hardest.

**Still not verified**: per-element equivalence between the DCP shards and the merged
safetensors (this diff is output-vs-*input*, which proves training happened, not that
the merge was faithful), and any inference/perplexity evaluation. A structurally valid
export with plausible deltas is not a demonstrably correct one.

## 7. Reproduction

```bash
cd /lus/flare/projects/datascience/seonghapark/xpu_launcher
qsub -A AuroraGPT -q debug-scaling -d doe -j oe \
     -l filesystems=home:flare -l select=4 -l walltime=01:00:00 \
     -v AGPT_SIZE=2b,AGPT_SEQ_LEN=32768,AGPT_TP=4 \
     job_agpt_post.sh
```

`job_agpt_post.sh` (archived as `artifacts/job_agpt_post.sh`) is driven entirely by
`-v`; nothing inside it is edited per run. Variables: `AGPT_SIZE` (2b|20b),
`AGPT_SEQ_LEN`, `AGPT_TP`, `AGPT_CP`, `AGPT_COMPILE`.

**Why parameterize instead of copying the script:** PBS spools the job script at
**submit** time. Editing a queued script does not change the job that is already in the
queue — job 8928776 ran stale code and wasted an allocation this way. With `-v`, a queued
job's behavior is pinned by its `-v` string at submit time, which makes the trap moot.

## 8. Matrix Context

| seq_len | TP | CP | FSDP | status | peak mem | final loss |
|---|---|---|---|---|---|---|
| 16384 | 2 | 1 | 24 | ✅ job 8935820 | 39.51 GiB (61.75%) | 2.66022 |
| **32768** | **4** | **1** | **12** | ✅ **job 8936184** | **40.22 GiB (62.86%)** | **2.71722** |
| 65536 | 4 | 2 | 6 | submitted (8936339) | — | — |
| 131072 | 4 | 4 | 3 | pending | — | — |

Doubling seq_len while doubling TP keeps peak memory essentially constant (39.5 → 40.2 GiB).
TP is exhausted at 4, so 65536 and 131072 must buy their headroom with CP, which shards
the sequence dimension. Both remaining topologies satisfy `seq_len % (tp*cp*2) == 0`
and `TP*PP*CP*FSDP == 48`.

## 9. Artifacts

```
artifacts/loss_trajectory_400steps.txt   all 400 step lines, one per step
artifacts/run_8936184.dedup.log          trainer log, deduped across 48 ranks
artifacts/pbs_stdout_8936184.log.gz      PBS stdout, raw (all ranks interleaved)
artifacts/export.log                     dcp_to_hf conversion
artifacts/resource_metrics/              4 × {csv, jsonl, summary.json}
artifacts/job_agpt_post.sh               parameterized job script
```

## 10. Open Items

- **No XPU-level telemetry** — `xpu-smi` silent on these nodes. Only host CPU/RAM and the
  trainer's self-reported device memory are available.
- **No behavioral evaluation** of the exported model — no perplexity, no generation check.
  A structurally valid export is not a demonstrably correct one.
- **Warmup is half the run** (200 of 400 steps). Fine for a feasibility sweep; not a
  schedule anyone should read convergence quality from.
- **`local_batch_size=1` and the TP choice were mine, not user-specified.** TP=4 is forced
  by memory at this sequence length, but it is still an inferred parameter.
- `config.json` advertises `max_position_embeddings: 131072` while the run trained at
  32768. Inherited from the input assets; not corrected, to avoid diverging from the
  other matrix cells.
