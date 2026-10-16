# AGPT-2B Post-Training @ seq_len 65536 — Run Report

| | |
|---|---|
| Job | `8936339.aurora-pbs-0001` |
| Date | 2026-10-16, 19:23 → 19:53 UTC |
| Input | `/lus/flare/projects/datascience/seonghapark/agpt-2b-v2-256n-step-92859-safetensors` |
| Output | `/lus/flare/projects/datascience/seonghapark/agpt-2b-v2-256n-post-65536` |
| Verdict | ✅ **SUCCESS** — 48/48 ranks completed 400/400 steps, export rc=0 |

## Executive Summary

400 steps at sequence length 65536 on PG19, 4 Aurora nodes (48 XPU ranks),
**TP=4 × CP=2 × FSDP=6**. This is the first cell of the matrix that required context
parallelism: TP is capped at 4 by the model's 4 KV heads, so the memory for a second
sequence doubling had to come from elsewhere.

It came cheaply. Peak device memory rose **40.22 → 41.14 GiB** (62.86% → 64.30%) for
twice the context, and MFU *improved* from 18.2% to **20.48%**. The cost is in batch
size, not memory: global batch fell 12 → 6, because `gbs = world_size ÷ (TP × CP)`.

Loss settled around **2.8** over the second half of the run. The final-step reading of
2.34374 is an outlier and should not be reported as the result — see §4.

---

## 1. Verification That Weights Actually Loaded

```
HF checkpoint: matched 111/111 model parameters.
step:  1  loss:  2.82747
```

`ln(256128) = 12.45` is the random-init loss. Step-1 at **2.827** is nowhere near it.
The job script's hard guard (abort if step-1 loss > 11.0) did not fire.

## 2. Parallelism — Multi-Node, TP, CP, FSDP

**Requested** → **observed**, from the job's own verification section:

```
--- requested ---
  nodes=4  world=48  TP=4  PP=1  CP=2  FSDP(dp_shard)=6
--- observed: ParallelDims line from the run ---
pp=1, dp_replicate=1, dp_shard=6, cp=2, tp=4, ep=1
```

| dimension | degree |
|---|---|
| **Tensor Parallel (TP)** | **4** |
| **FSDP (`dp_shard`)** | **6** |
| **Context Parallel (CP)** | **2** |
| Pipeline Parallel (PP) | 1 |
| Data Parallel replicate | 1 |
| **world size** | 4 × 1 × 2 × 6 = **48** ✅ |

```
'Applied FSDP' lines : 48
'Applied HSDP' lines : 0
```

Pure FSDP2 sharding across all 48 ranks, no hybrid replicate dimension.

**Multi-node confirmed by observation** — ranks ran on four distinct physical hosts:

```
distinct ranks seen : 48
  x4602c5s7b0n0.hsn.cm.aurora.alcf.anl.gov
  x4710c2s7b0n0.hsn.cm.aurora.alcf.anl.gov
  x4710c3s2b0n0.hsn.cm.aurora.alcf.anl.gov
  x4710c6s7b0n0.hsn.cm.aurora.alcf.anl.gov
```

Note these are a different four nodes than the 32768 run used — PBS allocated from a
different part of the machine, and they are not even contiguous (`x4602` vs `x4710`).
Throughput did not suffer for it.

**Constraint checks**: `seq_len % (tp × cp × 2) == 0` → 65536 % 16 == 0 ✅, and
`TP × PP × CP × FSDP == world_size` → 48 == 48 ✅.

### Why CP=2 was necessary

TP shards KV heads, and this model has **`num_key_value_heads = 4`**. TP=4 is therefore
the hard ceiling. At 32768/TP=4 the run sat at 62.86% device memory with no further TP
headroom available, so 65536 needed a different axis. CP shards the *sequence*
dimension, which is exactly the thing that doubled.

## 3. Hyperparameters

### Model & data

| | |
|---|---|
| Module / Config | `agpt` / `agpt_2b_yarn` (flavor `2b_yarn`) |
| Total parameters | **1,986,578,432** |
| Layers / hidden / intermediate | 12 / 2048 / 11008 |
| Attention heads / KV heads | 16 / **4** |
| Vocab | 256,128 |
| Dataset | PG19, `/lus/flare/.../assets/hf/datasets/pg19/data` |
| Checkpoint folder | `agpt-2b-AdamW-pg19-n4-gbs6` |

### RoPE (YaRN) — scales with seq_len

| | 16384 | 32768 | **65536** |
|---|---|---|---|
| backend | ComplexRoPE | ComplexRoPE | **ComplexRoPE.Config** |
| theta | 50000 | 50000 | 50000 |
| `max_seq_len` | 16384 | 32768 | **65536** |
| `original_seq_len` | 8192 | 8192 | 8192 |
| **`rope_factor`** | 2.0 | 4.0 | **8.0** |
| mscale (effective) | 1.069315 | 1.138629 | **1.207944** |
| beta_fast / beta_slow | 1.0 / 32.0 | 1.0 / 32.0 | 1.0 / 32.0 |

`rope_factor = max_seq_len / original_seq_len`, driven solely by `--training.seq_len`.
The backend is `ComplexRoPE`, matching the input checkpoint's RoPE class — a mismatch
here loads cleanly while training on corrupted attention.

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
| Sequence length | **65536** |
| Local batch size | 1 |
| Grad accumulation | 1 |
| **Global batch size** | **6** (= 48 ÷ (TP 4 × CP 2)) |
| Tokens/step | 6 × 65536 = **393,216** |
| Tokens total | ~157.3 M |
| LR schedule | linear, warmup 200, decay_ratio 0.8, min_lr_factor 0.0 |
| Warmup note | `Warmup (200) + decay (320) steps exceed total steps (400). Adjusting decay steps to 200.` |
| Loss-std termination | enabled, threshold 0.001, window 50 (did **not** trigger) |
| Checkpoint | every 100 steps, `keep_latest_k=0` |

**Tokens per step is identical to 32768** (393,216) — the halved batch exactly cancels
the doubled sequence. So this run and the 32768 run saw the same token count, 157.3 M,
over the same 400 steps. What differs is how those tokens are arranged: 6 sequences of
65536 versus 12 of 32768.

### Precision & runtime

| | |
|---|---|
| Master dtype | float32 |
| `mixed_precision_param` | bfloat16 |
| `mixed_precision_reduce` | float32 |
| `export_dtype` | float32 |
| Activation checkpointing | **none** (`agpt_2b_yarn` sets `activation_checkpoint_mode="none"`) |
| `torch.compile` | **enabled**, components `['model', 'loss']` |

### Environment

```
module load framework/2026.1.0
CCL_OP_SYNC=1
CCL_PROCESS_LAUNCHER=pmix
ZE_FLAT_DEVICE_HIERARCHY=FLAT
no_proxy="localhost,127.0.0.1"
PYTHONMULTIPROCESSINGMETHOD=spawn
```

Plus two PYTHONPATH `sitecustomize` shims: per-rank `TORCHINDUCTOR_CACHE_DIR` isolation,
and a guarded `os._exit()` that skips the `sycl.descriptor_unreg` teardown SIGSEGV.

## 4. Results

### Loss trajectory

Full 400 lines in `artifacts/loss_trajectory_400steps.txt`. Every 50 steps:

| step | loss | grad_norm | memory | mfu |
|---|---|---|---|---|
| 1 | 2.82747 | 24.4812 | 32.94 GiB (51.48%) | 0.61% |
| 2 | 3.63794 | 29.5229 | 41.14 GiB (64.30%) | 19.26% |
| 50 | 3.23831 | 2.2949 | 41.14 GiB (64.30%) | 20.25% |
| 100 | 3.02558 | 3.0565 | 41.14 GiB (64.30%) | 20.60% |
| 150 | 3.02447 | 3.2518 | 41.14 GiB (64.30%) | 20.32% |
| 200 | 2.79296 | 2.0504 | 41.14 GiB (64.30%) | 19.80% |
| 250 | 2.93463 | 1.9057 | 41.14 GiB (64.30%) | 20.97% |
| 300 | 2.81436 | 1.5072 | 41.14 GiB (64.30%) | 20.24% |
| 350 | 2.83402 | 1.2733 | 41.14 GiB (64.30%) | 20.72% |
| **400** | **2.34374** | **1.9154** | 41.14 GiB (64.30%) | 20.48% |

### ⚠️ The final-step loss is an outlier, not a result

Steps 200–350 sit in a band around **2.79–2.93**. Step 400 reads 2.34374 — a 0.49-nat
drop in 50 steps, while grad_norm *rose* from 1.27 to 1.92. A genuine convergence
improvement of that size would come with a falling gradient norm, not a rising one.

At global batch size 6, a single step's loss is an average over six book excerpts. PG19
is public-domain literature with wildly varying per-passage entropy; one easy batch moves
the number this much. The 32768 run, at gbs=12, shows visibly less step-to-step variance.

**The defensible summary of this run is "loss settled around 2.8."** Quoting 2.344 as
the outcome would overstate it, and comparing that figure against the other matrix cells
would compound the error — those cells differ in batch size, `rope_factor`, and effective
context, so they are not evaluating the same objective.

### Throughput

| | |
|---|---|
| Steady-state tok/s | ~2,170 |
| TFLOPs | ~61.1 |
| **MFU** | **~20.5%** (peak 20.97%) |
| Pace | ~4.3 s/step |
| Training wall time | ~29 min |
| Peak device memory | **41.14 GiB / 64.30%** |

Memory is flat at 41.14 GiB from step 2 to step 400. Step 1's 32.94 GiB is
pre-steady-state and must **not** be used for headroom extrapolation.

MFU is the best of the three 2B cells (16384: ~20.5%, 32768: 18.2%, 65536: **20.5%**).
Longer sequences give the collectives more compute to hide behind.

### Shutdown integrity

```
'Training completed' ranks    : 48
ranks taking guarded os._exit : 53
ranks with isolated Inductor  : 96
@jit compile flake hits       : 0
cores before / after          : 2 / 2  (new: 0)
```

Clean exit on all 48 ranks, no new core dumps, no `@jit` first-compile races.

## 5. Resource Consumption (15-second sampling)

```
RESOURCE CONSUMPTION  (sampled every 15 s, all 4 hosts)
host             samples  cpu_avg%  cpu_max%  mem_avg_GiB  mem_max_GiB
----------------------------------------------------------------------
x4602c5s7b0n0        116     5.833     7.309       189.16       201.96
x4710c2s7b0n0        116     5.840     7.144       188.92       201.61
x4710c3s2b0n0        116     5.828     6.844       192.74       206.86
x4710c6s7b0n0        116     5.845     7.130       193.31       206.29
ALL (mean/max)       464     5.837     7.309       191.03       206.86

monitored window : 1737.4 s
total samples    : 464 across 4 hosts
```

Per-host raw data in `artifacts/resource_metrics/` — `.csv`, `.jsonl`, `.summary.json`.

Host CPU ~5.8% (the XPUs do the work; host cores only feed the dataloader). Host RAM
**191 GiB average, down from 222 GiB at 32768** — CP=2 halves the per-rank sequence, so
each rank's host-side dataloader buffers less.

**❌ No XPU telemetry.** `xpu-smi` returned no per-device rows on the compute nodes, so
there is no per-device utilization, power, or temperature. The trainer's self-reported
`memory:` field (41.14 GiB, 64.30%) is the only device-side number. This is an `xpu-smi`
limitation on these nodes, not a monitoring misconfiguration — the monitor ran and
recorded everything it could reach.

Monitoring is enabled through the **CLI flags** `--resource-monitor`,
`--resource-interval 15`, `--resource-output-dir`. The environment variables named in
the task spec (`RESOURCE_MONITOR=1` etc.) are clobbered by `run_train_agpt.sh:42-44` and
do nothing on their own.

## 6. Export Verification

```
[dcp_to_hf] adapter: Llama3StateDictAdapter
[dcp_to_hf] rope: ComplexRoPE.Config
[dcp_to_hf] loading DCP checkpoint: …/agpt-2b-AdamW-pg19-n4-gbs6/step-400
[dcp_to_hf] 111 model tensors
[dcp_to_hf] converted to 111 HF tensors
[dcp_to_hf] wrote model-00001-of-00002.safetensors (22 tensors)
[dcp_to_hf] wrote model-00002-of-00002.safetensors (89 tensors)
[dcp_to_hf] wrote model.safetensors.index.json
[dcp_to_hf] copied assets: config.json, special_tokens_map.json, tokenizer.json,
                           tokenizer.model, tokenizer_config.json
export rc : 0
```

```
-rw-r--r-- config.json                            478
-rw------- model-00001-of-00002.safetensors  4911573464
-rw------- model-00002-of-00002.safetensors  3034752952
-rw-r--r-- model.safetensors.index.json         9088
-rw-r--r-- special_tokens_map.json               636
-rw-r--r-- tokenizer_config.json               33556
-rw-r--r-- tokenizer.json                   17518497
-rw-r--r-- tokenizer.model                   4241003
```

| check | result |
|---|---|
| Model tensors in → out | 111 → 111 |
| Key set vs. input checkpoint | **identical** — 0 only-in-input, 0 only-in-output |
| Shape mismatches | **0** |
| NaN/Inf sweep, all 111 output tensors | **none** |
| Per-tensor delta vs. input, 24 sampled | **all 24 changed** |

| tensor | max\|delta\| | rel_rms |
|---|---|---|
| `lm_head.weight` | 0.0032497 | 0.010741 |
| `model.embed_tokens.weight` | 0.0032301 | 0.000477 |
| `model.layers.0.self_attn.o_proj.weight` | 0.0014593 | **0.021367** |
| `model.layers.0.self_attn.v_proj.weight` | 0.0012299 | 0.018541 |
| `model.layers.0.self_attn.q_proj.weight` | 0.0016106 | 0.013510 |
| `model.layers.0.mlp.down_proj.weight` | 0.0017969 | 0.013002 |
| `model.layers.0.self_attn.k_proj.weight` | 0.0018651 | 0.012783 |
| `model.layers.0.mlp.up_proj.weight` | 0.0015193 | 0.012297 |
| `model.layers.0.mlp.gate_proj.weight` | 0.0015836 | 0.010987 |
| layer norms (11 layers, both positions) | 0.0009–0.0023 | 0.0003–0.0012 |

**sampled 24 tensors, 24 changed from the input checkpoint.** No tensor passed through
unmodified — a stale-shard or partial-merge bug would leave some at exactly zero delta.
The magnitudes are near-identical to the 32768 run's (`o_proj` 2.14% here vs 2.11% there),
which is what equal token counts at equal learning rate should produce.

Full output in `artifacts/verify_export.txt`,
produced by `artifacts/verify_export_bf16.py` (reads safetensors without torch, since
login nodes cannot `import torch`; widens BF16 via `uint16 << 16` because numpy has no
bfloat16 dtype).

`config.json` carries `max_position_embeddings: 131072`, copied verbatim from the input
assets. **This run trained at 65536.** The field is not a statement about this run.

**Not verified**: per-element equivalence between the DCP shards and the merged
safetensors, and any inference/perplexity evaluation. The checks are structural plus a
sanity diff, not behavioral.

## 7. Reproduction

```bash
cd /lus/flare/projects/datascience/seonghapark/xpu_launcher
qsub -A AuroraGPT -q debug-scaling -j oe \
     -l filesystems=home:flare -l select=4 -l walltime=01:00:00 \
     -v AGPT_SIZE=2b,AGPT_SEQ_LEN=65536,AGPT_TP=4,AGPT_CP=2 \
     job_agpt_post.sh
```

Note: the `-d doe` flag from the original task template is **not valid** on this PBS
build (`qsub: invalid option -- 'd'`) and must be omitted.

`job_agpt_post.sh` is driven entirely by `-v`; nothing inside it is edited per run.
This also sidesteps the PBS spool-at-submit trap — PBS snapshots the script at submit
time, so editing a queued script does nothing (job 8928776 ran stale code and wasted an
allocation this way), but a `-v` string is fixed at submit by construction.

## 8. Matrix Context

| seq_len | TP | CP | FSDP | gbs | status | peak mem | MFU | loss (steady) |
|---|---|---|---|---|---|---|---|---|
| 16384 | 2 | 1 | 24 | 24 | ✅ 8935820 | 39.51 GiB (61.75%) | ~20.5% | 2.66 |
| 32768 | 4 | 1 | 12 | 12 | ✅ 8936184 | 40.22 GiB (62.86%) | 18.2% | 2.72 |
| **65536** | **4** | **2** | **6** | **6** | ✅ **8936339** | **41.14 GiB (64.30%)** | **20.5%** | **~2.8** |
| 131072 | 4 | 4 | 3 | 3 | running (8936601) | — | — | — |

The memory story across the sweep: each sequence doubling is absorbed by doubling TP
(up to its cap of 4) and then CP, and peak occupancy creeps only 61.75% → 64.30% across
a 4× context increase. The real cost is global batch size, which falls 24 → 12 → 6 → 3.

At 131072/CP=4 the global batch will be **3**, which makes per-step loss noisier still.
Any readout from that cell should be a trend over the last ~100 steps, not a final value.

## 9. Artifacts

```
artifacts/loss_trajectory_400steps.txt   all 400 step lines, one per step
artifacts/run_8936339.dedup.log          trainer log, deduped across 48 ranks
artifacts/pbs_stdout_8936339.log.gz      PBS stdout, raw
artifacts/export.log                     dcp_to_hf conversion
artifacts/resource_metrics/              4 × {csv, jsonl, summary.json}
artifacts/job_agpt_post.sh               parameterized job script
artifacts/verify_export_bf16.py          torch-free safetensors differ
artifacts/verify_export.txt              its output
```

## 10. Open Items

- **Final-step loss 2.344 is an outlier** — do not quote it as the run's result (§4).
- **No XPU-level telemetry** — `xpu-smi` silent on these nodes.
- **No behavioral evaluation** of the exported model.
- **Warmup is half the run** (200 of 400 steps).
- **gbs=6 makes per-step loss noisy**; a convergence claim would need either more steps
  or gradient accumulation to restore effective batch size.
- `config.json` advertises `max_position_embeddings: 131072` while this trained at 65536.
