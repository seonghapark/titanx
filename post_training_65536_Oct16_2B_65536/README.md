# Post-Training AGPT-2B @ 65536 — 2026-10-16

Continued pretraining of `agpt-2b-v2-256n-step-92859-safetensors` on PG19,
400 steps at sequence length 65536, AdamW, 4 Aurora nodes / 48 ranks.

**Result**: ✅ success — job `8936339`, 400/400 steps, final loss **2.34374**.
**Output**: `/lus/flare/projects/datascience/seonghapark/agpt-2b-v2-256n-post-65536` (7.5 GB)

| | |
|---|---|
| **TP** | **4** |
| **FSDP (`dp_shard`)** | **6** |
| **CP** | **2** ← first cell needing context parallelism |
| Nodes × XPUs | 4 × 12 = 48 ranks |
| Global batch size | 6 (= 48 ÷ TP·CP) |
| Step-1 loss | **2.82747** (random init would be 12.45) |
| Params loaded | **111/111** |
| MFU / peak mem | **20.48%** / 41.14 GiB (64.30%) |
| Train wall time | ~29 min |
| Resource telemetry | ✅ 464 samples / 4 hosts / 15 s |

## Read this first

**[REPORT.md](REPORT.md)** — hyperparameters, parallelism evidence, loss trajectory,
resource consumption, export verification, and why the final-step loss should not be
read as a 0.5-nat improvement.

## What changed vs. [32768](../post_training_32768_Oct16_2B_32768/)

1. **CP=2 entered the picture, and it was cheap.** TP is capped at 4 by the model's
   4 KV heads, so doubling the sequence again had to come from context parallelism.
   Peak memory went 40.22 → 41.14 GiB — **1.4 points of occupancy for 2× the context**.

2. **MFU went up, not down**: 18.2% → **20.5%**, TFLOPs 54.3 → 61.1. Longer sequences
   amortize the collectives better. Wall time rose 22.7 → 29 min only because each step
   does twice the work.

3. **Global batch size halved, 12 → 6.** `gbs = world_size ÷ (TP × CP)`, so every CP
   doubling costs a factor of two in batch. Tokens/step is unchanged at 393,216, but the
   gradient is averaged over half as many independent sequences — which is why the loss
   curve here is visibly noisier than the 32768 one.

## ⚠️ Do not read the final loss as a result

Final-step loss is **2.34374**, against 2.83402 at step 350 and ~2.8 for the preceding
150 steps. That is a 0.5-nat drop in the last 50 steps with grad_norm *rising* 1.27 → 1.92.
This is almost certainly a lucky batch at gbs=6, not convergence. The honest summary of
this run is "loss settled around 2.8"; see REPORT.md §4.

Cross-cell loss comparison is not meaningful either — the cells differ in global batch
size, RoPE scaling factor, and effective context, so they are not the same objective.

## Open items

- `xpu-smi` reported no per-device rows → **no XPU utilization/power telemetry**.
- No inference/perplexity eval on the exported model — tensor-level checks only.
- Warmup is 200 of 400 steps; this is a feasibility sweep, not a convergence study.

## Layout

```
REPORT.md                                   full report
artifacts/loss_trajectory_400steps.txt      all 400 step lines
artifacts/run_8936339.dedup.log             trainer log, deduped across 48 ranks
artifacts/pbs_stdout_8936339.log.gz         PBS stdout, raw
artifacts/export.log                        dcp_to_hf conversion
artifacts/resource_metrics/                 15 s samples: 4 × {csv, jsonl, summary.json}
artifacts/job_agpt_post.sh                  parameterized job script
artifacts/verify_export_bf16.py             torch-free safetensors differ
artifacts/verify_export.txt                 its output
```
