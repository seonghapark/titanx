# Post-Training AGPT-2B @ 32768 — 2026-10-16

Continued pretraining of `agpt-2b-v2-256n-step-92859-safetensors` on PG19,
400 steps at sequence length 32768, AdamW, 4 Aurora nodes / 48 ranks.

**Result**: ✅ success — job `8936184`, 400/400 steps, final loss **2.71722**.
**Output**: `/lus/flare/projects/datascience/seonghapark/agpt-2b-v2-256n-post-32768` (7.5 GB)

| | |
|---|---|
| **TP** | **4** |
| **FSDP (`dp_shard`)** | **12** |
| CP | 1 |
| Nodes × XPUs | 4 × 12 = 48 ranks |
| Step-1 loss | **3.05390** (random init would be 12.45) |
| Params loaded | **111/111** |
| MFU / peak mem | ~18.2% / 40.22 GiB (62.86%) |
| Train wall time | 22.7 min (~2.8 s/step) |
| Resource telemetry | ✅ **364 samples / 4 hosts / 15 s** |

## Read this first

**[REPORT.md](REPORT.md)** — hyperparameters, parallelism evidence, loss trajectory,
resource consumption, export verification, and how this run differs from the 16384 one.

## What changed vs. [16384](../post_training_16384_Oct16_2B_16384/)

1. **TP went 2 → 4, FSDP 24 → 12.** Doubling the sequence doubles activation memory;
   halving the FSDP shard count is paid for by the wider tensor-parallel split. Net
   effect on peak device memory was almost nil: 39.51 GiB → 40.22 GiB.
   TP cannot go past 4 for this model — it has only 4 KV heads.

2. **15-second resource monitoring now works.** The 16384 run collected nothing because
   `RESOURCE_MONITOR=1` as an *environment variable* is clobbered by
   `run_train_agpt.sh:42-44`. Only the `--resource-monitor` / `--resource-interval` /
   `--resource-output-dir` **CLI flags** survive. Both are now passed. 364 samples landed.

3. **One parameterized job script.** `artifacts/job_agpt_post.sh` replaces the per-run
   copies; size and sequence length come from `qsub -v`. This also sidesteps the
   PBS spool-at-submit trap — a queued job's behavior is fixed by its `-v` string, so
   editing the script cannot silently change an already-queued run.

## Open items

- `xpu-smi` reported no per-device rows, so there is **no XPU utilization/power telemetry**.
  The trainer's own `memory:` field (40.22 GiB, 62.86%) is the only device-side number.
- No inference/perplexity eval on the exported model — tensor-level checks only
  (111/111 keys, 0 NaN/Inf, 24/24 sampled tensors moved; see REPORT.md §6).
- CP=1 here. 65536 and beyond need CP≥2; at 62.86% occupancy, 32768 is near the point
  where TP alone stops being enough.

## Layout

```
REPORT.md                                   full report
artifacts/loss_trajectory_400steps.txt      all 400 step lines
artifacts/run_8936184.dedup.log             trainer log, deduped across 48 ranks
artifacts/pbs_stdout_8936184.log.gz         PBS stdout, raw
artifacts/export.log                        dcp_to_hf conversion
artifacts/resource_metrics/                 15 s samples: 4 × {csv, jsonl, summary.json}
artifacts/job_agpt_post.sh                  parameterized job script
artifacts/verify_export_bf16.py             torch-free safetensors differ (BF16 by hand)
artifacts/verify_export.txt                 its output: key sets, NaN sweep, deltas
```
