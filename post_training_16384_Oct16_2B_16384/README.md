# Post-Training AGPT-2B @ 16384 — 2026-10-16

Continued pretraining of `agpt-2b-v2-256n-step-92859-safetensors` on PG19,
400 steps at sequence length 16384, AdamW, 4 Aurora nodes / 48 ranks.

**Result**: ✅ success — job `8935820`, 400/400 steps, final loss **2.66022**.
**Output**: `/lus/flare/projects/datascience/seonghapark/agpt-2b-v2-256n-post-16384` (7.5 GB)

| | |
|---|---|
| **TP** | **2** |
| **FSDP (`dp_shard`)** | **24** |
| Nodes × XPUs | 4 × 12 = 48 ranks |
| Step-1 loss | **2.68685** (random init would be 12.45) |
| Params loaded | **111/111** |
| MFU / peak mem | ~20.5% / 39.51 GiB (61.75%) |
| Wall time | 30.5 min |

## Read this first

**[REPORT.md](REPORT.md)** — full report: hyperparameters, parallelism evidence,
loss trajectory, export verification, the three checkpoint-load bugs, and the one
unmet requirement.

## Two things worth knowing

1. **A checkpoint load can log success and load zero weights.** Three prior allocations
   were lost to this; the first trained a *randomly-initialized* model for 250 steps while
   reporting a successful load. The only reliable detector is **step-1 loss vs
   `ln(vocab_size)`** (12.45 for this model). See REPORT.md §1 and §6.

2. **`RESOURCE_MONITOR=1` as an environment variable does nothing.** `run_train_agpt.sh`
   clobbers it at startup; only `--resource-monitor` / `--resource-interval` /
   `--resource-output-dir` work. This cost the 15-second telemetry for this run.
   Fix applied to the job script. See REPORT.md §7.

## Open items

- Resource telemetry still uncollected (fix applied, needs an allocation)
- TP=2 and local_batch_size=1 were Claude's choices, not user-approved
- No inference/perplexity eval on the exported model — tensor-level checks only

## Layout

```
REPORT.md                                   full report
artifacts/loss_trajectory_400steps.txt      all 400 step lines
artifacts/export.log                        dcp_to_hf conversion
artifacts/pbs_stdout_8935820.dedup.log      PBS stdout, deduped
artifacts/pbs_stdout_8935820.log.gz         PBS stdout, raw
artifacts/job_agpt2b_post16384.sh           job script (with the resource fix)
artifacts/test_strided.py                   CPU repro of the _StridedShard bug
artifacts/test_loadsd.py                    CPU rehearsal of the load path
```
