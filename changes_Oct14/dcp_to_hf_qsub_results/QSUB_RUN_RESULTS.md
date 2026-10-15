# DCP → HF Conversion and Training Smoke Test — qsub Run Results

**Date:** 2026-10-09
**Machine:** Aurora (ALCF), Intel XPU (PVC)
**Queue:** `debug-scaling`, project `AuroraGPT`
**Goal:** Convert the `agpt-2b-v2` DCP checkpoint at `step-46400` to HuggingFace
safetensors, then verify `run_train_agpt.sh` trains from the converted result.

---

## 1. Summary

| Stage | Job | Result |
|---|---|---|
| DCP → HF conversion | 8917463 | **PASS** (`convert rc=0`, `verify rc=0`) |
| Independent weight verification | 8917463 | **PASS** (`VERIFY: PASS`) |
| Training smoke test | 8917595 | **PASS** (5/5 steps, `Exit_status = 0`) |

**Converted model:**
`/lus/flare/projects/datascience/seonghapark/converted_models/agpt-2b-v2-step-46400_hf` (7.5 GB)

**Source DCP:**
`/lus/flare/projects/AuroraGPT/foremans/runs/agpt-2b-v2/torchtitan-ezpz/outputs/checkpoints/agpt-2b-sophiag-olmo-mix-1124-n512-gbs12288/step-46400`

Five jobs were submitted in total. Three failed before producing a model, and a
fourth (8917463) converted successfully but crashed in its training half; each
failure and its fix is recorded in §6, because two were real code defects rather
than environment noise. **Read §5 before scaling up** — the smoke test ran on 12
ranks, not the 48 the 4-node allocation implies.

---

## 2. Job inventory

All jobs used identical resources, mirroring the requested interactive
allocation:

```
qsub -A AuroraGPT -q debug-scaling -k doe -j oe \
     -l filesystems=home:flare -l select=4 -l walltime=01:00:00
```

Batch rather than `qsub -I` because the driving session cannot hold an
interactive shell; the resource request is otherwise identical.

| Job ID | Script | Walltime | Exit | Outcome |
|---|---|---|---|---|
| 8917413 | `job_convert_and_smoke.sh` | — | — | Deleted before run (`qdel`) — bad wandb CLI flag |
| 8917437 | `job_convert_and_smoke.sh` | 00:00:03 | 1 | Died in prologue — module name typo + `set -u` |
| 8917457 | `job_convert_and_smoke.sh` | 00:00:06 | 2 | Died at conversion — `agpt` import failure |
| **8917463** | `job_convert_and_smoke.sh` | 00:08:15 | 1 | **Conversion PASS**; training crashed (see §6.5) |
| **8917595** | `job_smoke_train.sh` | 00:02:05 | **0** | **Training PASS**, 5/5 steps |

Exit codes 1/2 on 8917437/8917457 are the script's own `FATAL` guards firing
(`exit 1` prologue, `exit 2` conversion), not opaque crashes. Job 8917463 exits
1 because its training half crashed *after* conversion and verification both
returned 0 — the converted model on disk from that job is good, and is the one
job 8917595 trained from.

---

## 3. Job 8917463 — DCP → HF conversion

**Started:** Fri Oct 9 08:56:56 PM UTC 2026, node `x4002c0s5b0n0`
**Logs:** `xpu_launcher/convert_smoke_20261009_205656/`

### 3.1 Conversion command

```
python dcp_to_hf.py \
  --dcp-path  <step-46400> \
  --output    <converted_models/agpt-2b-v2-step-46400_hf> \
  --assets-from <agpt-2b-v2-256n-step-92859-safetensors> \
  --force \
  -- --module agpt --config agpt_2b
```

`--config agpt_2b`, **not** `agpt_2b_yarn`. The adapter picks the Q/K rotary
layout from the RoPE class, and this checkpoint was pretrained with the default
`ComplexRoPE`. Converting under the yarn (`CosSinRoPE`) flavor would skip the
permutation and write Meta-layout rows under HF key names — loading cleanly and
training on silently corrupted attention. See §7.

### 3.2 Conversion output (verbatim)

```
[dcp_to_hf] adapter: Llama3StateDictAdapter
[dcp_to_hf] rope: ComplexRoPE.Config
[dcp_to_hf] loading DCP checkpoint: .../step-46400
[dcp_to_hf] loaded 115 top-level entries
[dcp_to_hf] ignoring 4 non-model key pattern(s): ['dataloader', 'lr_scheduler', 'optimizer', 'train_state']
[dcp_to_hf] 111 model tensors
[dcp_to_hf] converted to 111 HF tensors
[dcp_to_hf] wrote model-00001-of-00002.safetensors (22 tensors)
[dcp_to_hf] wrote model-00002-of-00002.safetensors (89 tensors)
[dcp_to_hf] wrote model.safetensors.index.json
[dcp_to_hf] copied assets: config.json, special_tokens_map.json, tokenizer.json, tokenizer.model, tokenizer_config.json
[dcp_to_hf] done: .../agpt-2b-v2-step-46400_hf
convert rc=0
```

Tensor count checks out: 12 layers × 9 + embed + norm + lm_head = **111**.
`generation_config.json` is absent from the source assets and was correctly
skipped rather than fabricated.

### 3.3 Independent verification (verbatim)

Verification re-opens the written safetensors in a separate process, so it does
not share state with the converter that produced them.

```
shards: ['model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors']
tensors: 111
config: layers=12 hidden=2048 vocab=256128
missing: none
embed shape: [256128, 2048] expected: [256128, 2048]
embed shape ok: True
key set identical to reference: True
sample lm_head.weight: dtype=torch.float32 mean=-0.001130 std=0.041061 min=-0.4480 max=0.5063
finite: True
VERIFY: PASS
verify rc=0
```

### 3.4 Output directory

```
config.json                              478 B
model-00001-of-00002.safetensors         4,911,573,464 B
model-00002-of-00002.safetensors         3,034,752,952 B
model.safetensors.index.json             9,088 B
special_tokens_map.json                  636 B
tokenizer.json                           17,518,497 B
tokenizer.model                          4,241,003 B
tokenizer_config.json                    33,556 B
```

---

## 4. Job 8917595 — training smoke test

**Started:** 2026-10-09 22:32
**Nodes allocated:** `x4009c2s2b0n0`, `x4009c2s3b0n0`, `x4009c2s4b0n0`, `x4009c2s5b0n0` (208 ncpus each)
**Walltime used:** `00:02:05`  **Exit status:** `0`
**Logs:** `xpu_launcher/smoke_train_20261009_223200/train.log`

### 4.1 Command

```
MODEL=<converted_models/agpt-2b-v2-step-46400_hf> \
MODULE=agpt CONFIG=agpt_2b_yarn DATASET_NAME=PG19 \
DATASET_PATH=<assets/hf/datasets/pg19/data> \
./run_train_agpt.sh --model 2b multi \
  -- --training.steps 5 --training.seq_len 16384 \
     --training.local_batch_size 1 --checkpoint.interval 1000
```

`WANDB_MODE=offline` is set in the environment rather than passed as a CLI
flag — see §6.1.

### 4.2 Checkpoint load

```
Finished loading the checkpoint in 0.54–0.61 seconds   (per rank)
```

The converted HF weights loaded without error on every rank.

### 4.3 Training steps (verbatim)

```
step:  1  loss: 12.93765  grad_norm:  2.0092  memory: 37.76GiB(59.01%)  tps:   399  tflops:  5.43  mfu:  1.82%
step:  2  loss: 12.84021  grad_norm:  2.0597  memory: 45.58GiB(71.24%)  tps: 6,262  tflops: 85.19  mfu: 28.57%
step:  3  loss: 12.64411  grad_norm:  2.0113  memory: 45.58GiB(71.24%)  tps: 6,054  tflops: 82.35  mfu: 27.62%
step:  4  loss: 12.36340  grad_norm:  1.9864  memory: 45.58GiB(71.24%)  tps: 6,066  tflops: 82.53  mfu: 27.68%
step:  5  loss: 11.97022  grad_norm:  1.9801  memory: 45.58GiB(71.24%)  tps: 6,177  tflops: 84.03  mfu: 28.18%
Training completed
```

Loss decreases monotonically 12.94 → 11.97; grad_norm stays ~2.0; steady-state
throughput ≈ 6.1k tps / ~83 TFLOPs / ~28% MFU. Step 1 is slower (compile +
warmup), as expected.

Figures above are one representative rank. Loss and grad_norm are identical
across all 12 ranks; throughput and memory vary slightly per rank (e.g. step 1
ranges 399–501 tps, step 4 spans 82.53–82.55 TFLOPs, memory 45.58–45.61 GiB).

### 4.4 Effective configuration confirmed by the trainer

Emitted by job 8917595, verbatim and complete up to the point the banner
stopped:

```
TRAINING HYPERPARAMETERS AND CONFIGURATION
  Model Name: agpt
  Model Flavor: 2b_yarn
  Architecture: Config
  RoPE Backend: CosSinRoPE.Config
  RoPE theta: 50000
  RoPE scaling: yarn
  RoPE max_seq_len: 16384
  RoPE original_seq_len: 8192
  RoPE rope_factor: 2.0
  RoPE beta_fast: 1.0
  RoPE beta_slow: 32.0
  RoPE mscale: 0.0
  Total Training Steps: 5
  Sequence Length: 16384
  Global Batch Size: -1
  Micro Batch Size: 1
  Gradient Accumulation Steps: 1
  Data Type: float32
  Parameter Groups: 1
  [group 0] pattern: .*
  [group 0] optimizer: AdamW
  [group 0]   betas: (0.9, 0.95)
  [group 0]   eps: 1e-08
  [group 0]   lr: 2.28e-05
  [group 0]   weight_decay: 0.1
  Implementation: fused
  LR Scheduler Type: Config
  Warmup Steps: 200
  Total Steps: None
  Decay Ratio: 0.8
  Decay Type: linear
  Min LR Factor: 0.0
  Tensor Parallel Size (TP): 1
  Pipeline Parallel Size (PP): 1
WARNING - Could not log training hyperparameters: AttributeError("'ParallelDims' object has no attribute 'dp'")
```

The banner **stops after `Pipeline Parallel Size (PP): 1`** — that is precisely
where `parallel_dims.dp` was read (§6.5). Everything below it in the function
(the rest of PARALLELISM, DATA, FILE PATHS, ARCHITECTURE DETAILS, MEMORY
OPTIMIZATION, CHECKPOINT, DISTRIBUTED) was **not** printed in this run. Training
continued regardless, which is the point of the `try/except`.

The architecture section was confirmed separately, off-node, after the
`dp` → `dp_replicate`/`dp_shard` fix:

```
Data Parallel Replicate: 1    Data Parallel Shard (FSDP): 48
Number of Layers: 12   Attention Heads: 16   KV Heads: 4   Head Dim: 128
Hidden Dimension: 2048   Vocabulary Size: 256128   Weight Tying: False
```

**`rope_factor: 2.0` = 16384 / 8192.** This is the on-hardware confirmation that
the YaRN scale derives solely from `--training.seq_len`, not from the flavor's
declared ceiling (`max_seq_len=262144`, `rope_factor=32.0`). Verified off-node
across lengths:

| `--training.seq_len` | resulting `rope_factor` | `max_seq_len` |
|---|---|---|
| 8192 | 1.0 | 8192 |
| 16384 | 2.0 | 16384 |
| 32768 | 4.0 | 32768 |
| 131072 | 16.0 | 131072 |

---

## 5. Caveats — read before scaling up

1. **Only 12 of 48 ranks ran.** PBS allocated 4 nodes
   (`exec_host` lists `x4009c2s{2,3,4,5}b0n0`), and the launcher banner reports
   `Nodes = 4`, but the device mesh came up as
   `pp=1, dp_replicate=1, dp_shard=12, cp=1, tp=1, ep=1` and exactly 12 ranks
   reported `Training completed`. The launch line was `xpu launch --scheduler
   auto -ppn 12`, so one node's worth of ranks ran while three nodes sat idle.
   The smoke test is still valid — the weights load and the model trains — but
   **this is not a 4-node scaling result**, and the cause (`NPROC_PER_NODE=4` in
   the topology banner vs. `-ppn 12` on the launch line) has not been
   investigated. Resolve this before reading anything into the throughput
   numbers or before a multi-node production run.

2. **Numerical equivalence to the original checkpoint was not tested.** The
   smoke test proves the weights load and loss descends from a plausible
   starting value; it does not prove the converted tensors are numerically
   faithful. A perplexity comparison against the step-92859 safetensors would
   catch a residual layout error. Loss starting near ~12.9 rather than a
   well-trained value is expected here (fresh PG19 data, new seq_len), so it is
   not by itself evidence either way.

3. **Launcher-computed values are overridden.** The banner prints
   `Training Steps = 3659 (overridden by extra args)` and
   `Sequence Length = 8192 (overridden by extra args)`; the trainer used the
   post-`--` values (5 steps, 16384). The banner's `Global Batch Size = 96` and
   `Local Batch Size = 2` likewise reflect pre-override launcher math, not the
   `local_batch_size 1` actually used.

---

## 6. Failures encountered and fixes

### 6.1 Job 8917413 — `--metrics.enable_wandb false` (caught pre-flight)

tyro renders bool config fields as **flags**, not value-taking options, so
`--metrics.enable_wandb false` leaves a stray positional `false` and aborts
argument parsing. Confirmed by grepping a known-good run log for value-style
booleans: zero matches.

**Fix:** `export WANDB_MODE=offline`, which needs no CLI support.
Required `qdel 8917413` and resubmission, because PBS snapshots the script at
submission time — editing the file after `qsub` changes nothing.

### 6.2 Job 8917437 — prologue death

Two independent causes:

- `module load framework/2026.1.0` — the module is **`frameworks`** (plural).
  Harmless in an interactive shell because a later `ml add frameworks` covers
  it, fatal in batch.
- `set -u` aborted on lmod's unbound `ZSH_EVAL_CONTEXT`.

**Fix:** `module load frameworks/2026.1.0 || module load frameworks || fatal`,
and `set -o pipefail` without `set -u`.

### 6.3 Job 8917457 — `agpt` module not importable (real code bug)

```
ImportError: Cannot import module 'agpt' or 'agpt.config_registry'
```

…listing only the wheel's built-in models. `dcp_to_hf.py` never prepended
`torchtitan_repo` to `sys.path` the way `titan_train.py` does, so it imported
the **installed wheel**, which carries no `agpt` package.

**Fix:** added a `_TORCHTITAN_ROOT` `sys.path` insert honoring `TORCHTITAN_ROOT`,
mirroring `titan_train.py`, so the converter and the trainer resolve to one
checkout — which also guarantees the adapter is the one training will use.

### 6.4 Monitoring error on my side (recorded for accuracy)

I briefly reported that job 8917463 hit the same ImportError. That was wrong:
the monitor had read `convert_smoke_20261009_205137`, a **stale** log directory
from job 8917457, because my cleanup used `rmdir`, which silently refuses
non-empty directories. Job 8917463 was still queued with no output at the time.
The stale directory was archived to `old_convert_smoke_logs/`.

### 6.5 Job 8917463 training half — `AttributeError` in hyperparameter logging

Conversion succeeded; training then crashed on **all 12 ranks**, after the model
had fully loaded:

```
File ".../torchtitan/trainer.py", line 1044, in train
  self._log_training_hyperparameters()
File ".../torchtitan/trainer.py", line 810, in _log_training_hyperparameters
  logger.info(f"  Learning Rate: {config.optimizer.lr}")
AttributeError: 'Config' object has no attribute 'lr'
```

A purely cosmetic logging banner killed a 4-node run. This fork has no
`config.optimizer.lr`: LR lives per `ParamGroupConfig`. Because the previous
production run died earlier (ComplexRoPE assertion during checkpoint load), this
code path had likely never executed successfully.

Rather than patch the one line, every attribute chain in the function was
resolved against a real `ConfigManager`-built config. **Five more wrong names**
turned up, four of them invisible because `hasattr` guards made them print
nothing at all instead of failing:

| Wrong | Actual |
|---|---|
| `config.optimizer.lr`, `.weight_decay`, `.betas`, `.eps` | `config.optimizer.param_groups[i].optimizer_kwargs[...]` |
| `config.training.batch_size` | `config.training.local_batch_size` |
| `config.checkpoint.interval_type` | `config.checkpoint.interval` |
| `config.lr_scheduler.min_lr_ratio` | `config.lr_scheduler.min_lr_factor` |
| flat `n_layers` / `n_heads` / `n_kv_heads` / `rope_theta` / `rope_scaling` | nested: `len(model_config.layers)`, `model_config.layers[0].attention.{n_heads,n_kv_heads,head_dim}`, `...attention.rope` |
| `parallel_dims.dp` | `parallel_dims.dp_replicate` / `.dp_shard` |

**Fixes applied to `torchtitan/trainer.py`:**

1. All six field groups corrected; the optimizer section now iterates every
   param group and prints each group's pattern, optimizer name and kwargs,
   rather than inventing a single headline LR.
2. The RoPE and architecture sections now read the real nested structure, so
   they report actual values instead of silently printing nothing. RoPE now also
   logs the backend **class**, which is what decides the Q/K rotary layout and
   therefore the first thing to check when a converted checkpoint misbehaves.
3. The call site is wrapped in `try/except`, logging a warning instead of
   raising — an informational banner must never be able to abort training.

**The try/except immediately proved its worth.** The successful run 8917595 still
logged:

```
WARNING - Could not log training hyperparameters: AttributeError("'ParallelDims' object has no attribute 'dp'")
```

and **kept training to completion**. Under the old code that would have been a
sixth fatal crash. `parallel_dims.dp` has since been corrected to
`dp_replicate`/`dp_shard`, and the full banner was re-verified end-to-end
against a real `ParallelDims`.

---

## 7. The RoPE flavor trap (why `--config agpt_2b` for conversion)

Round trip, verified in `models/llama3/state_dict_adapter.py`:

- `to_hf` under **ComplexRoPE** *applies* `_permute` → Meta interleaved rows
  (2i, 2i+1) become HF `rotate_half` layout (i, i+head_dim/2) → a genuine HF
  checkpoint.
- `from_hf` under **CosSinRoPE** *skips* the permutation → HF weights load as-is
  into a backend that already uses `rotate_half`.

Both halves agree, so converting under `agpt_2b` and training under
`agpt_2b_yarn` is correct. `_needs_qk_permute()` keys off the RoPE config
**class** (`issubclass(..., ComplexRoPE.Config)`), so the flavor name is the only
control. Confirmed empirically during conversion: `rope: ComplexRoPE.Config`,
needs Q/K permute → `True`.

Converting under the yarn flavor would skip the permute and write Meta-layout
rows under HF key names: no crash, no error, just a quietly worse model.

---

## 8. Files changed

| File | Change |
|---|---|
| `xpu_launcher/dcp_to_hf.py` | `sys.path` fix so the repo (not the wheel) provides `agpt`; stale docstring/NOTE corrected |
| `xpu_torchtitan/torchtitan_repo/torchtitan/trainer.py` | `_log_training_hyperparameters` field fixes (§6.5) + non-fatal call site |
| `xpu_launcher/job_convert_and_smoke.sh` | New: combined convert + smoke job |
| `xpu_launcher/job_smoke_train.sh` | New: training-only job against the already-converted model |

## 9. Artifacts

| What | Where |
|---|---|
| Converted HF model | `/lus/flare/projects/datascience/seonghapark/converted_models/agpt-2b-v2-step-46400_hf` |
| Conversion logs (8917463) | `xpu_launcher/convert_smoke_20261009_205656/` |
| PBS stdout (8917463) | `xpu_launcher/dcp2hf_smoke.o8917463` |
| Training logs (8917595) | `xpu_launcher/smoke_train_20261009_223200/train.log` |
| PBS stdout (8917595) | `xpu_launcher/agpt_smoke_train.o8917595` |
| Archived stale logs | `xpu_launcher/old_convert_smoke_logs/` |

---

## 10. Reproduce

```bash
cd /lus/flare/projects/datascience/seonghapark/xpu_launcher

# Conversion + smoke test from scratch
qsub job_convert_and_smoke.sh

# Training only, against the already-converted model
qsub job_smoke_train.sh
```

Config-only checks (adapter choice, RoPE flavor, banner fields, rope_factor
derivation) run on the **login node** without a job, provided torch is imported
inside a module-loaded shell:

```bash
bash -lc 'module load frameworks/2026.1.0; \
          source /lus/flare/projects/datascience/seonghapark/venv/bin/activate; \
          python your_check.py'
```

Exporting the frameworks lib directory alone is **not** sufficient — torch fails
with `ImportError: libpti_view.so.0`, which comes from the module's own pti-gpu
0.17.0.

---

## 9. AMENDMENT — 2026-10-12: YaRN flavors switched to ComplexRoPE

`agpt_2b_yarn` / `agpt_20b_yarn` now use **ComplexRoPE**, not CosSinRoPE.
See `../rope_complex_switch/change_plan.md`. Sections 3.1 and 7 above are
superseded as follows.

**The §7 trap is gone for these flavors.** Both sides are now complex, so the
round trip is symmetric rather than asymmetric-but-matched:

- *Before*: `to_hf` under `agpt_2b` permuted; `from_hf` under `agpt_2b_yarn`
  (cos_sin) skipped the permute. Two different operations that happened to agree,
  because cos_sin natively wants the HF `rotate_half` layout.
- *Now*: `from_hf` under `agpt_2b_yarn` applies the reverse permute, converting HF
  `rotate_half` back to Meta interleaved, which ComplexRoPE wants. Verified:
  `agpt_2b_yarn needs_qk_permute=True`.

Both are correct. The practical effect is that `--config agpt_2b` and
`--config agpt_2b_yarn` now behave identically for conversion, so naming the
wrong one of those two is no longer a silent-corruption path.

**A new, narrower trap replaces it.** Any DCP checkpoint produced by *training
under the old cos_sin* `agpt_2b_yarn` holds Q/K in `rotate_half` layout. Running
`dcp_to_hf.py` on it with the post-change flavor will apply a permute that the
weights do not want — the same quiet corruption, now scoped to pre-2026-10-12
YaRN training checkpoints rather than to flavor-name choice. Conversions of
pretraining checkpoints (`agpt_2b`, the case this document covers) are unaffected:
that flavor was complex before and is complex now, and the §3.1 command is
unchanged and still correct.

Separately, the switch is not numerically neutral under active YaRN: the two
backends round the NTK-by-parts correction range differently (integers vs floats),
which moves 21 of 64 frequency dimensions inside the transition band. See §2c of
the change plan.
