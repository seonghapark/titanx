# Script Comparison: AGPT Training vs Generic TorchTitan Training

## Overview
This document compares two scripts:
1. `run_train_torchtitan.sh` - Generic TorchTitan training wrapper (flexible, framework-agnostic within TorchTitan)
2. `train_agpt_2b_venv.sh` - AGPT-specific production training script (optimized for AGPT 2B model)

---

## Architecture Comparison

### run_train_torchtitan.sh
```
User provides: MODULE, CONFIG, DATASET, etc. via environment variables
    ↓
Script builds titan_train.py command with flexible arguments
    ↓
Delegates to run_train.sh (XPU launcher)
    ↓
xpu launch (distributed training)
```

### train_agpt_2b_venv.sh
```
PBS job scheduler context (PBS directives hardcoded)
    ↓
Loads modules + environment setup (OneAPI, Intel frameworks)
    ↓
Broadcasts .venv to compute nodes (yeet-env)
    ↓
Calculates AGPT-specific training parameters (GBS, TRAINING_STEPS)
    ↓
Calls: ezpz launch python3 -m torchtitan.experiments.ezpz.train
    ↓
Direct to training (no intermediate run_train.sh)
```

---

## Detailed Comparison Table

| Aspect | run_train_torchtitan.sh | train_agpt_2b_venv.sh |
|--------|------------------------|-----------------------|
| **Purpose** | Generic TorchTitan training launcher | AGPT 2B production training |
| **Framework Level** | High (flexible for any TorchTitan config) | Ultra-specific (AGPT 2B only) |
| **Scheduling** | Generic (relies on scheduler auto-detect) | PBS job (hardcoded PBS directives) |
| **User Inputs** | Command-line or env vars (flexible) | Computed from PBS/node context |
| **Python Binary** | Auto-detected or user-specified | .venv (broadcast via yeet-env) |
| **Module/Config** | User-configurable (default: llama3) | HARDCODED: agpt / agpt_2b |
| **Deployment** | Interactive or via run_train.sh | PBS-submitted job only |
| **Entry Point** | titan_train.py | torchtitan.experiments.ezpz.train |
| **venv Distribution** | Uses compute node's local python | Broadcast .venv.tar.gz to /tmp on all nodes |

---

## Line-by-Line Breakdown

### run_train_torchtitan.sh (Lines 1-60: Setup & Config Defaults)

```bash
# Line 1-10: Shebang & documentation
#!/usr/bin/env bash
set -euo pipefail

# Validates: MODEL path required, supports optional MODULE/CONFIG overrides
# Supports debugging with detailed config output
```

**Key defaults (lines 170-179):**
```bash
MODULE="${MODULE:-llama3}"           # Flexible: can be any registered module
CONFIG="${CONFIG:-llama3_debugmodel}"   # Flexible: user can override
DATASET_NAME="${DATASET_NAME:-pg19_multinews}"
TRAINING_STEPS="${TRAINING_STEPS:-100}"
```

**Variable handling (lines 156-169):**
- Validates MODEL exists
- Falls back through multiple Python binary candidates
- No framework-specific setup beyond finding Python

---

### train_agpt_2b_venv.sh (Lines 1-32: PBS Job + Environment Setup)

```bash
#!/bin/bash --login        # Line 1: Note --login flag (sources ~/.bash_profile)
#PBS -A AuroraGPT          # Line 2: Hardcoded project account
#PBS -l walltime=06:00:00  # Line 3: 6-hour walltime
#PBS -l filesystems=flare:home
#PBS -q workq              # Queue (workq for interactive, prod for production)
#PBS -j oe                 # Join stderr/stdout

# Line 9-14: Environment module loading
module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
```

**Key differences:**
- Loads Intel OneAPI frameworks (not auto-detected)
- Enables distributed Intel GPU support (ZE_FLAT_DEVICE_HIERARCHY, CCL, etc.)
- Hardcoded for Aurora Aurora GPU cluster

---

## Environment Setup Comparison

### run_train_torchtitan.sh

**Minimal setup:**
```bash
# Just validates paths and finds Python
TORCHTITAN_ROOT="${TORCHTITAN_ROOT:-${SCRIPT_DIR}/torchtitan_repo}"
TRAIN_PYTHON_BIN="${TRAIN_PYTHON_BIN:-$TRAIN_PYTHON_BIN_DEFAULT}"
mkdir -p "$LOG_DIR"
```

**No module loading, no environment variable exports (except TORCHTITAN_ROOT)**

---

### train_agpt_2b_venv.sh

**Rich setup (lines 9-31):**

1. **Module loading:**
   ```bash
   module load oneapi/release/2025.3.1 hdf5 pti-gpu
   ```

2. **Distributed Intel GPU setup:**
   ```bash
   export ZE_FLAT_DEVICE_HIERARCHY=FLAT      # Single-level GPU hierarchy
   export CCL_PROCESS_LAUNCHER=pmix           # PMIx-based process launching
   export CCL_OP_SYNC=1                       # Synchronous collective ops
   export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
   ```

3. **ezpz utilities (line 16):**
   ```bash
   source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job
   # Provides: NNODES, NGPUS, NHOSTS, WORLD_SIZE, etc.
   # Provides: log_message, ezpz, yeet-env, etc.
   ```

4. **Virtual environment distribution (lines 19-31):**
   ```bash
   source .venv/bin/activate
   if [[ -f .venv.tar.gz ]]; then
       ezpz yeet-env --src .venv.tar.gz  # Broadcast tarball to all nodes
   else
       ezpz yeet-env                      # Fallback: rsync individual files
   fi
   deactivate
   source /tmp/.venv/bin/activate         # Activate from /tmp on all nodes
   ```

**The .venv broadcast is critical at scale:** Copying 8.6GB of Python packages to every node is faster via tarball broadcast than per-file rsync.

---

## Configuration Calculation Comparison

### run_train_torchtitan.sh

**User provides or environment defaults (lines 156-179):**
```bash
TRAINING_STEPS="${TRAINING_STEPS:-100}"      # User must specify
SEQ_LEN="${SEQ_LEN:-16384}"
LOG_DIR="${LOG_DIR:-...xpu_torchtitan_$(date +...)}"
```

**No calculation** — passes through user's choices directly.

---

### train_agpt_2b_venv.sh

**Computes all training parameters from PBS context (lines 34-52):**

1. **Topology from scheduler:**
   ```bash
   NNODES="${NHOSTS:-$(wc -l < "${PBS_NODEFILE}")}"
   # Auto-reads node list from PBS job context
   ```

2. **Parallelism defaults:**
   ```bash
   TP="${TP:-1}"  # Tensor parallelism (default: 1)
   PP="${PP:-1}"  # Pipeline parallelism (default: 1)
   CP="${CP:-1}"  # Context parallelism (default: 1)
   LBS="${LBS:-2}"  # Local batch size
   GAS="${GAS:-1}"  # Gradient accumulation steps
   ```

3. **Derives global batch size from topology:**
   ```bash
   GBS=$(( NGPUS * LBS * GAS / (TP * PP * CP) ))
   # Example: 8 nodes × 12 GPUs × 2 LBS × 1 GAS / 1 = 192 GBS
   ```

4. **Calculates training steps from token target:**
   ```bash
   TRAIN_TOKENS="${TRAIN_TOKENS:-4673780159710}"  # 4.67T tokens target
   TRAINING_STEPS=$(( TRAIN_TOKENS / (GBS * SEQ_LEN) ))
   # Example: 4.67T / (192 GBS × 8192 SEQ_LEN) = 2,970,456 steps
   ```

5. **Optimizer and learning rate (hardcoded for AGPT):**
   ```bash
   OPTIMIZER="${OPTIMIZER:-sophiag}"  # AGPT uses SophiaG (not AdamW)
   LR="${LR:-2.28e-5}"  # Production learning rate for AGPT 2B
   ```

6. **Data list selection:**
   ```bash
   DFL_NAME="${DFL_NAME:-olmo-mix-1124}"  # AGPT production dataset mix
   DFL="torchtitan/experiments/ezpz/data-lists/$(ezpz_get_machine_name)/${DFL_NAME}.txt"
   ```

7. **Checkpoint strategy:**
   ```bash
   CKPT_DIR="checkpoints/agpt-${MODEL}-${OPTIMIZER}-${DFL_NAME}-n${NNODES}-gbs${GBS}"
   # Example: checkpoints/agpt-2b-sophiag-olmo-mix-1124-n8-gbs192/
   ```

---

## Command Building Comparison

### run_train_torchtitan.sh (Lines 191-240)

**Builds generic TorchTitan command:**
```bash
TRAIN_CMD=(
  "$TRAIN_PYTHON_BIN" "${SCRIPT_DIR}/titan_train.py"
  "--module" "$MODULE"
  "--config" "$CONFIG"
  "--hf_assets_path" "$HF_ASSETS_PATH"
  "--dump_folder" "$LOG_DIR"
  "--dataloader.dataset" "$DATASET_NAME"
  "--checkpoint.enable"
  "--checkpoint.folder" "$CKPT_FOLDER"
)

# Conditionally adds hyperparameters if user didn't override
if ! has_extra_arg "--training.steps"; then
  TRAIN_CMD+=("--training.steps" "$TRAINING_STEPS")
fi
# ... similar for seq_len, loss termination, etc.

# User can override via -- <extra args>
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  TRAIN_CMD+=("${EXTRA_ARGS[@]}")
fi
```

**Entry point:** `titan_train.py` (lightweight wrapper around torchtitan.train.main)

---

### train_agpt_2b_venv.sh (Lines 78-97)

**Builds AGPT-specific ezpz training command:**
```bash
ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module=agpt \
    --config="agpt_${MODEL}" \
    --checkpoint.enable \
    --checkpoint.folder="${CKPT_DIR}" \
    --checkpoint.interval="${CKPT_INTERVAL}" \
    --checkpoint.keep-latest-k="${CKPT_KEEP_LATEST_K}" \
    --checkpoint.no-last-save-model-only \
    --checkpoint.async-mode="${CHECKPOINT_ASYNC_MODE:-async}" \
    --dataloader.dataset=blendcorpus \
    --dataloader.dataset-path="${DFL}" \
    --dataloader.data-cache-path="${DATA_CACHE_PATH}" \
    --debug.print-config \
    --optimizer="${OPTIMIZER}" \
    --optimizer.lr="${LR}" \
    --training.local-batch-size="${LBS}" \
    --training.global-batch-size="${GBS}" \
    --training.seq-len="${SEQ_LEN}" \
    --training.steps="${TRAINING_STEPS}" \
    "$@"
```

**Key differences:**
1. Uses `ezpz launch` directly (NOT `run_train.sh`)
2. Module is hardcoded: `--module=agpt`
3. Config is derived: `--config="agpt_${MODEL}"` (agpt_2b)
4. Dataset is hardcoded: `--dataloader.dataset=blendcorpus` (not generic)
5. All parameters (GBS, TRAINING_STEPS, LR) are computed, not user defaults
6. `--debug.print-config` enabled for logging

---

## Output & Monitoring Comparison

### run_train_torchtitan.sh (Lines 249-297)

**Prints configuration summary (to stderr):**
```bash
cat >&2 <<EOF
================================================================================
TRAINING LAUNCH CONFIGURATION
================================================================================

[LAUNCH MODE]
  Mode                   = ${MODE}...
  Topology               = NNODES=${NNODES:-auto} NPROC_PER_NODE=${NPROC_PER_NODE:-4}...

[MODEL & DATASET]
  Model Path             = ${MODEL}
  Module/Config          = ${MODULE} / ${CONFIG}
  Dataset Name           = ${DATASET_NAME}...

[TRAINING HYPERPARAMETERS]
  Training Steps         = ${TRAINING_STEPS}${TRAINING_STEPS_NOTE}
  Sequence Length        = ${SEQ_LEN}...

[SYSTEM & PATHS]
  TorchTitan Root        = ${TORCHTITAN_ROOT}
  Log Directory          = ${LOG_DIR}...

[MONITORING]
  Resource Monitor       = ${RESOURCE_MONITOR}
  Resource Interval      = ${RESOURCE_INTERVAL}s...

[EXTRA ARGUMENTS]
  Additional Args        = ${EXTRA_ARGS[*]:-<none>}
...
EOF
```

**Detailed, readable, organized by category**

---

### train_agpt_2b_venv.sh (Lines 60-75)

**Prints configuration summary (via log_message):**
```bash
log_message INFO "==========================================="
log_message INFO "Training ${MODEL} on ${TRAIN_TOKENS} tokens"
log_message INFO "-------------------------------------------"
log_message INFO "TRAINING_STEPS: ${TRAINING_STEPS}"
log_message INFO "PBS_JOBID: ${PBS_JOBID}"
log_message INFO "NNODES: ${NNODES}"
log_message INFO "WORLD_SIZE: ${WORLD_SIZE:-${NGPUS}}"
log_message INFO "OPTIMIZER: ${OPTIMIZER}"
log_message INFO "LR: ${LR}"
log_message INFO "Local batch size (LBS): ${LBS}"
log_message INFO "Gradient accumulation steps (GAS): ${GAS}"
log_message INFO "Global batch size (GBS): ${GBS}"
log_message INFO "Training steps calculated as: ${TRAINING_STEPS}"
log_message INFO "DATASET_PATH: ${DFL}"
log_message INFO "Checkpoint directory: ${CKPT_DIR}"
log_message INFO "==========================================="
```

**Simpler, AGPT-focused, uses ezpz's log_message function (structured logging)**

---

## Execution Path Comparison

### run_train_torchtitan.sh

```bash
exec "${CMD[@]}"
# where CMD = (run_train.sh MODE [--dry-run] -- <train command>)
# → run_train.sh processes: topology, scheduler, hostfile, etc.
# → xpu launch <topology> -- python titan_train.py ...
```

**Two-hop:** run_train_torchtitan → run_train.sh → xpu launch

---

### train_agpt_2b_venv.sh

```bash
ezpz launch python3 -m torchtitan.experiments.ezpz.train ...
# Direct invocation of ezpz launch with computed parameters
```

**One-hop:** train_agpt_2b_venv → ezpz launch (no intermediate wrapper)

---

## Flexibility Comparison

### run_train_torchtitan.sh

**High flexibility:**
- Any TorchTitan MODULE (agpt, llama3, deepseek_v3, etc.)
- Any CONFIG within that module
- Any DATASET
- Customizable TRAINING_STEPS, SEQ_LEN, etc.
- Supports resource monitoring
- Supports loss-based early termination
- Accepts extra TorchTitan args via `--`

**Trade-off:** More defaults to understand, more parameters to configure

---

### train_agpt_2b_venv.sh

**Low flexibility (by design):**
- Fixed to AGPT 2B (MODULE=agpt, CONFIG=agpt_2b)
- Fixed to blendcorpus dataset
- Fixed to SophiaG optimizer
- Fixed to production learning rate (2.28e-5)
- Fixed to production dataset mix (olmo-mix-1124)
- Computes everything from node count (NNODES auto-derived from PBS)

**Trade-off:** Very little to configure, but optimized for AGPT 2B production specifically

---

## User Inputs Comparison

### run_train_torchtitan.sh

**Required:**
- `MODEL` (path to model/tokenizer assets)
- `MODE` (single or multi)

**Optional:**
- `MODULE` (default: llama3)
- `CONFIG` (default: llama3_debugmodel)
- `DATASET_NAME` (default: pg19_multinews)
- `DATASET_PATH`
- `TRAINING_STEPS` (default: 100)
- `SEQ_LEN` (default: 16384)
- `LOG_DIR`
- `CKPT_FOLDER`
- `RESOURCE_MONITOR` (0 or 1)
- Plus all `run_train.sh` env vars

**Example:**
```bash
MODEL=/path/to/agpt-2b MODULE=agpt CONFIG=agpt_2b \
TRAINING_STEPS=10000 \
./xpu_torchtitan/run_train_torchtitan.sh multi ./hosts.txt
```

---

### train_agpt_2b_venv.sh

**Required:**
- Run as PBS job (qsub)

**Optional (computed from PBS context):**
- `TP`, `PP`, `CP` (parallelism degrees)
- `LBS` (local batch size, default: 2)
- `GAS` (gradient accumulation, default: 1)
- `SEQ_LEN` (default: 8192)
- `OPTIMIZER` (default: sophiag)
- `LR` (default: 2.28e-5)
- `TRAIN_TOKENS` (default: 4.67T)
- `DFL_NAME` (dataset list, default: olmo-mix-1124)

**Example:**
```bash
# Submit job with overrides
qsub -l walltime=12:00:00 -l select=16 \
  -v TP=1,LBS=4,SEQ_LEN=16384 \
  train_agpt_2b_venv.sh
```

---

## When to Use Which

### Use run_train_torchtitan.sh

- Training any TorchTitan model (Llama3, AGPT with custom config, DeepSeek, etc.)
- Fine-tuning with different hyperparameters
- Interactive training on a compute node (via run_train_torchtitan.sh)
- Experimentation with different MODEL/MODULE/CONFIG combinations
- Academic or exploratory work

**Example:**
```bash
# Try AGPT 2B with different hyperparameters
MODULE=agpt CONFIG=agpt_2b MODEL=/path/to/agpt-2b \
  TRAINING_STEPS=50000 LBS=4 SEQ_LEN=16384 \
  ./xpu_torchtitan/run_train_torchtitan.sh multi ./hosts.txt
```

---

### Use train_agpt_2b_venv.sh

- AGPT 2B production training on Aurora
- Replicating known-good production configurations
- Submitting long-running jobs via PBS scheduler
- Scaling experiments with fixed production hyperparameters
- When you want "set it and forget it" (everything computed from node count)

**Example:**
```bash
# Submit 8-node production job
qsub -l select=8 train_agpt_2b_venv.sh

# Or with override
qsub -l select=64 -v TP=2 train_agpt_2b_venv.sh
```

---

## Key Differences Summary

| Feature | run_train_torchtitan.sh | train_agpt_2b_venv.sh |
|---------|------------------------|-----------------------|
| **Model choice** | User-configurable | Fixed to AGPT 2B |
| **Environment** | Auto-detect / fallback | Explicit module loads |
| **venv handling** | Local system Python | Broadcast .venv.tar.gz to all nodes |
| **Scheduling** | Generic (any scheduler) | PBS-specific with hardcoded directives |
| **Parameter calculation** | User defaults | Auto-derived from PBS context |
| **Dataset** | User-configurable | Fixed to blendcorpus |
| **Optimizer** | User-configurable | Fixed to SophiaG |
| **Training steps** | User specifies | Calculated from TRAIN_TOKENS target |
| **Use case** | Flexible, exploratory | Production, predictable, at-scale |
| **Overhead** | Minimal | Medium (yeet-env broadcast) |
| **Knowledge required** | Understand TorchTitan configs | Just `qsub` it (PBS handles the rest) |

---

## Production Considerations

### run_train_torchtitan.sh

**Scaling:** Tested and works, but requires manual topology specification
- All parameters must be correct
- No built-in safeguards against misconfiguration
- Good for controlled experiments with known-good parameters

### train_agpt_2b_venv.sh

**Scaling:** Optimized for Aurora production at scale
- .venv broadcast is sub-linear in scale (256N: 133s, 512N: 175s)
- Parallelism defaults tested at production scale (TP=1, LBS=2, GAS=1)
- Computes GBS automatically from NNODES × 12 GPUs/node
- Derived TRAINING_STEPS ensures token budget is met regardless of scale
- Uses SophiaG (production optimizer for AGPT, not AdamW which can NaN at large batches)

---

## Recommendations

**For AGPT 2B Training:**

1. **Quick experiments (1-8 nodes):** Use `run_train_torchtitan.sh`
   - More transparent, easier to debug
   - Can override any parameter

2. **Production training (16+ nodes):** Use `train_agpt_2b_venv.sh`
   - Pre-optimized for scale
   - .venv broadcast handles distribution
   - All safety parameters baked in (SophiaG optimizer, correct LR)

3. **Large-scale (256+ nodes):** Must use `train_agpt_2b_venv.sh`
   - Tarball broadcast for .venv is necessary
   - Derived TRAINING_STEPS ensures consistent token targets across scales
   - Only tested production path

