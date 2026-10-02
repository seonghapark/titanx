# Script Comparison: train_agpt20b_yarn.sh vs run_train_torchtitan.sh

## Executive Summary

| Aspect | train_agpt20b_yarn.sh | run_train_torchtitan.sh |
|--------|----------------------|------------------------|
| **Purpose** | High-level wrapper for YaRN-specific training | Universal TorchTitan training launcher |
| **Scope** | agpt-20b + YaRN context extension only | Any TorchTitan model (llama3, agpt, gpt_oss, etc.) |
| **Abstraction Level** | High (hides implementation details) | Low (exposes full configuration) |
| **Flexibility** | Low (opinionated defaults) | High (extensive customization) |
| **Use Case** | "I want to train agpt-20b with YaRN" | "I want full control over TorchTitan training" |
| **Target User** | Data scientist wanting quick start | ML engineers, framework developers |

---

## Detailed Comparison

### 1. Purpose & Design Philosophy

#### train_agpt20b_yarn.sh
- **Purpose**: Simplified, opinionated launcher specifically for YaRN context extension of agpt-20b
- **Philosophy**: "Just work" — hide complexity, provide sensible defaults
- **Audience**: Users who want to extend agpt-20b to 16K+ tokens with minimal configuration
- **Abstraction**: High — delegates to run_train_torchtitan.sh after setup

#### run_train_torchtitan.sh
- **Purpose**: Universal launcher for any TorchTitan training job
- **Philosophy**: Flexibility first — expose all options, minimal assumptions
- **Audience**: ML engineers, researchers running diverse training workloads
- **Abstraction**: Low — directly configures TorchTitan parameters

---

### 2. Interface & Parameters

#### train_agpt20b_yarn.sh

**Accepts positional arguments:**
```bash
./train_agpt20b_yarn.sh [seq_len] [training_steps] [num_nodes]
```

**Example:**
```bash
./train_agpt20b_yarn.sh 16384 400 8
```

**Parameter Summary:**
| Param | Default | What It Does |
|-------|---------|--------------|
| `seq_len` | 16384 | Target sequence length |
| `training_steps` | 400 | Number of training steps |
| `num_nodes` | 8 | Number of nodes to use |

**Accepts environment variables:**
- `CKPT` — Override DCP checkpoint path
- `DATASET_NAME` — Override dataset (default: pg19_multinews)
- `TP`, `PP`, `CP` — Data/model parallelism settings

**Fixed parameters (hardcoded):**
- Model name: agpt-20b-v2-512n-step-4000
- Config: agpt_20b_yarn
- Module: agpt
- Safetensors path: /lus/flare/projects/datascience/seonghapark/agpt-20b-v2-512n-step-4000-safetensors

---

#### run_train_torchtitan.sh

**Accepts mode as first argument:**
```bash
./run_train_torchtitan.sh single [--dry-run] [--resource-monitor] [-- <extra args>]
./run_train_torchtitan.sh multi [hostfile] [--dry-run] [-- <extra args>]
```

**Example:**
```bash
./run_train_torchtitan.sh multi /path/to/hostfile -- --training.steps 2000
```

**Accepts extensive environment variables:**
| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` / `MODEL_PATH` | (required) | Path to model/tokenizer assets |
| `MODULE` | llama3 | Config registry module |
| `CONFIG` | llama3_debugmodel | Config callable name |
| `HF_ASSETS_PATH` | MODEL | HF model path override |
| `DATASET_NAME` | pg19_multinews | Dataset name |
| `DATASET_PATH` | (empty) | Custom dataset path |
| `SEQ_LEN` | 16384 | Sequence length |
| `LOG_DIR` | torchtitan/outputs/<timestamp> | Logging directory |
| `CKPT_FOLDER` | checkpoint | Checkpoint folder |
| `TRAINING_STEPS` | 100 | Training steps |
| `TORCHTITAN_ROOT` | ./torchtitan_repo | TorchTitan root directory |
| `TRAIN_PYTHON_BIN` | Auto-detected | Python binary path |
| `RESOURCE_MONITOR` | 0 | Enable resource monitoring |
| `LOSS_STD_TERMINATION_ENABLED` | 0 | Enable early termination |
| Plus 8 more resource monitoring parameters |

**Accepts flags:**
- `--dry-run` — Print command without executing
- `--resource-monitor` — Enable resource monitoring
- `--resource-interval` — Resource monitor poll interval
- `--resource-output-dir` — Where to save resource metrics
- `--` — Pass additional TorchTitan arguments (unlimited)

---

### 3. Input Validation

#### train_agpt20b_yarn.sh

**Validates:**
```bash
✓ DCP checkpoint exists
✓ Safetensors model directory exists
✓ YaRN rope_scaling in config.json
✓ run_train_torchtitan.sh exists in current directory
```

**Error Handling:**
- Explicit error messages for missing paths
- Clear instructions on how to fix (e.g., "set CKPT environment variable")
- Exits immediately on validation failure

**Example:**
```bash
Error: YaRN rope_scaling not found in config.json
Please run: python3 yarn_validation_agpt20b.py
```

---

#### run_train_torchtitan.sh

**Validates:**
```bash
✓ Mode is "single" or "multi"
✓ MODEL path exists and is readable
✓ Hostfile exists (if provided in multi mode)
✓ PBS_NODEFILE exists (if multi mode and no hostfile)
✓ TORCHTITAN_ROOT exists
✓ Extra arguments don't override incompatibly
✓ Base launch script (run_train.sh) is executable
```

**Error Handling:**
- Validates mode, paths, and dependencies
- Prints usage and hints on failure
- More generic error messages (not YaRN-specific)

**Example:**
```bash
error: MODEL path not found: /path/to/model
       e.g. MODEL=${TORCHTITAN_ROOT}/tests/assets/tokenizer
```

---

### 4. Configuration & Execution

#### train_agpt20b_yarn.sh

**What it does:**

1. **Parse input** — Extract seq_len, training_steps, num_nodes from args
2. **Validate environment** — Check paths and YaRN config
3. **Set hardcoded values** — agpt, agpt_20b_yarn, specific paths
4. **Display configuration** — Echo setup for user verification
5. **Determine launch mode** — "single" if 1 node, "multi" otherwise
6. **Launch** — Calls `./run_train_torchtitan.sh $LAUNCH_MODE`

**Environment setup** (explicit):
```bash
export MODULE="agpt"
export CONFIG="agpt_20b_yarn"
export SEQ_LEN=$SEQ_LEN
export TRAINING_STEPS=$TRAINING_STEPS
export DATASET_NAME=$DATASET_NAME
export CKPT=$DCP_CKPT_PATH
export MODEL_PATH=$SAFETENSORS_PATH
export ACTIVATION_CHECKPOINT="none"
export TP=${TP:-8}
export PP=${PP:-1}
export CP=${CP:-1}
```

**Passes to run_train_torchtitan.sh:**
```bash
./run_train_torchtitan.sh [single|multi]
```

---

#### run_train_torchtitan.sh

**What it does:**

1. **Parse arguments** — Extract mode, hostfile, flags, extra args
2. **Validate inputs** — Check paths, modes, dependencies
3. **Set defaults** — Apply fallbacks for unspecified variables
4. **Build TorchTitan command** — Construct full argument list
5. **Display configuration** — Print full launch summary
6. **Delegate to run_train.sh** — Universal MPI/PBS launcher
7. **Execute** — Run titan_train.py with resource monitoring (optional)

**Environment processing** (flexible):
```bash
# Merges defaults with user-provided variables
MODULE="${MODULE:-llama3}"
CONFIG="${CONFIG:-llama3_debugmodel}"
SEQ_LEN="${SEQ_LEN:-16384}"
TRAINING_STEPS="${TRAINING_STEPS:-100}"
# ... 20+ more variables with defaults
```

**Builds command dynamically:**
```bash
TRAIN_CMD=(
  "$TRAIN_PYTHON_BIN" "${SCRIPT_DIR}/titan_train.py"
  "--module" "$MODULE"
  "--config" "$CONFIG"
  "--hf_assets_path" "$HF_ASSETS_PATH"
  # ... more args based on variables
)

# Optional: wrap with resource monitor
if [[ "$RESOURCE_MONITOR" == "1" ]]; then
  TRAIN_CMD=( "$TRAIN_PYTHON_BIN" "$RESOURCE_MONITOR_SCRIPT" ... "${TRAIN_CMD[@]}" )
fi
```

**Passes to run_train.sh:**
```bash
./run_train.sh [single|multi] [hostfile] [--dry-run] [-- ${TRAIN_CMD[@]}]
```

---

### 5. Output & Logging

#### train_agpt20b_yarn.sh

**Printed output (simple):**
```
==========================================
YaRN Context Extension Training
==========================================
Model: agpt-20b-v2-512n-step-4000
Sequence Length: 16384 tokens
Training Steps: 400
Number of Nodes: 8
Dataset: pg19_multinews
DCP Checkpoint: /path/to/checkpoint
==========================================

✓ All validations passed

Environment variables set:
  MODULE=agpt
  CONFIG=agpt_20b_yarn
  SEQ_LEN=16384
  TRAINING_STEPS=400
  TP=8 PP=1 CP=1

Launching single-node training (local testing)
Starting training...

==========================================
Training completed successfully!
==========================================
```

**Logging:**
- Direct to stdout
- User must redirect to file manually if needed
- Minimal summary, focused on YaRN training

---

#### run_train_torchtitan.sh

**Printed output (comprehensive):**
```
================================================================================
TRAINING LAUNCH CONFIGURATION
================================================================================

[LAUNCH MODE]
  Mode                   = multi (hostfile=/path/to/hosts)
  Topology               = NNODES=8 NPROC_PER_NODE=4 NPROC=32
  Auto Retry             = 1
  Spare Nodes            = auto

[MODEL & DATASET]
  Model Path             = /path/to/model
  Module/Config          = llama3 / llama3_8b
  Dataset Name           = pg19_multinews
  ...

[TRAINING HYPERPARAMETERS]
  Training Steps         = 100
  Sequence Length        = 16384
  Loss Std Termination   = 0 (threshold: 0.001, window: 50)

... (8 more sections)

Running: /path/to/run_train.sh multi /path/to/hosts -- python ...
```

**Logging:**
- Prints configuration to stderr (separate from execution output)
- Can redirect separately from training logs
- Shows full command line for transparency

---

### 6. Extensibility & Customization

#### train_agpt20b_yarn.sh

**Easy to customize:**
- Pass seq_len, training_steps, num_nodes as arguments
- Override CKPT, DATASET_NAME via env vars
- Adjust TP/PP/CP for parallelism

**Hard to customize:**
- Cannot change model, module, or config
- Hardcoded paths would require editing script
- No way to pass arbitrary TorchTitan flags
- No resource monitoring option
- No dry-run mode

**Use when:**
- ✓ Training agpt-20b specifically
- ✓ Want quick, simple command
- ✓ Don't need advanced options

**Don't use when:**
- ✗ Training other models
- ✗ Need resource monitoring
- ✗ Want to experiment with configs
- ✗ Need to pass custom TorchTitan args

---

#### run_train_torchtitan.sh

**Easy to customize:**
- Override any environment variable
- Pass unlimited extra TorchTitan args via `-- arg1 arg2 ...`
- Enable/disable features (resource monitor, loss termination, etc.)
- Choose any model, module, config combination
- Dry-run mode for testing commands

**Hard to customize:**
- More environment variables to remember
- Error messages less specific to your use case
- Need to know TorchTitan argument names

**Use when:**
- ✓ Training any TorchTitan model
- ✓ Need advanced features (monitoring, custom args)
- ✓ Experimenting with configurations
- ✓ Want transparency into all settings

**Don't use when:**
- ✗ Want simplest possible command
- ✗ Training agpt-20b and want quick start
- ✗ Don't want to learn all the variables

---

### 7. Dependency Chain

#### train_agpt20b_yarn.sh

```
train_agpt20b_yarn.sh (user entry point)
  ↓
  Sets MODULE, CONFIG, MODEL_PATH, etc.
  ↓
run_train_torchtitan.sh (universal launcher)
  ↓
  Sets HF_ASSETS_PATH, builds TRAIN_CMD
  ↓
run_train.sh (MPI/PBS wrapper)
  ↓
  Handles node allocation, environment setup
  ↓
titan_train.py (TorchTitan entry point)
  ↓
  torchtitan/train/main()
```

**Separation of concerns:**
- `train_agpt20b_yarn.sh` — YaRN-specific configuration
- `run_train_torchtitan.sh` — TorchTitan argument building
- `run_train.sh` — MPI/PBS orchestration
- `titan_train.py` — TorchTitan execution

---

#### run_train_torchtitan.sh

```
run_train_torchtitan.sh (user entry point)
  ↓
  Sets all TorchTitan variables with defaults
  ↓
  Builds complete TRAIN_CMD array
  ↓
run_train.sh (MPI/PBS wrapper)
  ↓
  Handles node allocation, environment setup
  ↓
titan_train.py (TorchTitan entry point)
  ↓
  torchtitan/train/main()
```

**Simplified chain:**
- Single point of configuration (run_train_torchtitan.sh)
- Direct delegation to MPI/PBS layer

---

### 8. Error Recovery & Debugging

#### train_agpt20b_yarn.sh

**Debugging approach:**
- Pre-flight validation before delegating
- Clear error messages with remediation steps
- Stops early if YaRN config is wrong

**If something goes wrong:**
```bash
$ ./train_agpt20b_yarn.sh 16384 400 8
Error: DCP checkpoint not found at /path/to/checkpoint
Please set CKPT environment variable to the path of your step-4000 checkpoint
```

**Troubleshooting:**
- Error message tells you exactly what to fix
- No need to understand TorchTitan internals

---

#### run_train_torchtitan.sh

**Debugging approach:**
- Comprehensive configuration logging
- Shows exact command being executed
- Supports --dry-run mode to inspect command

**If something goes wrong:**
```bash
$ run_train_torchtitan.sh multi /hosts -- --training.steps invalid
# Error would come from torchtitan/train.py, not the launcher

# Use --dry-run to inspect:
$ run_train_torchtitan.sh multi /hosts --dry-run
# Prints the full command without executing
```

**Troubleshooting:**
- Review printed configuration before execution
- Use --dry-run to see exact commands
- Check TorchTitan error messages
- Requires knowledge of TorchTitan argument names

---

### 9. Practical Example Comparison

#### Task: Train agpt-20b with YaRN at 16K length for 400 steps on 8 nodes

**Using train_agpt20b_yarn.sh:**
```bash
./train_agpt20b_yarn.sh 16384 400 8
```

**Using run_train_torchtitan.sh:**
```bash
export MODULE=agpt
export CONFIG=agpt_20b_yarn
export MODEL_PATH=/path/to/agpt-20b-v2-512n-step-4000-safetensors
export CKPT=/path/to/dcp/step-4000
export SEQ_LEN=16384
export TRAINING_STEPS=400

./run_train_torchtitan.sh multi /path/to/hostfile
```

**Difference:**
- train_agpt20b_yarn.sh: **1 command** (shorter, opinionated)
- run_train_torchtitan.sh: **7 commands** (longer, flexible)

---

#### Task: Train agpt-20b with 32K context, custom dataset, skip steps 50-200

**Using train_agpt20b_yarn.sh:**
```bash
# Can't do this — no way to specify custom dataset or skip specific steps
# Would need to edit the script
```

**Using run_train_torchtitan.sh:**
```bash
export MODULE=agpt
export CONFIG=agpt_20b_yarn
export MODEL_PATH=/path/to/model
export CKPT=/path/to/step-4000
export SEQ_LEN=32768
export TRAINING_STEPS=400
export DATASET_NAME=my_custom_dataset
export DATASET_PATH=/path/to/data

./run_train_torchtitan.sh multi /path/to/hostfile -- \
  --dataloader.resume_from_checkpoint /path/to/step-50
```

**Difference:**
- train_agpt20b_yarn.sh: **Not possible** (too rigid)
- run_train_torchtitan.sh: **Fully supported** (flexible)

---

### 10. When to Use Each Script

#### Use train_agpt20b_yarn.sh if:

✅ You're **specifically** training agpt-20b with YaRN  
✅ You want the **simplest possible command**  
✅ You're **not comfortable** with shell scripting  
✅ You like **opinionated defaults**  
✅ You want **clear, domain-specific error messages**  
✅ You're doing a **standard training run** (seq_len, steps, nodes)  

**Example users:**
- Data scientist running context extension
- Student learning about YaRN
- Practitioner running standard workload

---

#### Use run_train_torchtitan.sh if:

✅ You're training **any TorchTitan model** (not just agpt-20b)  
✅ You need **fine-grained control** over every option  
✅ You want **resource monitoring** or advanced features  
✅ You're **experimenting** with different configs  
✅ You need **custom TorchTitan arguments**  
✅ You're running **unusual/complex training** setups  

**Example users:**
- ML engineer optimizing training
- Framework developer
- Researcher comparing configurations
- DevOps operator managing diverse workloads

---

## Summary Table

| Category | train_agpt20b_yarn.sh | run_train_torchtitan.sh |
|----------|----------------------|------------------------|
| **Lines of code** | 107 | 314 |
| **Supported models** | 1 (agpt-20b) | All TorchTitan models |
| **Environment vars** | 3-5 | 20+ |
| **Argument complexity** | Low (3 args) | High (mode + flags + extras) |
| **Customization** | Limited | Extensive |
| **Learning curve** | Gentle | Steep |
| **Speed to launch** | ~1 minute | ~5 minutes |
| **For beginners** | ✅ Excellent | ✅ Good with docs |
| **For experts** | ✅ Fine | ✅ Excellent |
| **Error messages** | Domain-specific | Generic, TorchTitan-focused |
| **Debugging** | Easy | Moderate |
| **Dry-run mode** | ✗ No | ✅ Yes |
| **Resource monitoring** | ✗ No | ✅ Yes |

---

## Conclusion

**train_agpt20b_yarn.sh** is a **convenience wrapper** — it makes the common case (training agpt-20b with YaRN) as simple as possible.

**run_train_torchtitan.sh** is a **flexible launcher** — it handles all TorchTitan training scenarios with full configurability.

Think of it like this:
- **train_agpt20b_yarn.sh** = "Press this button to train agpt-20b with YaRN"
- **run_train_torchtitan.sh** = "Full control panel for TorchTitan training"

Use the button if that's what you're doing. Use the control panel if you need anything else.
