# Script Comparison: Training Launch Scripts

## Overview
This document compares three training launch scripts in the xpu_launcher directory:
1. `run_train.sh` - Base XPU launcher wrapper
2. `xpu_torchtitan/run_train_torchtitan.sh` - TorchTitan-specific training wrapper
3. `start_training.sh` - High-level training orchestrator with job monitoring

---

## High-Level Architecture

```
start_training.sh (user-facing orchestrator)
    ↓
run_train_torchtitan.sh (TorchTitan config & args builder)
    ↓
run_train.sh (base XPU launcher)
    ↓
xpu launch (low-level distributed launcher)
```

---

## Detailed Comparison

### 1. **run_train.sh** - Base XPU Launcher

**Purpose:** Low-level wrapper for `xpu launch` command, handles distributed training topology

**Key Responsibilities:**
- Launcher invocation (finds `xpu` command in PATH or fallback locations)
- Topology resolution: calculates NPROC_PER_NODE, NNODES, NPROC
- Scheduler auto-detection: PBS, SLURM, or manual specification
- Hostfile parsing for multi-node training
- Spare node configuration and failover setup
- Auto-retry setup for fault tolerance

**Command Line Interface:**
```bash
./run_train.sh single [--dry-run] [-- <train command...>]
./run_train.sh multi [hostfile] [--dry-run] [-- <train command...>]
```

**Environment Variables:**
- `XPU_CMD` - XPU launcher command (default: "xpu")
- `SCHEDULER` - Scheduler type (default: "auto")
- `NPROC_PER_NODE` - Devices per node (auto-detected, ~12 for Aurora)
- `NNODES` - Number of nodes for multi-mode
- `NPROC` - Total processes
- `AUTO_RETRY` - Enable auto-retry for multi-node (default: 1)
- `SPARE_NODES` - Number of spare nodes
- `SPARE_NODES_PERCENTAGE` - Spare as % of total
- `FAILOVER_PROFILE` - Failover strategy

**Key Features:**
- Auto-detects Python version ≥3.10 for launcher compatibility
- Falls back through multiple Python candidates
- Silences warnings unless `XPU_SHOW_WARNINGS=1`
- Supports both direct hostfile and scheduler-based topology
- No domain knowledge of training framework

---

### 2. **run_train_torchtitan.sh** - TorchTitan-Specific Wrapper

**Purpose:** Build TorchTitan training command with model/dataset/config management

**Key Responsibilities:**
- TorchTitan environment setup (requires MODEL path)
- Configuration registry: MODULE and CONFIG selection
- Dataset configuration (streaming or offline)
- Training hyperparameter defaults
- Resource monitoring setup
- Loss-based early termination configuration
- Delegates to `run_train.sh` for actual launching

**Command Line Interface:**
```bash
./run_train_torchtitan.sh single [--dry-run] [--resource-monitor] [-- <extra torchtitan args...>]
./run_train_torchtitan.sh multi [hostfile] [--dry-run] [--resource-monitor] [-- <extra torchtitan args...>]
```

**Environment Variables (Core):**
- `MODEL` or `MODEL_PATH` - **Required** path to model/tokenizer directory
- `MODULE` - TorchTitan config module (default: "llama3")
- `CONFIG` - Config callable (default: "llama3_debugmodel")
- `HF_ASSETS_PATH` - Override for HuggingFace assets (default: MODEL)
- `DATASET_NAME` - Dataset selection (default: "pg19_multinews")
- `DATASET_PATH` - Optional dataset path
- `SEQ_LEN` - Sequence length (default: 16384)
- `TRAINING_STEPS` - Steps to run (default: 100)
- `TORCHTITAN_ROOT` - Path to TorchTitan repo
- `LOG_DIR` - Output directory
- `CKPT_FOLDER` - Checkpoint folder name
- `TRAIN_PYTHON_BIN` - Python executable

**Resource Monitoring:**
- `RESOURCE_MONITOR` - Enable resource tracking (default: 0)
- `RESOURCE_INTERVAL` - Polling interval in seconds (default: 5)
- `RESOURCE_OUTPUT_DIR` - Where to save metrics

**Early Termination:**
- `LOSS_STD_TERMINATION_ENABLED` - Enable loss convergence check (default: 0)
- `LOSS_STD_THRESHOLD` - Loss std threshold (default: 0.001)
- `LOSS_STD_WINDOW` - Window size for std calc (default: 50)

**Key Features:**
- Validates MODEL path existence
- Extensive configuration printing for debugging
- Supports extra TorchTitan arguments via `--`
- Supports resource monitoring (optional wrapper)
- Detects Python binary from multiple fallback paths
- Delegates actual launch to `run_train.sh`

---

### 3. **start_training.sh** - High-Level Orchestrator

**Purpose:** User-friendly training job launcher with monitoring and summary generation

**Key Responsibilities:**
- Simple command-line argument parsing (no positional args)
- Training environment setup and validation
- Invokes `run_train_torchtitan.sh`
- Job monitoring (tracks execution time)
- Post-training summary generation via `train_monitor.py`
- User-facing output and progress tracking

**Command Line Interface:**
```bash
./start_training.sh [OPTIONS]

OPTIONS:
  --mode {single|multi}      Training mode (default: single)
  --hostfile FILE            Hostfile for multi-mode
  --model-path PATH          Model directory (required)
  --module NAME              Config module (default: llama3)
  --config NAME              Config callable (default: llama3_debugmodel)
  --dataset NAME             Dataset name (default: pg19_multinews)
  --training-steps N         Training steps (default: 18000)
  --log-dir DIR              Output directory
  --dry-run                  Print command without executing
  --help                     Show help
```

**Environment Variables:**
- `TRAIN_MODE` - Override mode (default: "single")
- `TRAIN_HOSTFILE` - Override hostfile path
- `LOG_DIR` - Output directory
- `TRAINING_STEPS` - Steps (default: 18000 for ~5 hours)
- `MONITOR_INTERVAL` - Log check frequency (default: 30s)
- `MODEL_PATH` - **Required** if not in args
- `MODULE`, `CONFIG`, `DATASET_NAME` - Training config
- Plus all vars supported by `run_train_torchtitan.sh`

**Key Features:**
- User-friendly named arguments (e.g., `--model-path` vs `MODEL`)
- Minimal required arguments (just `--model-path`)
- Validates required arguments upfront
- Prints human-readable training configuration
- Measures job duration in HH:MM:SS format
- Calls `train_monitor.py` to generate `TRAINING_SUMMARY.md`
- Supports dry-run mode

---

## Comparison Table

| Aspect | run_train.sh | run_train_torchtitan.sh | start_training.sh |
|--------|--------------|------------------------|-------------------|
| **Level** | Low-level | Mid-level | High-level |
| **Domain** | Generic distributed training | TorchTitan-specific | User-facing orchestration |
| **Launch Target** | Any Python training script | `titan_train.py` (TorchTitan) | `run_train_torchtitan.sh` |
| **Mode Syntax** | `single` / `multi [hostfile]` | `single` / `multi [hostfile]` | `--mode {single\|multi}` |
| **Arg Style** | Positional + shell env vars | Positional + shell env vars | Named arguments + env vars |
| **Required Args** | Train command after `--` | MODEL + train command | `--model-path` |
| **Framework Knowledge** | None | TorchTitan (configs, datasets) | None (delegates to run_train_torchtitan) |
| **Validation** | Hostfile exists, mode valid | MODEL path, TORCHTITAN_ROOT, mode | MODEL_PATH required, mode/hostfile match |
| **Configuration Output** | None | Detailed config summary (stderr) | Compact training config (stdout) |
| **Post-Training** | None | None | Generates TRAINING_SUMMARY.md |
| **Feature Flags** | None | Resource monitor, loss termination | Dry-run |
| **Dry-Run Support** | Yes | Yes | Yes |
| **Error Handling** | Basic validation | Extensive validation + helpful hints | Basic validation |

---

## Call Chain Examples

### Single-Node Training (End-to-End)

```bash
# User invokes:
./start_training.sh --model-path /path/model --training-steps 5000

# Which calls:
./xpu_torchtitan/run_train_torchtitan.sh single -- --training.steps 5000

# Which calls:
./run_train.sh single -- python titan_train.py --module llama3 --config llama3_debugmodel ...

# Which calls:
xpu launch -n 12 -ppn 12 -- python titan_train.py ...
```

### Multi-Node Training with Hostfile

```bash
# User invokes:
./start_training.sh --mode multi --hostfile ./hosts.txt --model-path /path/model

# Which calls:
./xpu_torchtitan/run_train_torchtitan.sh multi ./hosts.txt -- --training.steps 18000

# Which calls:
./run_train.sh multi ./hosts.txt -- python titan_train.py ...

# Which calls:
xpu launch --hostfile ./hosts.txt -nh 4 -n 48 -ppn 12 -- python titan_train.py ...
```

---

## Key Differences Summary

### **run_train.sh**
- ✅ Universal: works with any training script
- ✅ Minimal abstractions
- ✅ Full control via environment variables
- ❌ Requires knowledge of XPU launcher

### **run_train_torchtitan.sh**
- ✅ TorchTitan-aware: handles modules, configs, datasets
- ✅ Extensive configuration validation
- ✅ Resource monitoring support
- ✅ Better error messages
- ❌ TorchTitan-only
- ❌ Complex argument handling

### **start_training.sh**
- ✅ User-friendly: simple named arguments
- ✅ Post-training summary generation
- ✅ Clear configuration printing
- ✅ Minimal required inputs
- ❌ Less control (no resource monitor options visible)
- ❌ One abstraction level further from actual launcher

---

## When to Use Which

| Scenario | Script |
|----------|--------|
| Running arbitrary distributed Python code | `run_train.sh` |
| Training a TorchTitan model with custom args | `run_train_torchtitan.sh` |
| Quick training job with default configs | `start_training.sh` |
| Debugging launcher topology issues | `run_train.sh` |
| Fine-tuning TorchTitan hyperparameters | `run_train_torchtitan.sh` + extra args |
| Production training with monitoring | `start_training.sh` |
| Shared training script for team | `run_train_torchtitan.sh` |

---

## Environment Variable Precedence

**run_train.sh:**
1. Environment variables (XPU_CMD, SCHEDULER, etc.)
2. Hardcoded defaults
3. Auto-detection from scheduler

**run_train_torchtitan.sh:**
1. Environment variables (MODEL, MODULE, CONFIG, etc.)
2. Hardcoded defaults
3. Auto-detection (Python binary, LOG_DIR timestamp, etc.)

**start_training.sh:**
1. Command-line arguments (--model-path, --mode, etc.)
2. Environment variables (TRAIN_MODE, TRAIN_HOSTFILE, etc.)
3. Hardcoded defaults

---

## Configuration Passing Flow

```
start_training.sh
├─ Accepts: --model-path, --mode, --training-steps, etc.
│           Environment vars: TRAIN_MODE, LOG_DIR, SEQ_LEN, etc.
└─ Exports as shell env vars
   └─> run_train_torchtitan.sh
       ├─ Reads: MODEL, MODULE, CONFIG, DATASET_NAME, etc.
       ├─ Builds TorchTitan command: titan_train.py --module --config ...
       └─ Calls run_train.sh with train command
          └─> run_train.sh
              ├─ Reads: XPU_CMD, SCHEDULER, NPROC, etc.
              ├─ Resolves topology from hostfile/scheduler
              └─ Calls: xpu launch [topology] -- <train command>
```

---

## Error Handling Differences

### run_train.sh
- Checks hostfile existence
- Validates mode (single/multi)
- Python version check (≥3.10)

### run_train_torchtitan.sh
- Validates MODEL path exists
- Checks TORCHTITAN_ROOT exists
- Validates base launcher executable
- Helpful hints on errors (e.g., chmod +x)

### start_training.sh
- Validates MODEL_PATH required
- Checks MODE/HOSTFILE compatibility
- Simple error messages

