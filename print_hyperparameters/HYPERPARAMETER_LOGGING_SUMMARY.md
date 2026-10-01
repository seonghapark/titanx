# TorchTitan Training Hyperparameter Logging Enhancement

## Overview
Enhanced the TorchTitan training pipeline to print comprehensive hyperparameters, parameters, and arguments when training runs. This ensures all critical configuration details are logged for debugging, validation, and experiment tracking purposes.

## Changes Made

### 1. Shell Script Update: `run_train_torchtitan.sh`

**Location**: Lines 258-297

**What's printed at startup**:
```
================================================================================
TRAINING LAUNCH CONFIGURATION
================================================================================

[LAUNCH MODE]
  Mode                   = single|multi
  Topology               = NNODES=... NPROC_PER_NODE=... NPROC=... 
  Auto Retry             = ...
  Spare Nodes            = ...

[MODEL & DATASET]
  Model Path             = <path to model>
  Module/Config          = llama3 / llama3_debugmodel
  Dataset Name           = pg19_multinews|c4|custom
  Dataset Path           = <path or default>
  HF Assets Path         = <path>

[TRAINING HYPERPARAMETERS]
  Training Steps         = 100 (or user-specified)
  Sequence Length        = 16384 (or user-specified)
  Loss Std Termination   = 0|1 (threshold: 0.001, window: 50)

[SYSTEM & PATHS]
  TorchTitan Root        = <repo path>
  Log Directory          = ./outputs/xpu_torchtitan_TIMESTAMP
  Checkpoint Folder      = checkpoint
  Checkpoint Load        = <or unset>
  Python Binary          = /path/to/venv/bin/python

[MONITORING]
  Resource Monitor       = 0|1
  Resource Interval      = 5s
  Resource Output Dir    = <path>/resource_metrics

[EXTRA ARGUMENTS]
  Additional Args        = <any extra CLI args or none>

================================================================================
```

### 2. Python Trainer Enhancement: `torchtitan_repo/torchtitan/trainer.py`

**New method**: `_log_training_hyperparameters()` at line 787

**When called**: At the start of the train() method (line 1028), right after "Training starts at step X" log

**What's printed during training startup**:

#### [MODEL] Section
- Model Name (e.g., "llama3")
- Model Flavor (e.g., "llama3_8b")  
- Architecture class name (e.g., "Llama")
- **rope_scaling** (if modified with YaRN or other positional embedding methods)
- **rope_theta** (if modified)

#### [TRAINING] Section
- **Learning Rate** (from config.optimizer.lr)
- **Total Training Steps**
- **Sequence Length**
- **Global Batch Size**
- **Micro Batch Size**
- **Gradient Accumulation Steps**
- **Data Type** (float32, bfloat16, etc.)

#### [OPTIMIZER] Section
- Optimizer Type (AdamW, SGD, etc.)
- Learning Rate
- Weight Decay
- Betas (if Adam)
- Epsilon (if Adam)

#### [LR SCHEDULER] Section
- LR Scheduler Type (CosineAnnealingWarmRestarts, linear_warmup, etc.)
- Warmup Steps (if available)
- Min LR Ratio (if available)

#### [PARALLELISM] Section
- **Tensor Parallel Size (TP)**
- **Pipeline Parallel Size (PP)**
- **Data Parallel Size (DP)**
- **Context Parallel Size (CP)**
- SPMD Backend (spmd_types or local_tensor)
- Sequence Parallel enabled/disabled

#### [DATA] Section
- Dataset Name
- Dataset Path

#### [FILE PATHS] Section
- **HF Assets Path** (base path for model assets)
- **Tokenizer Config** path (tokenizer_config.json) - if file exists
- **Tokenizer File** path (tokenizer.json) - if file exists
- **Model Config** path (config.json) - if file exists

#### [ARCHITECTURE DETAILS] Section
- Hidden Dimension
- Number of Layers
- Number of Attention Heads
- Number of KV Heads
- Vocabulary Size
- Intermediate Dimension (FFN size)

#### [MEMORY OPTIMIZATION] Section
- Activation Checkpoint Type (SelectiveAC, MemoryBudgetAC, etc.)
- Compile Enabled (True/False)
- Compiled Components (model, fsdp, etc. if enabled)

#### [CHECKPOINT] Section
- Checkpoint Enabled
- Checkpoint Folder
- Checkpoint Interval

#### [DISTRIBUTED] Section
- World Size
- Global Rank
- Local Rank

## Key Features

✅ **Comprehensive Coverage**: Logs all major hyperparameters including:
- Learning rate
- Tensor parallel size
- Architecture name and modifications (YaRN, etc.)
- Batch size (micro and global)
- Sequence length
- All optimizer settings
- All parallelism configurations
- Model architecture details

✅ **Safe Implementation**: Uses `hasattr()` checks for optional attributes like rope_scaling and rope_theta, so it works with different model types

✅ **Distributed-Aware**: Handles both single-GPU and multi-GPU distributed training scenarios

✅ **Clean Output**: Organized in logical sections with clear headers for easy scanning in logs

✅ **No Performance Impact**: Logging only happens once at startup, not during training iterations

## Files Modified

1. **`/lus/flare/projects/datascience/seonghapark/xpu_launcher/xpu_torchtitan/run_train_torchtitan.sh`**
   - Lines 258-297: Reformatted configuration output

2. **`/lus/flare/projects/datascience/seonghapark/xpu_launcher/xpu_torchtitan/torchtitan_repo/torchtitan/trainer.py`**
   - Lines 787-886: Added `_log_training_hyperparameters()` method
   - Line 1028: Added call to log hyperparameters at training startup

## Testing

Both files have been verified for syntax correctness:
- Shell script: ✅ `bash -n` validation passed
- Python file: ✅ `python3 -m py_compile` validation passed

## Usage

Simply run your training script as usual:
```bash
MODEL=/path/to/hf/model ./run_train_torchtitan.sh single

# Or with custom parameters:
MODEL=/path/to/hf/model CONFIG=llama3_8b \
  TRAINING_STEPS=2000 SEQ_LEN=4096 \
  ./run_train_torchtitan.sh multi /path/to/hosts
```

The hyperparameter logs will appear in stderr during startup, right after "Training starts at step 1".

## Benefits

1. **Debugging**: Easily verify that intended hyperparameters were actually used
2. **Reproducibility**: Complete record of all settings for each training run
3. **Experimentation**: Quick sanity check that parameter overrides worked correctly
4. **Log Analysis**: All critical parameters in one easy-to-find section
5. **Documentation**: Serves as inline documentation of what parameters are available
