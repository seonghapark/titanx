# Before & After Comparison: Hyperparameter Logging

## BEFORE: Limited Parameter Output

### Launch-Time Output (BEFORE)
```
[ARGS] mode              = single
[ARGS] MODEL             = /path/to/model
[ARGS] MODULE/CONFIG     = llama3 / llama3_8b
[ARGS] DATASET_NAME      = c4
[ARGS] DATASET_PATH      = allenai/c4
[ARGS] TRAINING_STEPS    = 2000
[ARGS] SEQ_LEN           = 4096
[ARGS] TORCHTITAN_ROOT   = /lus/flare/.../torchtitan_repo
[ARGS] HF_ASSETS_PATH    = /path/to/model
[ARGS] LOG_DIR           = /path/to/outputs/xpu_torchtitan_20260929_143022
[ARGS] CKPT_FOLDER       = checkpoint
[ARGS] CKPT              = <unset>
[ARGS] RESOURCE_MONITOR  = 0
[ARGS] RESOURCE_INTERVAL = 5
[ARGS] RESOURCE_OUTPUT   = /path/to/outputs/xpu_torchtitan_20260929_143022/resource_metrics
[ARGS] TRAIN_PYTHON_BIN  = /lus/flare/projects/datascience/seonghapark/venv/bin/python
[ARGS] topology          = NNODES=auto NPROC_PER_NODE=4 NPROC=auto SPARE_NODES=auto AUTO_RETRY=1(multi)
[ARGS] extra train args  = <none>
```

**Issues:**
- ❌ No batch size information
- ❌ No learning rate visible
- ❌ No tensor/pipeline/data parallelism sizes
- ❌ Hard to organize and parse
- ❌ No indication of YaRN or other modifications
- ❌ No model architecture details

### Training-Time Output (BEFORE)
```
torchtitan version: 0.0.0 (0.0.0 means __version__ is not defined correctly).
Model llama3 llama3_8b size: 8,030,261,248 total parameters
Training starts at step 1
```

**Issues:**
- ❌ No hyperparameter logging at all
- ❌ Only model size is shown
- ❌ No learning rate, optimizer settings, or parallelism details
- ❌ Difficult to verify configuration was applied correctly

---

## AFTER: Comprehensive Parameter Output

### Launch-Time Output (AFTER)
```
================================================================================
TRAINING LAUNCH CONFIGURATION
================================================================================

[LAUNCH MODE]
  Mode                   = single
  Topology               = NNODES=auto NPROC_PER_NODE=4 NPROC=auto
  Auto Retry             = 1(multi)
  Spare Nodes            = auto

[MODEL & DATASET]
  Model Path             = /path/to/Llama-3.1-8B
  Module/Config          = llama3 / llama3_8b
  Dataset Name           = c4
  Dataset Path           = allenai/c4
  HF Assets Path         = /path/to/Llama-3.1-8B

[TRAINING HYPERPARAMETERS]
  Training Steps         = 2000
  Sequence Length        = 4096
  Loss Std Termination   = 0 (threshold: 0.001, window: 50)

[SYSTEM & PATHS]
  TorchTitan Root        = /lus/flare/.../torchtitan_repo
  Log Directory          = /path/to/outputs/xpu_torchtitan_20260929_143022
  Checkpoint Folder      = checkpoint
  Checkpoint Load        = <unset>
  Python Binary          = /lus/flare/projects/datascience/seonghapark/venv/bin/python

[MONITORING]
  Resource Monitor       = 0
  Resource Interval      = 5s
  Resource Output Dir    = /path/to/outputs/xpu_torchtitan_20260929_143022/resource_metrics

[EXTRA ARGUMENTS]
  Additional Args        = <none>

================================================================================
```

**Improvements:**
- ✅ Organized into logical sections
- ✅ Clear headers for easy scanning
- ✅ Better alignment and readability
- ✅ All essential launch parameters visible

### Training-Time Output (AFTER)
```
torchtitan version: 0.0.0 (0.0.0 means __version__ is not defined correctly).
Model llama3 llama3_8b size: 8,030,261,248 total parameters
Training starts at step 1

================================================================================
TRAINING HYPERPARAMETERS AND CONFIGURATION
================================================================================

[MODEL]
  Model Name: llama3
  Model Flavor: llama3_8b
  Architecture: Llama
  Rope Scaling Type: yarn
  Rope Theta: 10000000.0

[TRAINING]
  Learning Rate: 3e-05
  Total Training Steps: 2000
  Sequence Length: 4096
  Global Batch Size: 128
  Micro Batch Size: 4
  Gradient Accumulation Steps: 32
  Data Type: bfloat16

[OPTIMIZER]
  Optimizer Type: AdamW
  Learning Rate: 3e-05
  Weight Decay: 0.1
  Betas (Adam): (0.9, 0.95)
  Epsilon: 1e-08

[LR SCHEDULER]
  LR Scheduler Type: CosineAnnealingWarmRestarts
  Warmup Steps: 400
  Min LR Ratio: 0.1

[PARALLELISM]
  Tensor Parallel Size (TP): 4
  Pipeline Parallel Size (PP): 1
  Data Parallel Size (DP): 8
  Context Parallel Size (CP): 1
  SPMD Backend: spmd_types
  Sequence Parallel: DISABLED

[DATA]
  Dataset Name: c4
  Dataset Path: allenai/c4

[ARCHITECTURE DETAILS]
  Hidden Dimension: 4096
  Number of Layers: 32
  Number of Attention Heads: 32
  Number of KV Heads: 8
  Vocabulary Size: 128256
  Intermediate Dimension (FFN): 14336

[MEMORY OPTIMIZATION]
  Activation Checkpoint Type: SelectiveAC
  Compile Enabled: True
  Compiled Components: model

[CHECKPOINT]
  Checkpoint Enabled: True
  Checkpoint Folder: checkpoint
  Checkpoint Interval (steps): steps

[DISTRIBUTED]
  World Size: 32
  Global Rank: 0
  Local Rank: 0

================================================================================
```

**Improvements:**
- ✅ **Learning Rate clearly visible** (3e-05)
- ✅ **Tensor Parallel Size visible** (TP: 4)
- ✅ **Architecture with YaRN detected** (Llama with rope_scaling: yarn)
- ✅ **Batch sizes logged** (Global: 128, Micro: 4)
- ✅ **Pipeline Parallel visible** (PP: 1)
- ✅ **Data Parallel visible** (DP: 8)
- ✅ **Full model architecture details** (32 layers, 4096 hidden, 32 heads, etc.)
- ✅ **Optimizer settings visible** (AdamW with specific betas and epsilon)
- ✅ **LR Scheduler details** (CosineAnnealing with 400 warmup steps)
- ✅ **Memory optimization settings** (SelectiveAC, compiled model)
- ✅ **Distributed setup** (World size 32, ranks)
- ✅ Organized, scannable format
- ✅ Can quickly verify configuration was applied correctly

---

## Value Added by This Enhancement

### 1. **Debugging & Verification**
- **BEFORE:** Had to dig through config files and terminal logs to verify settings
- **AFTER:** All critical parameters in one place at startup

### 2. **Experiment Tracking**
- **BEFORE:** Hard to correlate which hyperparameters were used for which run
- **AFTER:** Complete configuration snapshot in each run's logs

### 3. **YaRN Detection**
- **BEFORE:** No visibility into positional embedding modifications
- **AFTER:** `rope_scaling: yarn` and `rope_theta` clearly logged

### 4. **Parallelism Verification**
- **BEFORE:** TP/PP/DP sizes not visible in output
- **AFTER:** All parallelism dimensions clearly displayed

### 5. **Batch Size Clarity**
- **BEFORE:** No batch size information in logs
- **AFTER:** Both global and micro batch sizes visible

### 6. **Model Architecture Transparency**
- **BEFORE:** Only total parameters shown
- **AFTER:** Complete architecture breakdown (layers, heads, FFN size, vocab, etc.)

### 7. **Learning Rate Assurance**
- **BEFORE:** Had to check config files
- **AFTER:** Learning rate prominently displayed at startup

### 8. **Optimizer Settings Visibility**
- **BEFORE:** Adam hyperparameters not logged
- **AFTER:** Betas, epsilon, weight decay all visible

---

## Real-World Use Cases

### Use Case 1: Running an Experiment
```
$ MODEL=/models/llama3-8b ./run_train_torchtitan.sh single -- \
    --training.lr 1e-4 \
    --training.seq_len 8192
```

**BEFORE:** You'd have to manually verify that 1e-4 was actually used
**AFTER:** See `Learning Rate: 1e-4` immediately in the logs ✅

### Use Case 2: Debugging Unexpected Behavior
```
$ # Training is slower than expected
$ # Question: What's the actual configuration?
```

**BEFORE:** Check multiple config files and git branches
**AFTER:** Look at training logs startup section, see everything in one place ✅

### Use Case 3: Reproducing an Experiment
```
$ # Need to match exact configuration from run X
$ # Where are all the hyperparameters recorded?
```

**BEFORE:** Scattered across multiple files, hard to collect
**AFTER:** Complete snapshot in logs right at startup ✅

---

## Summary Table

| Aspect | Before | After |
|--------|--------|-------|
| Learning Rate visible | ❌ | ✅ |
| Tensor Parallel Size visible | ❌ | ✅ |
| Architecture name visible | ❌ | ✅ |
| YaRN detection | ❌ | ✅ |
| Batch sizes visible | ❌ | ✅ |
| Optimizer settings visible | ❌ | ✅ |
| Model architecture details | ❌ | ✅ |
| Parallelism config visible | ❌ | ✅ |
| LR Scheduler details | ❌ | ✅ |
| Organized output | ❌ | ✅ |
| Easy to scan | ❌ | ✅ |
| Complete configuration | ❌ | ✅ |

**Result:** Users now have full transparency into training configuration at startup, making debugging, verification, and reproducibility significantly easier.
