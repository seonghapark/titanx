# Hyperparameter Tracking Status Report

## What IS Currently Printed

### Pre-Training (from `run_train_torchtitan.sh`)

```
[LAUNCH MODE]
  Topology = NNODES, NPROC_PER_NODE, NPROC, Auto Retry, Spare Nodes

[TRAINING HYPERPARAMETERS]
  Training Steps ✓
  Sequence Length ✓
  Loss Std Termination ✓

[MODEL & DATASET]
  Model Path, Module/Config, Dataset Name, Dataset Path
```

### During Training (from TorchTitan startup)

TorchTitan prints its full config including:
- Optimizer settings (learning rate, betas, eps, weight_decay)
- Model architecture (num_layers, hidden_size, num_heads, vocab_size)
- Distributed parallelism (DP, TP, PP degrees)
- Precision settings (dtype, mixed precision)
- Training settings (batch size, max norm, etc.)

---

## What is NOT Currently Printed

### Missing from Pre-Training Phase

| Category | Parameter | Why Missing | Status |
|----------|-----------|------------|--------|
| **Optimization** | Learning rate | Not passed to wrapper script | ❌ |
| **Optimization** | Optimizer type | Hardcoded in config_registry | ❌ |
| **Optimization** | Momentum | Not exposed in interface | ❌ |
| **Optimization** | Betas | In config_registry but not printed | ❌ |
| **Optimization** | Gradient clipping epsilon | Not in launch config section | ❌ |
| **Training Schedule** | Epochs | Implicit via steps | ❌ |
| **Training Schedule** | Gradient accumulation steps | Derived at runtime | ❌ |
| **Model Architecture** | Activation function | Not exposed at config level | ❌ |
| **Model Architecture** | Dropout rate | Hardcoded in model | ❌ |
| **Model Architecture** | Normalization type | Part of model design | ❌ |
| **Initialization** | Weight initialization method | TorchTitan default, not configurable | ❌ |
| **Initialization** | Bias initialization | TorchTitan default, not configurable | ❌ |
| **Initialization** | Random seed | Not exposed in config | ❌ |
| **Regularization** | Label smoothing | Not implemented in cross-entropy loss | ❌ |
| **Regularization** | Data augmentation | Dataset-specific, not in config | ❌ |
| **Loss** | Loss weights | Not used (single cross-entropy loss) | ❌ |
| **Loss** | Class weights | Not implemented | ❌ |
| **Loss** | Auxiliary losses | Not in base config | ❌ |
| **Data** | Dataset size | Not tracked in config | ❌ |
| **Data** | Sampling strategy | Dataset-specific | ❌ |
| **Data** | Train/val split | Hardcoded in dataset loaders | ❌ |
| **Data** | Preprocessing details | Dataset-specific | ❌ |
| **Data** | Augmentation | Dataset-specific | ❌ |
| **Batching** | Gradient accumulation | Derived from global_batch_size | ❌ |
| **Batching** | Padding strategy | Dataset-specific | ❌ |
| **Batching** | Dynamic batching | Not supported | ❌ |
| **Context Extension** | RoPE backend type | Available but not printed | ❌ (Partial) |
| **Context Extension** | Scaling method | Available but not printed | ❌ (Partial) |
| **Context Extension** | YaRN parameters (alpha, beta) | Available but not printed | ❌ (Partial) |

---

## Currently Printed Hyperparameters

### Optimization ✓
- `optimizer.lr` (learning rate) — Default: 8e-4
- `optimizer.betas` (from default_adamw) — Default: (0.9, 0.95)
- `optimizer.eps` — Default: 1e-8
- `optimizer.weight_decay` — Default: varies by config

### Training Schedule ✓
- `training.steps` (total training steps)
- `training.max_duration_hours` (max wall-clock time)
- `lr_scheduler.warmup_steps` — Default: 200
- `lr_scheduler.decay_ratio` — Default: 0.8
- `lr_scheduler.decay_type` — Default: "linear"
- `lr_scheduler.min_lr_factor` — Default: 0.0

### Model Architecture ✓
- Model class (AgptModel, Llama3Model, etc.)
- `n_layers` (number of layers)
- `hidden_size` (embedding dimension)
- `num_heads` (attention heads)
- `vocab_size`
- `max_seq_len`

### Batching and Data ✓
- `training.local_batch_size` — per-device batch size
- `training.global_batch_size` — computed at runtime
- `training.seq_len` — sequence length
- `dataloader.dataset` — dataset name

### Precision and System ✓
- `training.dtype` (float32, bfloat16)
- `training.mixed_precision_param` (float32, bfloat16)
- `training.mixed_precision_reduce` (float32)
- `training.enable_cpu_offload` (bool)

### Distributed Training ✓
- `parallelism.data_parallel_replicate_degree`
- `parallelism.data_parallel_shard_degree`
- `parallelism.tensor_parallel_degree`
- `parallelism.pipeline_parallel_degree`
- `parallelism.fsdp_reshard_after_forward`
- `parallelism.enable_sequence_parallel`
- `parallelism.enable_async_tensor_parallel`

### Regularization ✓
- `training.max_norm` (gradient clipping)
- Dropout (part of model, available at runtime)

### Loss ✓
- `loss` function type (CrossEntropyLoss)
- `training.enable_loss_std_termination` (early stopping)
- `training.loss_std_threshold`
- `training.loss_std_window`

### Activation Checkpointing ✓
- Type: FullAC, SelectiveAC, MemoryBudgetAC, or None

### Validation ✓
- `validator.enable`
- `validator.freq`
- `validator.steps`

### Checkpoint/Compile ✓
- `checkpoint.enable`
- `checkpoint.interval`
- `compile.enable`
- `compile.components`

### Context Extension (Partial) ⚠️
- RoPE backend type is available in model.layers[0].attention.rope
- YaRN parameters available if used
- **NOT currently printed in launch config**

---

## How to Add Missing Hyperparameters

### Option 1: Add to run_train_torchtitan.sh Print Section

The launch configuration is printed at lines 258-297. Add a new section:

```bash
cat >&2 <<EOF
...
[OPTIMIZER SETTINGS]
  Learning Rate           = ${OPTIMIZER_LR:-8e-4}
  Optimizer Type         = ${OPTIMIZER_TYPE:-AdamW}
  Betas                  = ${OPTIMIZER_BETAS:-0.9,0.95}
...
EOF
```

Then pass these via environment or `-- ` arguments.

### Option 2: Use the hyperparameters.py Script

The included `hyperparameters.py` script can:
1. Extract all available hyperparameters from Trainer.Config
2. Display which are missing
3. Output in JSON or text format

Usage:
```bash
# Show what's not tracked
python hyperparameters.py --missing-only

# Show all available parameters (when Trainer.Config loaded)
python hyperparameters.py --format json

# Integrate into your training startup
python hyperparameters.py >> training_hyperparams.log
```

### Option 3: Custom Model Metadata

For context extension techniques and custom model info, extend config_registry.py:

```python
def agpt_2b_yarn() -> Trainer.Config:
    """AGPT 2B with YaRN context extension."""
    cfg = agpt("2b", ...)
    cfg.model_metadata = {
        "base_model": "Llama3",
        "original_model": "AGPT-2B",
        "context_extension": "YaRN",
        "yarn_alpha": 32,
        "yarn_beta": 1,
    }
    return cfg
```

---

## Implementation Summary

### Files Created
1. **`hyperparameters.py`** — Script to collect and report all hyperparameters
   - Extracts all available config fields
   - Detects RoPE backend and YaRN parameters
   - Shows missing hyperparameters
   - Outputs as text or JSON

### What the Script Does

```bash
# Show missing hyperparameters and tracking status
python hyperparameters.py --missing-only

# Output shows:
# ✓ = Currently tracked/printed
# ✗ = NOT tracked
```

### Integration into Training

To integrate comprehensive hyperparameter printing:

```bash
# Before training starts
python hyperparameters.py --missing-only > hyperparams_status.txt

# During training, TorchTitan prints full config on rank 0
# Check LOG_DIR/rank_0.log for complete settings

# After training, collect metrics and hyperparams
LOG_DIR=outputs/run_20241001 python hyperparameters.py --format json > $LOG_DIR/hyperparams.json
```

---

## Current Model Information

### From Your Command

```bash
MODEL=/lus/flare/projects/datascience/seonghapark/agpt-2b-v2-256n-step-92859-safetensors
MODULE=agpt_yarn
CONFIG=agpt_2b_yarn
```

This resolves to:
- **Model class**: `AgptModel` (extends Llama3Model)
- **Base model**: AGPT-2B
- **Original model**: Llama3
- **Context extension**: Check if "yarn" is in config → likely **YaRN**
- **Model size**: 2B parameters

### Not Currently Printed

To track the above, you need:
```bash
# Option 1: Add to config_registry.py function
def agpt_2b_yarn() -> Trainer.Config:
    cfg = agpt("2b", ...)
    # Metadata not currently stored in Config dataclass
    return cfg

# Option 2: Track in training log
echo "base_model=Llama3" >> $LOG_DIR/metadata.txt
echo "current_model=AGPT-2B" >> $LOG_DIR/metadata.txt
echo "context_extension=YaRN" >> $LOG_DIR/metadata.txt
```

---

## Recommendations

1. **Short-term**: Use `hyperparameters.py --missing-only` to document current gaps
2. **Medium-term**: Add a metadata section to run_train_torchtitan.sh that prints current and original model names
3. **Long-term**: Extend Trainer.Config to include optional metadata field for model provenance and context extension info

The script `hyperparameters.py` is ready to use now and can be integrated into your training pipeline.
