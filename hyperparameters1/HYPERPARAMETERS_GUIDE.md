# Hyperparameters: What Gets Printed and How to Change Them

## What Gets Printed

### From run_train_torchtitan.sh (BEFORE training starts)

The script prints a **TRAINING LAUNCH CONFIGURATION** with these sections:

#### [LAUNCH MODE]
```
Mode                   = multi (hostfile=auto from PBS_NODEFILE)
Topology               = NNODES=150 NPROC_PER_NODE=12 NPROC=1800
Auto Retry             = 1
Spare Nodes            = 50
```
✅ **Includes node/spare info**

#### [MODEL & DATASET]
```
Model Path             = /path/to/model
Module/Config          = agpt_yarn / agpt_2b_yarn
Dataset Name           = PG19
Dataset Path           = /path/to/data
HF Assets Path         = /path/to/model
```

#### [TRAINING HYPERPARAMETERS]
```
Training Steps         = 400
Sequence Length        = 16384
Loss Std Termination   = 1 (threshold: 0.001, window: 50)
```
✅ **Only training.steps, seq_len, loss_std settings** (from `run_train_torchtitan.sh`)

❌ **NOT in this section**: learning rate, batch size, optimizer (these come from the config registry)

#### [SYSTEM & PATHS]
- TorchTitan Root, Log Directory, Checkpoint Folder, Python Binary

#### [MONITORING]
- Resource Monitor settings

#### [EXTRA ARGUMENTS]
- Any `--` arguments you pass

### From TorchTitan (DURING training)

Once training starts, TorchTitan prints the actual config including:
- Learning rate
- Optimizer settings (AdamW parameters)
- Local batch size
- Global batch size
- Gradient accumulation steps
- And many more training metrics

## How to Change Learning Rate and Optimizer

### 1. **Learning Rate** (Easiest)

**Default from config registry:**
```python
optimizer=default_adamw(lr=8e-4)  # 0.0008
```

**Override via command line:**
```bash
./xpu_torchtitan/run_train_torchtitan.sh multi -- \
  --training.steps 400 \
  --optimizer.lr 1e-3
```

Or in your command:
```bash
LOSS_STD_TERMINATION_ENABLED=1 \
./xpu_torchtitan/run_train_torchtitan.sh multi -- \
  --optimizer.lr 0.001 \
  --training.steps 400 \
  --training.seq_len 16384
```

### 2. **Optimizer Type and Settings**

**Default:** `AdamW` with default settings from `default_adamw()`

**Available optimizers in TorchTitan:**
- `AdamW` (default, from `torchtitan.components.optimizer.default_adamw`)
- Others defined in the optimizer component

**Override AdamW parameters via CLI:**
```bash
./xpu_torchtitan/run_train_torchtitan.sh multi -- \
  --optimizer.lr 1e-3 \
  --optimizer.betas 0.9,0.95 \
  --optimizer.eps 1e-8 \
  --optimizer.weight_decay 0.01
```

### 3. **Batch Size**

**Default from config registry:**
```python
training=TrainingConfig(
    local_batch_size=8,  # per device per node
    seq_len=2048,
    steps=10000,
),
```

**Global batch size:** = local_batch_size × data_parallel_degree

**Override local batch size:**
```bash
./xpu_torchtitan/run_train_torchtitan.sh multi -- \
  --training.local_batch_size 4 \
  --training.steps 400
```

Or set via env var in `run_train_torchtitan.sh` (would need modification):
```bash
# Currently not supported via env var in the wrapper script
# Would need to add this to run_train_torchtitan.sh:
# TRAIN_CMD+=("--training.local_batch_size" "${LOCAL_BATCH_SIZE:-8}")
```

### 4. **Learning Rate Scheduler**

**Default:**
```python
lr_scheduler=LRSchedulersContainer.Config(
    warmup_steps=200,
    decay_ratio=0.8,
    decay_type="linear",
    min_lr_factor=0.0,
),
```

**Override via CLI:**
```bash
./xpu_torchtitan/run_train_torchtitan.sh multi -- \
  --lr_scheduler.warmup_steps 100 \
  --lr_scheduler.decay_ratio 0.5 \
  --lr_scheduler.decay_type linear \
  --training.steps 400
```

## Where These Settings Come From

### Hierarchy (in order of precedence):

1. **Command-line args** (highest priority)
   ```bash
   -- --optimizer.lr 1e-3 --training.local_batch_size 4
   ```

2. **Environment variables** (set in wrapper script)
   ```bash
   TRAINING_STEPS=400 SEQ_LEN=16384
   ```

3. **Config registry function** (from MODULE/CONFIG)
   ```python
   # E.g., agpt_2b_yarn() or agpt_2b()
   # Located in torchtitan/models/agpt/config_registry.py
   ```

4. **Base config** (in _base_config())
   ```python
   # Default lr=8e-4, local_batch_size=8, etc.
   ```

## Example: Changing Multiple Hyperparameters

```bash
MODEL=/lus/flare/projects/datascience/seonghapark/agpt-2b-v2-256n-step-92859-safetensors \
MODULE=agpt_yarn \
CONFIG=agpt_2b_yarn \
DATASET_NAME=PG19 \
DATASET_PATH=/lus/flare/projects/datascience/seonghapark/assets/hf/datasets/pg19/data \
TRAINING_STEPS=400 \
SEQ_LEN=16384 \
NNODES=150 \
SPARE_NODES=50 \
./xpu_torchtitan/run_train_torchtitan.sh multi \
-- \
--optimizer.lr 5e-4 \
--optimizer.weight_decay 0.05 \
--training.local_batch_size 4 \
--training.steps 400 \
--training.seq_len 16384 \
--lr_scheduler.warmup_steps 150 \
--training.max_duration_hours 4.0
```

## Configuration Files to Understand

1. **Config Registry** (where defaults are defined):
   ```
   torchtitan/models/agpt/config_registry.py
   ```
   - Functions like `agpt_2b()`, `agpt_2b_yarn()`
   - Each returns a `Trainer.Config` with model, optimizer, training settings

2. **Trainer Config** (schema):
   ```
   torchtitan/trainer.py → Trainer.Config dataclass
   ```
   - Defines all available config fields
   - optimizer, training, lr_scheduler, etc.

3. **Optimizer Component**:
   ```
   torchtitan/components/optimizer.py
   ```
   - `default_adamw()` function
   - Defines default AdamW parameters

## What If I Need Different Defaults?

**Option 1: Create a new config function** (best practice)
```python
# In torchtitan/models/agpt/config_registry.py
def agpt_2b_custom_lr() -> Trainer.Config:
    cfg = agpt_2b()
    cfg.optimizer = default_adamw(lr=5e-4)  # Custom LR
    cfg.training.local_batch_size = 4
    return cfg
```

Then use:
```bash
MODULE=agpt CONFIG=agpt_2b_custom_lr ./run_train_torchtitan.sh multi -- ...
```

**Option 2: Override all at once via CLI** (easier for one-off runs)
```bash
./run_train_torchtitan.sh multi -- \
  --optimizer.lr 5e-4 \
  --training.local_batch_size 4 \
  --training.steps 400
```

## Viewing Full Config at Runtime

TorchTitan prints the full resolved config at rank 0 startup:
```
[rank 0] Configuration:
optimizer: AdamW(lr=0.0008, ...)
training: TrainingConfig(steps=400, local_batch_size=8, ...)
...
```

Check your `LOG_DIR/rank_0.log` or job output for this.

## Summary Table

| Parameter | Where | How to Change |
|-----------|-------|---------------|
| Learning Rate | optimizer | `--optimizer.lr 1e-3` |
| Optimizer Type | config registry | Modify config function |
| Local Batch Size | training | `--training.local_batch_size 4` |
| Training Steps | run_train_torchtitan.sh env | `TRAINING_STEPS=100` or `--training.steps 100` |
| Sequence Length | run_train_torchtitan.sh env | `SEQ_LEN=4096` or `--training.seq_len 4096` |
| Warmup Steps | lr_scheduler | `--lr_scheduler.warmup_steps 100` |
| Nodes/Spare | run_train.sh env | `NNODES=150 SPARE_NODES=50` |

✅ = Printed in launch config
🔧 = Requires command-line override or config modification
