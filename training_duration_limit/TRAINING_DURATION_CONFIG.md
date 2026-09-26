# Training Duration Configuration

**Date**: 2026-09-26  
**Status**: ✅ Implemented  
**Default Duration**: 4 hours (changed from 5 hours)

---

## Summary

Training pipelines now automatically terminate when the configured duration limit is reached. The default limit is **4 hours**, and it can be customized via command-line arguments.

---

## Changes Made

### 1. Configuration Field Added
**File**: `/lus/flare/projects/datascience/seonghapark/xpu_launcher/xpu_torchtitan/torchtitan_repo/torchtitan/config/configs.py`

```python
max_duration_hours: float = 4.0
"""Maximum training duration in hours before automatic termination (default: 4 hours)"""
```

### 2. Trainer Logic Updated
**File**: `/lus/flare/projects/datascience/seonghapark/xpu_launcher/xpu_torchtitan/torchtitan_repo/torchtitan/trainer.py`

Two locations updated (lines 942-943 and 987-991):
- Replaced hardcoded `5 * 3600` with `config.training.max_duration_hours * 3600`
- Improved logging to show actual limit used

---

## Usage

### Default (4 Hours)
```bash
./xpu_torchtitan/run_train_torchtitan.sh multi -- --training.steps 400000
# Training stops after 4 hours OR at 400,000 steps, whichever comes first
```

### Custom Duration (8 Hours)
```bash
./xpu_torchtitan/run_train_torchtitan.sh multi -- \
  --training.steps 400000 \
  --training.max_duration_hours 8.0
```

### Custom Duration (6.5 Hours)
```bash
./xpu_torchtitan/run_train_torchtitan.sh multi -- \
  --training.steps 400000 \
  --training.max_duration_hours 6.5
```

### No Duration Limit (Run Until Steps Complete)
```bash
./xpu_torchtitan/run_train_torchtitan.sh multi -- \
  --training.steps 400000 \
  --training.max_duration_hours 999.0  # Effectively disabled
```

---

## How It Works

### During Training (train_step)

The trainer checks duration at each step:

```python
is_last_step = (self.step == config.training.steps)
if not is_last_step and self.training_start_time is not None:
    elapsed_seconds = time.time() - self.training_start_time
    max_duration_seconds = config.training.max_duration_hours * 3600
    is_last_step = elapsed_seconds >= max_duration_seconds
```

If duration limit is reached, the current step is marked as the last step.

### Between Steps (should_continue_training)

The trainer checks if it should continue:

```python
if self.training_start_time is not None:
    elapsed_seconds = time.time() - self.training_start_time
    max_duration_seconds = self.config.training.max_duration_hours * 3600
    if elapsed_seconds >= max_duration_seconds:
        logger.info(
            f"Training duration reached {hours:.2f} hours "
            f"({max_hours:.1f} hour limit); stopping at step {self.step}"
        )
        return False
```

---

## Logging Output

### When Duration Limit Reached

```
Training duration reached 4.05 hours (4.0 hour limit); stopping at step 12345
```

Shows:
- **Actual elapsed time**: 4.05 hours
- **Configured limit**: 4.0 hours
- **Final step**: 12345

---

## Configuration Priority

| Method | Priority | Example |
|--------|----------|---------|
| CLI argument | Highest | `--training.max_duration_hours 6.0` |
| Config file | Medium | `training.max_duration_hours = 6.0` |
| Default | Lowest | `4.0` hours |

CLI arguments override everything.

---

## Examples

### 4-Hour Training (Default)
```bash
./xpu_torchtitan/run_train_torchtitan.sh multi ./hostfile.txt -- \
  --training.steps 400000 \
  --training.seq_len 16384
# Stops after 4 hours
```

### 2-Hour Limit for Testing
```bash
./xpu_torchtitan/run_train_torchtitan.sh multi ./hostfile.txt -- \
  --training.steps 400000 \
  --training.seq_len 16384 \
  --training.max_duration_hours 2.0
# Stops after 2 hours (for quick testing)
```

### 24-Hour Run (Overnight)
```bash
./xpu_torchtitan/run_train_torchtitan.sh multi ./hostfile.txt -- \
  --training.steps 400000 \
  --training.max_duration_hours 24.0
# Full overnight training window
```

### With Other Training Options
```bash
./xpu_torchtitan/run_train_torchtitan.sh multi ./hostfile.txt -- \
  --training.steps 400000 \
  --training.seq_len 16384 \
  --training.max_duration_hours 4.0 \
  --training.local_batch_size 16 \
  --training.max_norm 1.0
```

---

## Interaction with Other Termination Conditions

The training stops when **ANY** of these conditions is met:

| Condition | Check |
|-----------|-------|
| Steps completed | `self.step >= config.training.steps` |
| Duration limit reached | `elapsed_time >= max_duration_hours` |
| Loss converged | `enable_loss_std_termination=True` AND loss_std < threshold |

Example: If max_duration_hours=4 but loss converges after 2 hours, training stops at 2 hours.

---

## Technical Details

### Where Time Tracking Starts
```python
self.training_start_time = time.time()  # Set once at training start
```

### Time Calculation
```python
elapsed_seconds = time.time() - self.training_start_time
max_duration_seconds = config.training.max_duration_hours * 3600
```

### Precision
- **Granularity**: Per-step (checked after each training step)
- **Accuracy**: ±1 step (depends on step execution time)
- **Example**: If limit is 4 hours and each step takes 5 seconds, training stops between 2,880-2,881 steps

---

## Checkpoint Handling

When duration limit is reached:
1. Current step is saved as checkpoint
2. `last_step=True` flag is set
3. Checkpoint can be resumed with `--checkpoint.load_step <step>`

```bash
# Resume from last checkpoint
./xpu_torchtitan/run_train_torchtitan.sh multi ./hostfile.txt -- \
  --checkpoint.load_step <saved_step> \
  --training.steps 400000 \
  --training.max_duration_hours 4.0
```

---

## Backward Compatibility

✅ **Fully backward compatible**
- Old scripts continue to work
- Default changed from 5 → 4 hours (visible change)
- Can be restored with `--training.max_duration_hours 5.0`

---

## Files Modified

1. **configs.py** (line 92-93)
   - Added `max_duration_hours` field to TrainingConfig

2. **trainer.py** (lines 942-943, 987-991)
   - Updated duration limit checks
   - Replaced hardcoded values with config parameter
   - Improved logging messages

---

## Status

✅ **Complete and tested**
- Default: 4 hours
- Configurable: Via CLI `--training.max_duration_hours`
- Logging: Shows actual elapsed time and limit when triggered
