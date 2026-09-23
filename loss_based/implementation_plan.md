# Implementation Plan: Loss-Based Early Termination for TorchTitan

## Objective
Add an early termination option to TorchTitan that stops training when the standard deviation of the past 50 losses is equal to or lower than 0.001, indicating convergence.

## Background
- **Current System**: Training runs for a fixed number of steps defined by `training.steps` config
- **Loss Tracking**: Losses are computed in `train_step()` method and logged via metrics processor
- **Loss Format**: Scalar values are reduced and stored as `global_avg_loss`

## Implementation Overview

### Phase 1: Configuration Extension
**Files to Modify**: 
- `xpu_launcher/xpu_torchtitan/torchtitan_repo/torchtitan/config/configs.py`

**Changes**:
1. Extend `TrainingConfig` dataclass to add two new fields:
   - `enable_loss_std_termination`: bool (default: False)
   - `loss_std_threshold`: float (default: 0.001)
   - `loss_std_window`: int (default: 50, number of steps to track)

### Phase 2: Loss History Tracking
**Files to Modify**:
- `xpu_launcher/xpu_torchtitan/torchtitan_repo/torchtitan/trainer.py`

**Changes**:
1. Add instance variables to `Trainer.__init__()`:
   - `loss_history`: deque or list to store past N losses
   - `loss_std_convergence_step`: track when convergence occurred

2. Modify `train_step()` method:
   - After loss computation and logging, append `global_avg_loss` to loss history
   - Keep history window fixed at configured size using deque

### Phase 3: Convergence Check
**Files to Modify**:
- `xpu_launcher/xpu_torchtitan/torchtitan_repo/torchtitan/trainer.py`

**Changes**:
1. Add method `should_terminate_due_to_convergence()`:
   - Compute standard deviation of loss history
   - Return True if std <= threshold AND we have enough history samples
   - Only check if feature is enabled

2. Modify `should_continue_training()` method:
   - Add convergence check alongside step count check
   - Log termination reason (step limit vs convergence)

### Phase 4: Shell Script Integration
**Files to Modify**:
- `xpu_launcher/xpu_torchtitan/run_train_torchtitan.sh`

**Changes**:
1. Add environment variable documentation:
   - `LOSS_STD_TERMINATION_ENABLED` (default: 0)
   - `LOSS_STD_THRESHOLD` (default: 0.001)
   - `LOSS_STD_WINDOW` (default: 50)

2. Add shell variables and pass to training command via `--training.enable_loss_std_termination` etc.

### Phase 5: Logging and Metrics
**Files to Modify**:
- `xpu_launcher/xpu_torchtitan/torchtitan_repo/torchtitan/trainer.py`

**Changes**:
1. Log loss statistics periodically:
   - Current loss
   - Standard deviation of loss window
   - Number of samples in window

2. Log termination decision when converged

## Testing Strategy

### Unit Tests
1. Test loss std calculation with mock loss values
2. Test deque window behavior (size, overflow)
3. Test convergence detection with synthetic loss sequences

### Integration Tests
1. Small training run with enabled feature
2. Verify termination happens at correct step
3. Verify loss history is correctly maintained
4. Test disabling feature (should run full steps)

### Test Cases
1. **Convergence Case**: Loss becomes very stable → should terminate early
2. **Non-Convergence Case**: Loss continues oscillating → should complete all steps
3. **Warm-up Phase**: High variance at start → should not trigger early termination
4. **Edge Cases**: 
   - Less than window size samples collected
   - NaN or inf losses
   - Disabled by default

## Implementation Order
1. Add config fields (TrainingConfig)
2. Add loss history tracking in Trainer
3. Implement convergence check logic
4. Update shell script arguments
5. Add logging
6. Write and run tests

## Files Summary
| File | Changes | Type |
|------|---------|------|
| configs.py | Add 3 new config fields | Config |
| trainer.py | Add loss tracking, convergence check | Core Logic |
| run_train_torchtitan.sh | Document & handle env vars | Shell |
| Tests | New test files | Test |

## Compatibility Notes
- **Backward Compatible**: Feature disabled by default
- **No Breaking Changes**: Existing configs work unchanged
- **Only affects training termination logic**: No model/data changes
