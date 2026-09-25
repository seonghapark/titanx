# 5-Hour Checkpoint Feature - Quick Reference

## What Changed
Modified TorchTitan trainer to automatically stop and save a checkpoint after 5 hours of training.

## Files Modified
- `torchtitan_repo/torchtitan/trainer.py` (5 changes)

## Changes at a Glance

| Line | Change | Purpose |
|------|--------|---------|
| 227 | Add `training_start_time: float \| None` field | Track when training starts |
| 507 | Initialize `self.training_start_time = None` | Initialization in __init__ |
| 907 | Set `self.training_start_time = time.time()` | Record actual start time |
| 929-939 | Check elapsed time before saving checkpoint | Mark 5-hour checkpoint as final |
| 975-986 | Check 5-hour elapsed in `should_continue_training()` | Stop training at 5 hours |

## How It Works

```
Training Starts (time recorded)
    ↓
Training Loop
    ├─ Each step: check if 5 hours elapsed?
    └─ If yes: stop & save final checkpoint
         (if no: continue until 5 hours or other stop condition)
    ↓
Training Ends
```

## Usage (No Changes Required)

```bash
# Run as normal - automatic 5-hour limit applies
MODEL=/path/to/model ./run_train_torchtitan.sh single
```

## Key Features

✅ **Automatic** - No configuration needed  
✅ **Safe** - Works in distributed training  
✅ **Logged** - Clear messages when limit reached  
✅ **Compatible** - Doesn't break existing features  
✅ **Efficient** - Minimal code overhead  

## Testing

```bash
# The feature works if:
1. Training runs normally for up to 5 hours
2. Logs: "Training duration reached X.XX hours (5 hour limit)"
3. Checkpoint is saved with last_step=True
4. Training process exits cleanly
```

## Notes

- Duration is measured in wall-clock time (5 × 3600 = 18,000 seconds)
- Other stopping conditions (steps, loss convergence) still apply
- All ranks stop simultaneously in distributed training
- Only rank 0 logs the 5-hour message to avoid duplication

---
For detailed information, see: `TRAINING_5HOUR_CHECKPOINT_SUMMARY.md`
