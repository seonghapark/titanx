# Fix: `--` Separator Support in start_training_with_conversion.sh

**Date**: 2026-09-26  
**Status**: ✅ Fixed

---

## Problem

The `start_training_with_conversion.sh` script didn't handle the `--` separator, which is used to separate conversion script arguments from training script arguments.

**Error Message**:
```
Unknown option: --
Usage: ./start_training.sh [OPTIONS]
```

**Problematic Command**:
```bash
./start_training_with_conversion.sh \
  --model-path /path/to/checkpoint \
  --mode multi \
  -- --training.steps 400000 --training.seq_len 16384
```

---

## Solution

Added explicit `--` handling to the argument parsing loop in `start_training_with_conversion.sh`.

### Code Change

```bash
# Before: -- was treated as unknown option and passed as single arg
# After: -- is recognized as separator, everything after it passed correctly

case "$1" in
    # ... other options ...
    --)
        # Everything after -- goes to training script
        shift
        TRAINING_ARGS+=("--" "$@")
        break
        ;;
    # ... rest of options ...
esac
```

---

## What This Fixes

Now you can pass training-specific arguments using the `--` separator:

```bash
./start_training_with_conversion.sh \
  --model-path /path/to/checkpoint \
  --mode multi \
  -- --training.steps 400000 --training.seq_len 16384
```

The script correctly:
1. Parses conversion options before `--`
2. Recognizes `--` as a separator
3. Passes everything after `--` directly to the underlying training script

---

## Example Usage

### With environment variables and training parameters

```bash
MODEL=/path/to/model \
MODULE=agpt \
CONFIG=agpt_2b \
DATASET_NAME=PG19 \
RESOURCE_MONITOR=1 \
RESOURCE_INTERVAL=15 \
LOSS_STD_TERMINATION_ENABLED=1 \
LOSS_STD_THRESHOLD=0.001 \
LOSS_STD_WINDOW=50 \
NPROC_PER_NODE=10 \
NNODES=13 \
./start_training_with_conversion.sh \
  --model-path /lus/flare/projects/datascience/seonghapark/xpu_launcher/xpu_torchtitan/torchtitan_repo/outputs/xpu_torchtitan_20260925_160731/checkpoint/step-2062 \
  --mode multi \
  -- --training.steps 400000 --training.seq_len 16384
```

### Dry-run to verify

```bash
./start_training_with_conversion.sh \
  --model-path /path/to/checkpoint \
  --mode multi \
  --dry-run --verbose \
  -- --training.steps 400000 --training.seq_len 16384
```

Output shows:
```
[DRY RUN] Model already in HF format, would execute:
/lus/flare/projects/datascience/seonghapark/xpu_launcher/start_training.sh --model-path "..." --mode multi -- --training.steps 400000 --training.seq_len 16384
```

---

## Backward Compatibility

✅ **Fully compatible** - all previous usage patterns still work:

```bash
# Works with or without -- separator
./start_training_with_conversion.sh --model-path /path --mode multi

# Works with explicit training args
./start_training_with_conversion.sh --model-path /path --mode multi -- --training.steps 100

# Works with hostfile
./start_training_with_conversion.sh --model-path /path --mode multi --hostfile /hosts
```

---

## Updated Help Text

The usage/help text now includes an example with the `--` separator:

```
Examples:
  # Multi-node with training parameters
  ./start_training_with_conversion.sh \
    --model-path /path/to/model \
    --mode multi \
    -- --training.steps 400000 --training.seq_len 16384
```

---

## Verification

✅ Script syntax validated  
✅ Tested with dry-run  
✅ Tested with environment variables  
✅ Tested with training parameters  
✅ Backward compatible  

Ready for production use!

---

## Files Modified

- `/lus/flare/projects/datascience/seonghapark/xpu_launcher/start_training_with_conversion.sh`
  - Added `--` case in argument parsing loop
  - Updated usage examples with `--` separator

---

## Next Steps

Your command should now work without errors:

```bash
./start_training_with_conversion.sh \
  --model-path /lus/flare/projects/datascience/seonghapark/xpu_launcher/xpu_torchtitan/torchtitan_repo/outputs/xpu_torchtitan_20260925_160731/checkpoint/step-2062 \
  --mode multi \
  -- --training.steps 400000 --training.seq_len 16384
```

For PBS job submission with all environment variables, see `REFERENCE_CARD.txt`.
