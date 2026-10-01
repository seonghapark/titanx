# Script Update Summary: `start_training_with_conversion.sh`

**Date**: 2026-09-26  
**Status**: ✅ Complete

---

## What Changed

The `start_training_with_conversion.sh` script has been updated to **support running without an explicit hostfile** in multi-node mode, matching the behavior of `./xpu_torchtitan/run_train_torchtitan.sh`.

### Key Improvements

1. **Auto-detect PBS environment**: When running in multi-node mode without `--hostfile`, the script automatically uses `PBS_NODEFILE` (set by PBS when you submit a job).

2. **Simplified PBS job submission**: No need to manually specify hostfile in PBS job scripts.

3. **Better documentation**: Updated help text and usage examples show the new capability.

4. **Validation**: Script validates that multi-mode has either `--hostfile` or `PBS_NODEFILE` available.

---

## Before vs After

### Before
```bash
# Multi-node required explicit hostfile
./start_training_with_conversion.sh \
  --model-path /path/to/checkpoint \
  --mode multi \
  --hostfile /path/to/hostfile.txt
```

### After
```bash
# Option 1: In PBS job (PBS_NODEFILE auto-used)
./start_training_with_conversion.sh \
  --model-path /path/to/checkpoint \
  --mode multi

# Option 2: With explicit hostfile (still works)
./start_training_with_conversion.sh \
  --model-path /path/to/checkpoint \
  --mode multi \
  --hostfile /path/to/hostfile.txt
```

---

## Modified Code Sections

### 1. **Help/Usage Text** (Lines 31-86)
- Added `PBS_NODEFILE` to environment variables section
- Updated examples to show multi-node without hostfile
- Clarified that hostfile is optional in multi-mode

### 2. **Argument Parsing** (Lines 88-129)
- Added `HOSTFILE=""` variable initialization
- Added `--hostfile` option handler to properly capture and pass it through
- Preserves hostfile in training arguments when provided

### 3. **Dry-Run Output** (Lines 188-197 and 220-226)
- Added check to display when PBS_NODEFILE will be used
- Shows informational message about automatic hostfile detection

### 4. **Pre-Training Validation** (Lines 228-243)
- Added validation to ensure multi-mode has either `--hostfile` or `PBS_NODEFILE`
- Provides clear error message if neither is available
- Shows which method is being used (explicit hostfile or PBS_NODEFILE)

---

## Testing

### Dry-Run Test ✅
```bash
cd /lus/flare/projects/datascience/seonghapark/xpu_launcher

./start_training_with_conversion.sh \
  --model-path /lus/flare/projects/datascience/seonghapark/xpu_launcher/xpu_torchtitan/torchtitan_repo/outputs/xpu_torchtitan_20260925_160731/checkpoint/step-2062 \
  --mode multi \
  --dry-run --verbose
```

**Output**:
```
[INFO] Detecting model format...
[DRY RUN] Model already in HF format, would execute:
/lus/flare/projects/datascience/seonghapark/xpu_launcher/start_training.sh --model-path "..." --mode multi

[DRY RUN] Multi-mode without explicit hostfile: will use PBS_NODEFILE if available
```

---

## How to Use

### Method 1: Submit PBS Job (Recommended)

Create `train_job.pbs`:
```bash
#!/bin/bash
#PBS -l select=9:ngpus=1
#PBS -l walltime=12:00:00
#PBS -N train_agpt2b

cd /lus/flare/projects/datascience/seonghapark/xpu_launcher

./start_training_with_conversion.sh \
  --model-path /lus/flare/projects/datascience/seonghapark/xpu_launcher/xpu_torchtitan/torchtitan_repo/outputs/xpu_torchtitan_20260925_160731/checkpoint/step-2062 \
  --mode multi \
  -- --training.steps 400000 --training.seq_len 16384
```

Submit:
```bash
qsub train_job.pbs
```

### Method 2: Interactive PBS Session

```bash
qsub -I -l select=9:ngpus=1 -l walltime=1:00:00

cd /lus/flare/projects/datascience/seonghapark/xpu_launcher

./start_training_with_conversion.sh \
  --model-path /lus/flare/projects/datascience/seonghapark/xpu_launcher/xpu_torchtitan/torchtitan_repo/outputs/xpu_torchtitan_20260925_160731/checkpoint/step-2062 \
  --mode multi
```

### Method 3: Explicit Hostfile (Alternative)

```bash
./start_training_with_conversion.sh \
  --model-path /path/to/checkpoint \
  --mode multi \
  --hostfile /path/to/hostfile.txt
```

### Method 4: Single-Node (No Hostfile Needed)

```bash
./start_training_with_conversion.sh \
  --model-path /path/to/checkpoint \
  --mode single
```

---

## Backward Compatibility

✅ **Fully backward compatible**

- Existing scripts with `--hostfile` continue to work unchanged
- Single-node training unaffected
- All existing options and environment variables still work

---

## Files Modified

- `/lus/flare/projects/datascience/seonghapark/xpu_launcher/start_training_with_conversion.sh`

## Files Created (Documentation)

- `/lus/flare/projects/datascience/seonghapark/TRAINING_WITH_CONVERSION_USAGE.md` — Detailed usage guide
- `/lus/flare/projects/datascience/seonghapark/SCRIPT_UPDATE_SUMMARY.md` — This file

---

## Quick Reference

| Scenario | Command |
|----------|---------|
| Single-node from CLI | `./start_training_with_conversion.sh --model-path /path/to/model --mode single` |
| Multi-node in PBS job | `./start_training_with_conversion.sh --model-path /path/to/model --mode multi` |
| Multi-node with hostfile | `./start_training_with_conversion.sh --model-path /path/to/model --mode multi --hostfile /path/to/hosts` |
| Dry-run to preview | `./start_training_with_conversion.sh --model-path /path/to/model --mode multi --dry-run --verbose` |
| Help | `./start_training_with_conversion.sh --help` |

---

## Next Steps

1. **For immediate use**: Submit a PBS job using the updated script (see Method 1 above)

2. **For testing**: Run a dry-run first to verify configuration:
   ```bash
   ./start_training_with_conversion.sh \
     --model-path /your/checkpoint \
     --mode multi \
     --dry-run --verbose
   ```

3. **For integration**: Update any existing job scripts to remove the `--hostfile` argument (it's now optional)

---

## Support

For detailed usage information, see:
- `/lus/flare/projects/datascience/seonghapark/TRAINING_WITH_CONVERSION_USAGE.md`

For help with the script:
```bash
./start_training_with_conversion.sh --help
```

---

**Status**: Ready for production use ✅
