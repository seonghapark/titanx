# Training with Model Format Conversion - Usage Guide

## Overview

The `start_training_with_conversion.sh` script now supports running multi-node training **without requiring an explicit hostfile**. It automatically uses `PBS_NODEFILE` when running inside a PBS job environment.

## Key Changes

### Before
```bash
# Had to provide explicit hostfile
./start_training_with_conversion.sh \
  --model-path /path/to/checkpoint \
  --mode multi \
  --hostfile /path/to/hostfile
```

### After
```bash
# No hostfile needed - automatically uses PBS_NODEFILE
./start_training_with_conversion.sh \
  --model-path /path/to/checkpoint \
  --mode multi
```

---

## Usage Examples

### 1. Single-Node Training (from command line)

```bash
cd /lus/flare/projects/datascience/seonghapark/xpu_launcher

# Train from checkpoint with model conversion
./start_training_with_conversion.sh \
  --model-path /lus/flare/projects/datascience/seonghapark/xpu_launcher/xpu_torchtitan/torchtitan_repo/outputs/xpu_torchtitan_20260925_160731/checkpoint/step-2062 \
  --mode single
```

### 2. Multi-Node Training in PBS Job (Recommended)

```bash
# Inside a PBS job, no hostfile needed
./start_training_with_conversion.sh \
  --model-path /lus/flare/projects/datascience/seonghapark/xpu_launcher/xpu_torchtitan/torchtitan_repo/outputs/xpu_torchtitan_20260925_160731/checkpoint/step-2062 \
  --mode multi
```

**Why this works**: When you submit a PBS job, the `PBS_NODEFILE` environment variable is automatically set and contains the list of allocated nodes.

### 3. Multi-Node with Explicit Hostfile (Alternative)

If you prefer to specify a hostfile explicitly:

```bash
./start_training_with_conversion.sh \
  --model-path /path/to/checkpoint \
  --mode multi \
  --hostfile /path/to/hostfile.txt
```

### 4. Full PBS Job Script

Create a file `train_job.pbs`:

```bash
#!/bin/bash
#PBS -l select=9:ngpus=1
#PBS -l walltime=12:00:00
#PBS -N train_agpt2b

set -euo pipefail

cd /lus/flare/projects/datascience/seonghapark/xpu_launcher

./start_training_with_conversion.sh \
  --model-path /lus/flare/projects/datascience/seonghapark/xpu_launcher/xpu_torchtitan/torchtitan_repo/outputs/xpu_torchtitan_20260925_160731/checkpoint/step-2062 \
  --mode multi \
  -- --training.steps 400000 --training.seq_len 16384
```

Submit with:
```bash
qsub train_job.pbs
```

### 5. Dry-Run (Preview without executing)

```bash
./start_training_with_conversion.sh \
  --model-path /path/to/checkpoint \
  --mode multi \
  --dry-run --verbose
```

Output shows:
```
[DRY RUN] Model already in HF format, would execute:
/lus/flare/projects/datascience/seonghapark/xpu_launcher/start_training.sh --model-path ...
[DRY RUN] Multi-mode without explicit hostfile: will use PBS_NODEFILE if available
```

---

## Command-Line Options

### Core Options

| Option | Argument | Description |
|--------|----------|-------------|
| `--model-path` | PATH | **Required**. Path to model checkpoint (distcp or HF format) |
| `--mode` | single\|multi | Training mode (default: single) |
| `--hostfile` | FILE | Hostfile for multi-node (optional if PBS_NODEFILE available) |

### Conversion Options

| Option | Description |
|--------|-------------|
| `--output-dir DIR` | Where to save converted models (default: ./converted_models) |
| `--no-auto-convert` | Don't auto-convert, fail if distcp format |
| `--delete-converted` | Delete converted model after training completes |

### Execution Options

| Option | Description |
|--------|-------------|
| `--dry-run` | Show what would be executed without running |
| `--verbose` | Verbose output during conversion and execution |
| `--help` | Show help message |

### Training Options (passed to training script)

These are passed through to the underlying training script:

```bash
./start_training_with_conversion.sh \
  --model-path /path/to/model \
  --mode multi \
  -- --training.steps 400000 --training.seq_len 16384
```

---

## Environment Variables

### Auto-Detected

| Variable | Purpose |
|----------|---------|
| `PBS_NODEFILE` | PBS job node list (auto-used for multi-node if --hostfile not provided) |

### Optional Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PYTHON_BIN` | python3 | Python binary to use |
| `CONVERT_OUTPUT_DIR` | ./converted_models | Where to save converted models |
| `AUTO_CONVERT` | 1 | Enable automatic format conversion (0 to disable) |
| `KEEP_CONVERTED` | 1 | Keep converted model after training (0 to delete) |

### Training-Related

| Variable | Default | Description |
|----------|---------|-------------|
| `MODULE` | llama3 | TorchTitan module config |
| `CONFIG` | llama3_debugmodel | Config callable |
| `DATASET_NAME` | pg19_multinews | Dataset to use |
| `TRAINING_STEPS` | 100 | Number of training steps |
| `SEQ_LEN` | 16384 | Sequence length |

---

## Common Use Cases

### Use Case 1: Resume Training from Checkpoint

```bash
./start_training_with_conversion.sh \
  --model-path /lus/flare/projects/datascience/seonghapark/xpu_launcher/xpu_torchtitan/torchtitan_repo/outputs/xpu_torchtitan_20260925_160731/checkpoint/step-2062 \
  --mode multi \
  -- --training.steps 400000 --training.seq_len 16384
```

**What happens**:
1. Detects checkpoint is in HF format (no conversion needed)
2. Starts multi-node training using PBS_NODEFILE
3. Resumes from step 2062, continues for 400,000 total steps

### Use Case 2: Convert and Train in One Step

```bash
./start_training_with_conversion.sh \
  --model-path /path/to/agpt-2b-v2-256n-distcp \
  --mode multi \
  --output-dir /lus/flare/projects/datascience/seonghapark/converted_models \
  -- --training.steps 400000
```

**What happens**:
1. Detects model is in distcp format
2. Automatically converts to HF format in specified output directory
3. Starts multi-node training with converted model
4. Keeps converted model for future training runs (can be deleted with `--delete-converted`)

### Use Case 3: Test Conversion Without Training

```bash
./start_training_with_conversion.sh \
  --model-path /path/to/model \
  --dry-run --verbose
```

**Output shows**:
```
Model Path:    /path/to/model
Auto-Convert:  1
[INFO] Detecting model format...
[DRY RUN] Would execute:
python3 convert_model_format.py --dcp "/path/to/model" --output "/lus/flare/projects/datascience/seonghapark/xpu_launcher/converted_models/model_hf"
```

---

## Troubleshooting

### Error: "Multi-mode requires either --hostfile or PBS_NODEFILE"

**Cause**: You specified `--mode multi` but:
- Did not provide `--hostfile`
- Are not running inside a PBS job (PBS_NODEFILE not set)

**Solution**: Either:
```bash
# Option 1: Provide hostfile
./start_training_with_conversion.sh --model-path ... --mode multi --hostfile /path/to/hostfile

# Option 2: Submit as PBS job
qsub -I -l select=9:ngpus=1 -N training_session
# Then run inside the job:
./start_training_with_conversion.sh --model-path ... --mode multi
```

### Error: "Model path does not exist"

**Solution**: Use absolute paths:
```bash
cd /lus/flare/projects/datascience/seonghapark/xpu_launcher

# Get absolute path first
MODEL_PATH="/lus/flare/projects/datascience/seonghapark/xpu_launcher/xpu_torchtitan/torchtitan_repo/outputs/xpu_torchtitan_20260925_160731/checkpoint/step-2062"

./start_training_with_conversion.sh --model-path "$MODEL_PATH" --mode multi
```

### Conversion Takes Too Long

The conversion runs on the launch node. For faster conversion, you can:

1. **Pre-convert the model**:
```bash
python3 convert_model_format.py --dcp /path/to/model --output /path/to/converted

# Then use converted model for multiple training runs
./start_training_with_conversion.sh --model-path /path/to/converted --mode multi
```

2. **Disable conversion and use HF format directly**:
```bash
./start_training_with_conversion.sh --model-path /path/to/hf/model --no-auto-convert --mode multi
```

---

## Advanced: Multi-Step Workflow

Example workflow for resuming long training:

```bash
#!/bin/bash
set -euo pipefail

CHECKPOINT_DIR="/lus/flare/projects/datascience/seonghapark/xpu_launcher/xpu_torchtitan/torchtitan_repo/outputs/xpu_torchtitan_20260925_160731/checkpoint"

# Find the latest checkpoint
LATEST_CHECKPOINT=$(ls -t "$CHECKPOINT_DIR"/step-* 2>/dev/null | head -1)

if [[ -z "$LATEST_CHECKPOINT" ]]; then
    echo "No checkpoint found!"
    exit 1
fi

echo "Resuming from: $LATEST_CHECKPOINT"

cd /lus/flare/projects/datascience/seonghapark/xpu_launcher

# Submit training job
qsub -v CHECKPOINT="$LATEST_CHECKPOINT" -N resumed_training << 'EOF'
#!/bin/bash
#PBS -l select=9:ngpus=1
#PBS -l walltime=12:00:00

./start_training_with_conversion.sh \
  --model-path "$CHECKPOINT" \
  --mode multi \
  -- --training.steps 400000 --training.seq_len 16384
EOF
```

---

## Key Differences from Original

| Aspect | Before | After |
|--------|--------|-------|
| Requires hostfile in multi-mode | ✅ Yes | ❌ No (optional) |
| PBS_NODEFILE support | ❌ No | ✅ Yes |
| Single command for PBS jobs | ❌ No | ✅ Yes |
| Works like `run_train_torchtitan.sh` | ❌ No | ✅ Yes |

---

## Related Scripts

- `./start_training.sh` — Base training launcher (called by conversion script)
- `./xpu_torchtitan/run_train_torchtitan.sh` — Direct training without conversion
- `./convert_model_format.py` — Model format conversion utility

---

**Last Updated**: 2026-09-26
