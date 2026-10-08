# AGPT Training with TorchTitan

This guide explains how to use the new AGPT parameter calculator and wrapper scripts to train AGPT models (2B, 20B, 80B) with pre-optimized parameters.

## Quick Start

### Option 1: Using the Wrapper Script (Recommended)

The simplest way to train AGPT models:

```bash
# Single-node training
./run_train_agpt.sh --model 2b single

# Multi-node training (8 nodes)
./run_train_agpt.sh --model 2b --nnodes 8 multi /path/to/hosts.txt

# In a PBS job (auto-detects node count)
./run_train_agpt.sh --model 20b multi

# Dry-run to see the command without executing
./run_train_agpt.sh --model 2b single --dry-run
```

### Option 2: Using the Parameter Calculator Directly

For more control, you can calculate parameters and pass them to `run_train_torchtitan.sh`:

```bash
# Calculate parameters and load them
eval $(./calculate_agpt_params.sh --model 2b --nnodes 8)

# Then run training
MODEL=~/models/agpt-2b \
  MODULE=agpt \
  CONFIG=agpt_2b \
  TRAINING_STEPS=$AGPT_TRAINING_STEPS \
  SEQ_LEN=$AGPT_SEQ_LEN \
  ./run_train_torchtitan.sh multi /path/to/hosts.txt
```

## Scripts Overview

### `calculate_agpt_params.sh`

Calculates pre-optimized AGPT training parameters based on model size and node count.

**Features:**
- Supports 2B, 20B, and 80B models
- Automatically detects node count from PBS_NODEFILE (when in PBS job)
- Calculates global batch size (GBS) and training steps from token budget
- Outputs environment variables with `AGPT_` prefix
- Supports parameter overrides

**Usage:**
```bash
./calculate_agpt_params.sh --model 2b --nnodes 8 [OPTIONS]
```

**Output variables:**
```bash
AGPT_MODEL              # Model size (2b, 20b, 80b)
AGPT_NNODES             # Number of nodes
AGPT_NGPUS              # Total GPUs
AGPT_TP, AGPT_PP, AGPT_CP  # Parallelism degrees
AGPT_LBS, AGPT_GAS      # Batch size components
AGPT_GBS                # Global batch size (calculated)
AGPT_SEQ_LEN            # Sequence length
AGPT_TRAINING_STEPS     # Training steps (calculated)
AGPT_TRAIN_TOKENS       # Total token budget
AGPT_OPTIMIZER          # Optimizer (sophiag)
AGPT_LR                 # Learning rate (2.28e-5)
AGPT_DATASET            # Dataset (blendcorpus)
AGPT_DATA_LIST          # Data list name
AGPT_CKPT_DIR           # Checkpoint directory pattern
```

### `run_train_agpt.sh`

Convenient wrapper that automatically calculates parameters and launches training via `run_train_torchtitan.sh`.

**Features:**
- Simpler interface than manual parameter calculation
- Automatically detects node count in PBS jobs
- Validates model path before launching
- Works with single-node and multi-node training
- Supports hostfile-based and scheduler-based topology

**Usage:**
```bash
./run_train_agpt.sh --model {2b|20b|80b} [OPTIONS] {single|multi} [HOSTFILE]
```

## Model-Specific Defaults

Each model has pre-optimized parallelism and batch size settings:

| Model | Tensor Parallelism | Local Batch Size | Reason |
|-------|-------------------|------------------|--------|
| 2B | 1 | 2 | Memory efficient, pure data parallelism |
| 20B | 1 | 2 | Memory efficient, pure data parallelism |
| 80B | 2 | 1 | Requires TP=2 to fit in GPU memory |

All models share:
- Global token budget: 4.67 trillion tokens
- Optimizer: SophiaG (safer than AdamW at large batches)
- Learning rate: 2.28e-5 (production-tuned)
- Sequence length: 8192 tokens (default)
- Dataset: blendcorpus with olmo-mix-1124 mix

## Parameter Calculation

### How Global Batch Size is Calculated

```
GBS = (Total GPUs × Local Batch Size × Gradient Accumulation) / (TP × PP × CP)
```

Example (2B on 8 nodes):
```
GBS = (8 nodes × 12 GPUs/node × 2 LBS × 1 GAS) / (1 TP × 1 PP × 1 CP)
GBS = 96 GPUs × 2 / 1 = 192
```

### How Training Steps are Calculated

```
Training Steps = Total Tokens / (GBS × Sequence Length)
```

Example (2B on 8 nodes with 192 GBS):
```
Steps = 4.67 trillion / (192 × 8192)
Steps = 2,970,456 steps
```

This ensures the same token budget is trained regardless of node count.

## Common Usage Patterns

### Single-Node Experimentation

Start on a single GPU/node for quick prototyping:

```bash
./run_train_agpt.sh --model 2b single
```

This will:
- Calculate parameters for 1 node (12 GPUs on Aurora)
- GBS = 12 × 2 / 1 = 24
- Training steps = 4.67T / (24 × 8192) ≈ 23.6M steps
- Output to `outputs/agpt_2b_<timestamp>/`

### Small Cluster (8 nodes)

For testing on a small multi-node cluster:

```bash
# With hostfile
./run_train_agpt.sh --model 2b --nnodes 8 multi /path/to/hosts.txt

# In PBS job (auto-detect)
qsub -l select=8 -v "MODEL=2b" batch_script.sh
# Inside batch_script.sh:
./run_train_agpt.sh --model 2b multi
```

Parameters (2B on 8 nodes):
- GBS = 96 × 2 / 1 = 192 ✓ Recommended
- NNODES = 8
- Training steps ≈ 2.97M

### Large-Scale Training (512 nodes)

For production training on 512 nodes:

```bash
# In PBS submission script
qsub -l select=512 -A AuroraGPT run_train_agpt_batch.sh
```

Parameters (2B on 512 nodes):
```bash
eval $(./calculate_agpt_params.sh --model 2b --nnodes 512)
# AGPT_GBS=12,288 (same GBS maintained by increasing nodes!)
# AGPT_TRAINING_STEPS=2,970,456 (same token budget)
```

**Key insight:** GBS stays constant across scales because:
- More nodes → more GPUs → more data parallelism → automatic load balancing
- Training steps stay the same → consumes same tokens

### Custom Parallelism

Override defaults for specific needs:

```bash
# Use tensor parallelism for memory-intensive experiments
./run_train_agpt.sh --model 2b --nnodes 8 --tp 2 --lbs 1 multi hosts.txt

# This sets:
# GBS = (96 × 1 × 1) / (2 × 1 × 1) = 48 (same as 1 node with LBS=2 but split)

# Increase batch size for convergence speed
./run_train_agpt.sh --model 2b --nnodes 8 --lbs 4 multi hosts.txt
# GBS = (96 × 4) / 1 = 384 (2x larger)
```

## Dry-Run Mode

Always verify the command before running large jobs:

```bash
./run_train_agpt.sh --model 20b --nnodes 32 multi hosts.txt --dry-run
```

Output:
```
==========================================
AGPT Training Configuration
==========================================
Model:                    20b
Mode:                     multi
Nodes:                    32
Global Batch Size:        768
Tensor Parallelism:       1
Training Steps:           2970456
Sequence Length:          8192
Model Path:               $HOME/models/agpt-20b
Log Directory:            outputs/agpt_20b_20261007_123456
Checkpoint Directory:     agpt-20b-sophiag-olmo-mix-1124-n32-gbs768
==========================================
```

## Advanced: Manual Parameter Calculation

For maximum control, use the calculator directly:

```bash
# See all calculated values
./calculate_agpt_params.sh --model 80b --nnodes 64 --dry-run
```

### JSON Output

For integration with scripts or tools:

```bash
./calculate_agpt_params.sh --model 2b --nnodes 8 --output json
```

## PBS Job Submission

### Batch Job Submission

Submit a job to the scheduler:

```bash
# Create submit_2b_training.sh
#!/bin/bash
#PBS -A AuroraGPT
#PBS -l walltime=06:00:00
#PBS -l select=8
#PBS -q workq

cd "${PBS_O_WORKDIR}"
./run_train_agpt.sh --model 2b multi
# No --nnodes needed; auto-detected from PBS_NODEFILE
```

Submit:
```bash
qsub submit_2b_training.sh
```

### Continuation Jobs

Resume from a checkpoint:

```bash
# Find latest checkpoint
LATEST_CKPT=$(ls -1dt outputs/agpt_2b_*/agpt-2b-sophiag-*/checkpoint-* | head -1)

# Submit continuation job with checkpoint
CKPT=$LATEST_CKPT ./run_train_agpt.sh --model 2b multi
```

## Troubleshooting

### Error: "Model path not found"

The script defaults to `$HOME/models/agpt-{model}`. Provide the correct path:

```bash
./run_train_agpt.sh --model 2b --model-path /lus/flare/path/to/agpt-2b single
```

### Error: "NNODES required for multi mode"

Outside a PBS job, you must specify node count:

```bash
./run_train_agpt.sh --model 2b --nnodes 8 multi /path/to/hosts.txt
```

### Error: "TP×PP×CP exceeds total GPUs"

Your parallelism settings don't fit in available GPUs:

```bash
# Wrong: TP=4 on 8 nodes = 96 GPUs, but TP=4 alone uses 4 GPUs
./calculate_agpt_params.sh --model 80b --nnodes 8 --tp 4
# Error: TP×PP×CP (4) > NGPUS (96) ✗

# Correct: Use TP=2 on 8 nodes
./calculate_agpt_params.sh --model 80b --nnodes 8 --tp 2
# TP×PP×CP = 2×1×1 = 2 ≤ 96 ✓
```

### GBS is Very Small (< 32)

This happens on single node or very small clusters. Consider:
1. Increase LBS: `--lbs 4` (more GPU memory needed)
2. Use gradient accumulation: `--gas 4` (slower per-step, but allows smaller LBS)
3. Use more nodes

```bash
# Single node: GBS = 12×2 = 24 (very small)
./run_train_agpt.sh --model 2b single

# Better: Use gradient accumulation
./run_train_agpt.sh --model 2b single --gas 2
# GBS = 12×2×2 = 48 (accumulate 2 steps before update)
```

## Reference: Parameter Defaults

| Parameter | 2B | 20B | 80B | Notes |
|-----------|----|----|------|-------|
| TP (Tensor Parallelism) | 1 | 1 | 2 | 80B needs TP=2 for memory |
| PP (Pipeline Parallelism) | 1 | 1 | 1 | All models use PP=1 |
| CP (Context Parallelism) | 1 | 1 | 1 | All models use CP=1 |
| LBS (Local Batch Size) | 2 | 2 | 1 | 80B uses smaller LBS |
| GAS (Gradient Accumulation) | 1 | 1 | 1 | Can be overridden |
| SEQ_LEN (Sequence Length) | 8192 | 8192 | 8192 | Default; can be overridden |
| TRAIN_TOKENS (Token Budget) | 4.67T | 4.67T | 4.67T | Shared target across models |
| OPTIMIZER | SophiaG | SophiaG | SophiaG | Production optimizer |
| LR (Learning Rate) | 2.28e-5 | 2.28e-5 | 2.28e-5 | Production-tuned |

## See Also

- `run_train_torchtitan.sh` - Generic TorchTitan launcher for any model
- `train_agpt_2b_venv.sh`, `train_agpt_20b_venv.sh`, `train_agpt_80b_venv.sh` - Original PBS scripts
- AGPT model configs in TorchTitan repo: `torchtitan/experiments/ezpz/agpt/`
