# Training Guide with Automatic Summary Generation

This guide explains how to start model training and automatically generate a comprehensive summary document when training completes.

## Quick Start

### Single-Node Training

```bash
cd /lus/flare/projects/datascience/seonghapark/xpu_launcher

# Dry run to inspect the command
./start_training.sh \
  --model-path /path/to/model \
  --dry-run

# Real training (5 hours, ~18000 steps)
./start_training.sh \
  --model-path /path/to/llama-3.1-8b
```

### Multi-Node Training

```bash
cd /lus/flare/projects/datascience/seonghapark/xpu_launcher

# With explicit hostfile
./start_training.sh \
  --mode multi \
  --hostfile /path/to/hosts \
  --model-path /path/to/llama-3.1-8b

# Inside a PBS job (hostfile from PBS_NODEFILE)
./start_training.sh \
  --mode multi \
  --model-path /path/to/llama-3.1-8b
```

## Command Options

```
./start_training.sh [OPTIONS]

OPTIONS:
  --mode {single|multi}        Training mode (default: single)
  --hostfile FILE              Hostfile path (required for multi mode)
  --model-path PATH            Model/tokenizer directory (REQUIRED)
  --module NAME                Config module name (default: llama3)
  --config NAME                Config callable (default: llama3_debugmodel)
  --dataset NAME               Dataset name (default: pg19_multinews)
  --training-steps N           Number of training steps (default: 18000)
  --log-dir DIR                Output directory (default: outputs/xpu_torchtitan_<timestamp>)
  --dry-run                    Print command without running
  --help                       Show help message
```

## Examples

### Llama 3.1 8B Training

```bash
./start_training.sh \
  --model-path /path/to/Llama-3.1-8B \
  --module llama3 \
  --config llama3_8b \
  --dataset c4 \
  --training-steps 5000
```

### AGPT 2B with Checkpoint Resume

```bash
./start_training.sh \
  --model-path /path/to/agpt-2b \
  --module agpt \
  --config agpt_2b \
  --dataset PG19 \
  --training-steps 1000 \
  --log-dir ./outputs/agpt_resume_run
```

Environment variable for checkpoint:
```bash
CKPT=/path/to/checkpoint/step-92859 \
./start_training.sh \
  --model-path /path/to/model \
  ...
```

### Custom Sequence Length and Steps

```bash
SEQ_LEN=8192 \
./start_training.sh \
  --model-path /path/to/model \
  --training-steps 10000 \
  --log-dir ./outputs/custom_seq_run
```

### Multi-Node Training with Aurora

```bash
./start_training.sh \
  --mode multi \
  --hostfile nodes.txt \
  --model-path /path/to/model \
  --training-steps 18000 \
  NPROC_PER_NODE=6 \
  --log-dir ./outputs/multi_node_run
```

## Training Duration

Training time depends on the number of steps:

- **18,000 steps** (default): ~5 hours at ~1 step/second
- **5,000 steps**: ~1.4 hours
- **1,000 steps**: ~17 minutes

Actual timing varies based on:
- Hardware (CPU, GPU, XPU count)
- Model size
- Batch size and sequence length
- Network latency (multi-node)

## Output Structure

After training completes, the output directory contains:

```
outputs/xpu_torchtitan_YYYYMMDD_HHMMSS/
├── TRAINING_SUMMARY.md              ← Generated summary report
├── checkpoint/                       ← Model checkpoints
│   ├── step-1000/
│   ├── step-2000/
│   └── ...
├── logs/                            ← Training logs
│   └── *.log
└── events.out.tfevents.*            ← TensorBoard events (if enabled)
```

## Training Summary Report

The `TRAINING_SUMMARY.md` file generated in the output directory contains:

### Metrics Collected
- **Training Duration**: Total elapsed time in HH:MM:SS format
- **Loss Metrics**: Min, max, mean loss and trend (improving/degrading)
- **Accuracy Metrics**: Min, max, mean accuracy and trend
- **Resource Consumption**: Peak memory (GB and %), CPU usage
- **Output Paths**: Locations of checkpoints, logs, and other artifacts
- **Errors & Warnings**: All errors and warnings encountered during training
- **Execution Details**: Start time, report generation time, log directory

### Example Summary Format
```markdown
# Training Summary Report
Generated: 2026-09-25 14:32:15

## Training Duration
- **Elapsed Time**: 05:00:23
- **Start Time**: 2026-09-25 09:32:00

## Loss Metrics
- **Observations**: 18000
- **Mean Loss**: 2.345678
- **Min Loss**: 1.234567
- **Max Loss**: 5.432109
- **Trend**: improving (Δ -0.156789)

## Accuracy Metrics
- **Observations**: 18000
- **Mean Accuracy**: 0.756234
- **Trend**: improving (Δ +0.087654)

## System Resource Consumption
- **Peak Memory**: 45.67 GB
- **Max Memory %**: 78.90%
- **Max CPU %**: 95.00%

## Output Paths
### Checkpoints
- `/path/to/outputs/checkpoint/step-1000`
- `/path/to/outputs/checkpoint/step-2000`
...
```

## Environment Variables

### Key Variables for Training

| Variable | Purpose | Example |
|---|---|---|
| `MODEL_PATH` | Model/tokenizer directory | `/path/to/Llama-3.1-8B` |
| `MODULE` | TorchTitan config module | `llama3` or `agpt` |
| `CONFIG` | Config callable | `llama3_8b` or `agpt_2b` |
| `DATASET_NAME` | Dataset to use | `c4`, `pg19_multinews`, `PG19` |
| `TRAINING_STEPS` | Number of training steps | `5000`, `18000` |
| `LOG_DIR` | Output directory | `./outputs/run_name` |
| `SEQ_LEN` | Sequence length | `16384`, `8192` |
| `CKPT` | Checkpoint to resume from | `/path/to/checkpoint/step-N` |
| `NPROC_PER_NODE` | Processes per node | `4`, `6`, `8` |
| `NNODES` | Number of nodes (multi) | `4`, `8`, `16` |

### Launch Environment Variables

| Variable | Purpose |
|---|---|
| `AUTO_RETRY` | Enable auto-retry on failures (default: 1 for multi-node) |
| `SPARE_NODES` | Number of spare nodes for failover |
| `FAILOVER_PROFILE` | Failover strategy (`auto`, `aurora`, `slurm`) |
| `HOST_IP_MAP` | JSON file with IP to hostname mapping |
| `XPU_SHOW_WARNINGS` | Show import warnings (default: 0 = silent) |

## Monitoring During Training

### Watch Training in Real-Time

```bash
# Monitor the main log
tail -f outputs/xpu_torchtitan_*/logs/*.log

# Check training progress
watch -n 5 'tail -20 outputs/xpu_torchtitan_*/logs/*.log | grep -i "step\|loss\|accuracy"'
```

### Check Resource Usage

```bash
# Monitor system resources during training
watch -n 2 'nvidia-smi'  # for CUDA
watch -n 2 'xpu-smi'     # for XPU
```

### Monitor Training Completion

The training will:
1. Run for the specified number of steps
2. Automatically generate the summary report
3. Print the summary location to stdout

Example output:
```
==========================================
Training Completed
==========================================
Duration: 05:00:23
Log directory: /lus/flare/projects/datascience/seonghapark/xpu_launcher/outputs/xpu_torchtitan_20260925_093200

Generating training summary...
Summary document generated: /lus/flare/projects/datascience/seonghapark/xpu_launcher/outputs/xpu_torchtitan_20260925_093200/TRAINING_SUMMARY.md
```

## Troubleshooting

### Training Not Starting

1. Verify model path exists:
   ```bash
   ls -la /path/to/model
   ```

2. Check xpu_launcher is installed:
   ```bash
   pip show xpu-launcher
   ```

3. Verify TorchTitan is available:
   ```bash
   python -c "from torchtitan.train import main; print('OK')"
   ```

### Training Crashes

1. Check logs for error messages:
   ```bash
   grep -i error outputs/xpu_torchtitan_*/logs/*.log | head -20
   ```

2. Verify compute resources:
   ```bash
   xpu doctor  # for XPU systems
   nvidia-smi  # for CUDA systems
   ```

3. Try reducing batch size or sequence length:
   ```bash
   SEQ_LEN=8192 ./start_training.sh ...
   ```

### Summary Not Generated

1. Check if training completed:
   ```bash
   tail -20 outputs/xpu_torchtitan_*/logs/*.log
   ```

2. Manually generate summary:
   ```bash
   python train_monitor.py outputs/xpu_torchtitan_<timestamp>
   ```

## Integration with Job Schedulers

### PBS (Aurora)

```bash
#!/bin/bash
#PBS -l select=4:ngpus=6
#PBS -l walltime=6:00:00
#PBS -q queue_name

cd /lus/flare/projects/datascience/seonghapark/xpu_launcher

./start_training.sh \
  --mode multi \
  --model-path /path/to/model \
  --training-steps 18000
```

### SLURM

```bash
#!/bin/bash
#SBATCH --nnodes=4
#SBATCH --ntasks-per-node=6
#SBATCH --time=6:00:00

cd /lus/flare/projects/datascience/seonghapark/xpu_launcher

./start_training.sh \
  --mode multi \
  --model-path /path/to/model \
  --training-steps 18000
```

## Advanced Usage

### Dry-Run Mode

Test your configuration without executing training:

```bash
./start_training.sh \
  --model-path /path/to/model \
  --dry-run
```

Output shows the exact command that would be executed.

### Custom Log Directories

Organize outputs with meaningful names:

```bash
LOG_DIR=./outputs/llama3_8b_v2_seq16k \
./start_training.sh \
  --model-path /path/to/model \
  --training-steps 18000
```

### Resume from Checkpoint

Continue training from a previous run:

```bash
CKPT=./outputs/previous_run/checkpoint/step-10000 \
./start_training.sh \
  --model-path /path/to/model \
  --training-steps 5000  # additional steps
```

## Support and Documentation

- **xpu_launcher docs**: See `README.md` in the launcher directory
- **TorchTitan docs**: Check `xpu_torchtitan/torchtitan_repo` for TorchTitan documentation
- **Training script details**: See `start_training.sh --help`
- **Summary generator**: See `train_monitor.py --help`

## Additional Notes

- Summary generation is automatic when training completes
- All metrics are collected from the training logs
- The summary includes both training metrics and system resource usage
- For multi-node training, ensure all nodes have synchronized clocks
- Check `XPU_LAUNCHER_DEBUG=1` environment variable for detailed debug output
