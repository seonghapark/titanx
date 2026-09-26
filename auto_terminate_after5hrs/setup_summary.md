# Training Monitoring & Summary Generation Setup

## Overview

I've created an automated system to run your 5-hour model training jobs and generate comprehensive summary documents containing:

- ✅ Training metrics (loss, accuracy trends)
- ✅ Resource consumption (memory, CPU)
- ✅ Output paths (checkpoints, logs)
- ✅ Errors and warnings encountered
- ✅ Execution timing and statistics

## Files Created

### 1. **`start_training.sh`** (Main launcher)
A user-friendly training launcher script that:
- Validates configuration and model paths
- Starts single-node or multi-node training
- Monitors training completion
- Automatically generates summary when done

**Usage:**
```bash
./start_training.sh \
  --model-path /path/to/model \
  --training-steps 18000
```

### 2. **`train_monitor.py`** (Summary generator)
A Python utility that:
- Parses training logs for loss and accuracy metrics
- Collects system resource metrics
- Extracts errors and warnings
- Generates a Markdown summary report

**Standalone usage:**
```bash
python train_monitor.py /path/to/log/directory
```

### 3. **`TRAINING_GUIDE.md`** (Documentation)
Comprehensive guide covering:
- Quick start examples
- Command-line options
- Environment variables
- Output structure
- Troubleshooting tips
- Job scheduler integration

## Quick Start

### Single-Node Training (5 hours)
```bash
cd /lus/flare/projects/datascience/seonghapark/xpu_launcher

./start_training.sh \
  --model-path /path/to/Llama-3.1-8B \
  --training-steps 18000  # ~5 hours
```

### Multi-Node Training
```bash
./start_training.sh \
  --mode multi \
  --hostfile nodes.txt \
  --model-path /path/to/model \
  --training-steps 18000
```

### Dry-Run (Preview command)
```bash
./start_training.sh \
  --model-path /path/to/model \
  --dry-run
```

## What Happens

1. **Configuration Phase**
   - Validates required arguments (model path, etc.)
   - Creates output directory
   - Displays training configuration

2. **Training Phase**
   - Launches training via `xpu launch`
   - TorchTitan runs for specified steps
   - Logs are written to `LOG_DIR`

3. **Summary Generation Phase** (Automatic)
   - Parses all training logs
   - Collects loss/accuracy metrics
   - Gathers resource consumption data
   - Generates `TRAINING_SUMMARY.md` in output directory

## Output Structure

After training completes:
```
outputs/xpu_torchtitan_YYYYMMDD_HHMMSS/
├── TRAINING_SUMMARY.md          ← Summary report (AUTO-GENERATED)
├── checkpoint/
│   ├── step-1000/
│   ├── step-2000/
│   └── ...
├── logs/
│   └── *.log
└── events.out.tfevents.*
```

## Summary Report Example

The generated `TRAINING_SUMMARY.md` includes:

```markdown
# Training Summary Report
Generated: 2026-09-25 14:32:15

## Training Duration
- **Elapsed Time**: 05:00:23

## Loss Metrics
- **Observations**: 18000
- **Mean Loss**: 2.345678
- **Trend**: improving (Δ -0.156789)

## Accuracy Metrics
- **Mean Accuracy**: 0.756234
- **Trend**: improving (Δ +0.087654)

## System Resource Consumption
- **Peak Memory**: 45.67 GB
- **Max Memory %**: 78.90%
- **Max CPU %**: 95.00%

## Output Paths
### Checkpoints
- `/outputs/checkpoint/step-1000`
...

## Errors & Warnings
[Lists any errors or warnings encountered]
```

## Key Features

### 1. Flexible Configuration
- Single or multi-node training
- Customizable training steps (default: 18000 for ~5 hours)
- Support for any TorchTitan model (llama3, agpt, etc.)
- Optional checkpoint resume
- Custom sequence lengths

### 2. Automatic Summary Generation
- Runs automatically after training completes
- No manual intervention required
- Parses logs intelligently
- Collects resource metrics
- Organizes results in clean Markdown format

### 3. Multiple Dataset Support
- `pg19_multinews` (default, streaming)
- `PG19` (standalone streaming)
- `c4` (streaming)
- `c4_test` (offline sample)
- Custom dataset paths

### 4. Robust Error Handling
- Validates paths before training
- Captures all errors and warnings
- Provides clear error messages
- Includes troubleshooting guidance

### 5. Job Scheduler Integration
- Works with PBS (Aurora)
- Works with SLURM
- Auto-detects scheduler environment
- Supports hostfile from PBS_NODEFILE

## Common Use Cases

### Case 1: Quick Testing
```bash
./start_training.sh \
  --model-path /path/to/model \
  --training-steps 100  # ~2 minutes
```

### Case 2: Standard 5-Hour Run
```bash
./start_training.sh \
  --model-path /path/to/model  # Required
  # Uses defaults: 18000 steps, single-node, llama3_debugmodel
```

### Case 3: Multi-Node Large Model
```bash
./start_training.sh \
  --mode multi \
  --hostfile /path/to/hosts \
  --model-path /path/to/Llama-3.1-70B \
  --module llama3 \
  --config llama3_70b \
  --training-steps 5000
```

### Case 4: Resume from Checkpoint
```bash
CKPT=/previous/run/checkpoint/step-10000 \
./start_training.sh \
  --model-path /path/to/model \
  --training-steps 8000  # Continue for 8000 more steps
```

## Customization

### Change Default Training Steps
Edit line in `start_training.sh`:
```bash
TRAINING_STEPS="${TRAINING_STEPS:-18000}"  # Change 18000 to desired value
```

### Add More Metrics to Summary
Edit `train_monitor.py` to:
- Add custom metric extraction patterns
- Include additional resource metrics
- Generate custom visualizations

### Integrate with Experiment Tracking
The summary can be extended to post results to:
- Weights & Biases (wandb)
- MLflow
- TensorBoard
- Custom dashboards

## Integration Points

### With Existing Workflows
1. Include in PBS/SLURM job scripts
2. Trigger from GitHub Actions/CI-CD
3. Integrate with experiment management systems
4. Combine with data preprocessing pipelines

### Monitor from Remote
1. SSH to server and check output directory
2. Copy summary to local machine
3. Parse JSON version (can extend `train_monitor.py`)
4. Push to cloud storage for tracking

## Environment Variables

All environment variables are passed through:
```bash
NPROC_PER_NODE=6 \
SEQ_LEN=8192 \
CKPT=/path/to/checkpoint \
./start_training.sh --model-path /path/to/model
```

## Troubleshooting

### Training doesn't start
```bash
./start_training.sh --model-path /path/to/model --dry-run  # Check command
ls -la /path/to/model  # Verify model exists
```

### Summary not generated
```bash
# Manually run:
python train_monitor.py outputs/xpu_torchtitan_<timestamp>
```

### Missing metrics in summary
Check log format in `train_monitor.py` line where patterns are defined.
Common log patterns can be added for your specific logging framework.

## Next Steps

1. **Review Documentation**
   - Read `TRAINING_GUIDE.md` for detailed options
   - Check `README.md` for xpu_launcher details
   - Read `README_launcher_torchtitan.md` for TorchTitan info

2. **Prepare for Training**
   - Locate your model and tokenizer
   - Note the exact paths
   - Check available compute resources

3. **Run Training**
   ```bash
   ./start_training.sh --model-path /your/model/path
   ```

4. **Monitor Output**
   - Tail the logs in real-time
   - Check summary when training completes
   - Review metrics and resource usage

## Support

For issues with:
- **xpu_launcher**: See `README.md`
- **TorchTitan**: See `xpu_torchtitan/torchtitan_repo` docs
- **Summary generation**: Check `train_monitor.py` or adjust patterns
- **Training workflow**: Refer to `TRAINING_GUIDE.md`

---

**Created**: 2026-09-25
**Location**: `/lus/flare/projects/datascience/seonghapark/xpu_launcher/`
