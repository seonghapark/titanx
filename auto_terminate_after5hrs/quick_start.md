================================================================================
                        XPU LAUNCHER TRAINING QUICK START
================================================================================

LOCATION: /lus/flare/projects/datascience/seonghapark/xpu_launcher/

================================================================================
BASIC TRAINING (5 HOURS, ~18K STEPS)
================================================================================

Single-Node:
  cd /lus/flare/projects/datascience/seonghapark/xpu_launcher
  ./start_training.sh --model-path /path/to/Llama-3.1-8B

Multi-Node:
  ./start_training.sh --mode multi --hostfile nodes.txt \
    --model-path /path/to/model

================================================================================
WHAT HAPPENS
================================================================================

1. Training runs for specified steps (~5 hours for 18000 steps)
2. Logs are written to outputs/xpu_torchtitan_YYYYMMDD_HHMMSS/
3. When training finishes, TRAINING_SUMMARY.md is AUTO-GENERATED
4. Summary contains:
   - Loss trends (min/max/mean/improvement)
   - Accuracy trends
   - Memory & CPU usage
   - Checkpoint locations
   - Errors & warnings

================================================================================
SUMMARY REPORT PREVIEW
================================================================================

The generated TRAINING_SUMMARY.md includes:

  # Training Summary Report
  Generated: 2026-09-25 14:32:15

  ## Training Duration
  - Elapsed Time: 05:00:23

  ## Loss Metrics
  - Mean Loss: 2.345678
  - Trend: improving (Δ -0.156789)

  ## Accuracy Metrics
  - Mean Accuracy: 0.756234
  - Trend: improving (Δ +0.087654)

  ## System Resource Consumption
  - Peak Memory: 45.67 GB
  - Max Memory %: 78.90%

  ## Output Paths
  - Checkpoints: outputs/checkpoint/step-*/
  - Logs: outputs/logs/*.log

  ## Errors & Warnings
  [Any errors/warnings found]

================================================================================
COMMON COMMANDS
================================================================================

Dry-run (preview without executing):
  ./start_training.sh --model-path /path/to/model --dry-run

Quick test (100 steps, ~2 minutes):
  ./start_training.sh --model-path /path/to/model \
    --training-steps 100

Different model/config:
  ./start_training.sh --model-path /path/to/model \
    --module agpt --config agpt_2b

Custom sequence length:
  SEQ_LEN=8192 ./start_training.sh --model-path /path/to/model

Resume from checkpoint:
  CKPT=/path/to/checkpoint/step-10000 \
  ./start_training.sh --model-path /path/to/model \
    --training-steps 8000

Multi-node with 6 processes per node:
  NPROC_PER_NODE=6 ./start_training.sh --mode multi \
    --hostfile nodes.txt --model-path /path/to/model

================================================================================
MONITORING DURING TRAINING
================================================================================

Watch logs in real-time:
  tail -f outputs/xpu_torchtitan_*/logs/*.log

Check resource usage (XPU):
  watch xpu-smi

Check resource usage (CUDA):
  watch nvidia-smi

Check progress:
  watch -n 5 'tail -20 outputs/xpu_torchtitan_*/logs/*.log | grep -i loss'

================================================================================
OUTPUT LOCATION
================================================================================

After training completes, find results at:
  ./outputs/xpu_torchtitan_YYYYMMDD_HHMMSS/

Contents:
  ├── TRAINING_SUMMARY.md        ← Main summary report (AUTO-GENERATED)
  ├── checkpoint/
  │   ├── step-1000/
  │   ├── step-2000/
  │   └── ...
  ├── logs/                      ← Training logs
  └── events.out.tfevents.*      ← TensorBoard events

================================================================================
REQUIRED ARGUMENTS
================================================================================

MUST specify: --model-path /path/to/model

The model path should contain:
  - config.json        (model configuration)
  - tokenizer files    (tokenizer_config.json, vocab files, etc.)

Example paths:
  - /path/to/Llama-3.1-8B/
  - /path/to/agpt-2b/
  - /lus/flare/projects/.../model_dir/

================================================================================
OPTIONS REFERENCE
================================================================================

  --mode {single|multi}          Training mode (default: single)
  --hostfile FILE                Hostfile for multi-node (required for multi)
  --model-path PATH              Model directory (REQUIRED)
  --module NAME                  TorchTitan module (default: llama3)
  --config NAME                  Config callable (default: llama3_debugmodel)
  --dataset NAME                 Dataset (default: pg19_multinews)
  --training-steps N             Training steps (default: 18000 for ~5 hours)
  --log-dir DIR                  Output directory (default: auto-generated)
  --dry-run                      Preview command, don't execute
  --help                         Show help

================================================================================
ENVIRONMENT VARIABLES
================================================================================

Optional env vars to customize training:

  SEQ_LEN=16384              Sequence length (default: 16384)
  CKPT=/path/to/checkpoint   Resume from checkpoint
  NPROC_PER_NODE=4           Processes per node (default: 4)
  NNODES=4                   Number of nodes (default: auto-detect)

Example:
  SEQ_LEN=8192 NPROC_PER_NODE=6 ./start_training.sh \
    --model-path /path/to/model

================================================================================
DOCUMENTATION
================================================================================

For more details, see:

  SETUP_SUMMARY.md       Overview of the system
  TRAINING_GUIDE.md      Comprehensive guide with examples
  README.md              xpu_launcher documentation
  README_launcher_torchtitan.md  TorchTitan integration details

================================================================================
TROUBLESHOOTING
================================================================================

Q: Error "model directory does not exist"
A: Check the path: ls -la /your/model/path/

Q: Training doesn't start
A: Run with --dry-run to see the exact command being executed:
   ./start_training.sh --model-path /path/to/model --dry-run

Q: Summary not generated
A: Manually generate it:
   python train_monitor.py outputs/xpu_torchtitan_<timestamp>

Q: How long will training take?
A: Default 18000 steps ≈ 5 hours (varies by hardware)
   Adjust with: --training-steps N
   (100 steps = ~2 minutes, 5000 steps = ~1.4 hours)

Q: Can I resume training?
A: Yes, provide the checkpoint path:
   CKPT=/path/to/checkpoint/step-10000 \
   ./start_training.sh --model-path /path/to/model --training-steps 8000

================================================================================
QUICK EXAMPLES
================================================================================

1. BASIC (5 hours, single-node):
   ./start_training.sh --model-path /path/to/Llama-3.1-8B

2. QUICK TEST (2 minutes):
   ./start_training.sh --model-path /path/to/model --training-steps 100

3. MULTI-NODE:
   ./start_training.sh --mode multi --hostfile nodes.txt \
     --model-path /path/to/model

4. AGPT 2B (different model):
   ./start_training.sh --model-path /path/to/agpt-2b \
     --module agpt --config agpt_2b

5. PREVIEW COMMAND:
   ./start_training.sh --model-path /path/to/model --dry-run

6. CUSTOM SETTINGS:
   SEQ_LEN=8192 NPROC_PER_NODE=6 \
   ./start_training.sh --model-path /path/to/model --training-steps 5000

================================================================================
SUMMARY METRICS IN REPORT
================================================================================

The TRAINING_SUMMARY.md contains:

TIMING:
  ├─ Elapsed Time (HH:MM:SS format)
  └─ Start/End times

LOSS METRICS:
  ├─ Number of observations
  ├─ Min/Max/Mean loss
  └─ Trend (improving/degrading/stable)

ACCURACY METRICS:
  ├─ Min/Max/Mean accuracy
  └─ Trend direction

RESOURCES:
  ├─ Peak memory (GB and %)
  ├─ Max CPU %
  └─ Hardware utilization

OUTPUT PATHS:
  ├─ Checkpoint locations
  ├─ Log file paths
  └─ Event files

ISSUES:
  ├─ Errors encountered
  └─ Warnings logged

================================================================================
