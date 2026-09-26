# XPU Launcher Training System - Complete Index

**Location**: `/lus/flare/projects/datascience/seonghapark/xpu_launcher/`

**Created**: 2026-09-25

---

## 📋 Quick Access

| Document | Purpose | Read Time |
|----------|---------|-----------|
| **QUICK_START.txt** | Fast reference guide | 5 min |
| **TRAINING_GUIDE.md** | Comprehensive manual | 15 min |
| **SETUP_SUMMARY.md** | System overview | 10 min |
| **This file (INDEX.md)** | Navigation guide | 5 min |

---

## 🚀 Getting Started in 3 Steps

### 1. Start Training
```bash
cd /lus/flare/projects/datascience/seonghapark/xpu_launcher
./start_training.sh --model-path /path/to/model
```

### 2. Wait for Completion
Training runs for ~5 hours (18,000 steps by default). Monitor with:
```bash
tail -f outputs/xpu_torchtitan_*/logs/*.log
```

### 3. Review Summary
When training finishes, the summary is **automatically generated**:
```bash
cat outputs/xpu_torchtitan_*/TRAINING_SUMMARY.md
```

---

## 📁 Files in This Directory

### Executable Scripts

#### `start_training.sh` (5.4 KB)
**Main training launcher script**
- Handles single-node and multi-node training
- Validates configuration
- Launches TorchTitan via xpu_launcher
- Auto-generates summary when training completes

**Basic usage:**
```bash
./start_training.sh --model-path /path/to/model
```

**Features:**
- Dry-run mode (`--dry-run`)
- Custom training steps
- Supports different models (llama3, agpt, etc.)
- Checkpoint resume capability
- PBS/SLURM integration

**See also:** `TRAINING_GUIDE.md` for detailed options

---

#### `train_monitor.py` (12 KB)
**Summary generation and metrics collection**
- Parses training logs for loss/accuracy data
- Extracts error and warning messages
- Collects system resource metrics (memory, CPU)
- Generates Markdown summary report
- Can be run independently

**Usage:**
```bash
python train_monitor.py /path/to/log/directory
python train_monitor.py /path/to/log/directory /path/to/output.md
```

**Features:**
- Intelligent log parsing
- Trend analysis (improving/degrading)
- Resource utilization tracking
- Error/warning collection
- Clean Markdown output

---

### Documentation Files

#### `QUICK_START.txt` (8.6 KB)
**Quick reference card**
- Command examples
- Common use cases
- Argument reference
- Troubleshooting tips
- Example outputs

**Start here if:** You want fast answers and command examples

---

#### `TRAINING_GUIDE.md` (9.5 KB)
**Comprehensive training manual**
- Detailed command options
- Examples for all scenarios
- Environment variable reference
- Real-time monitoring tips
- Job scheduler integration
- Troubleshooting guide
- Advanced usage patterns

**Start here if:** You need detailed explanation and setup guidance

---

#### `SETUP_SUMMARY.md` (7.4 KB)
**System overview and architecture**
- What was created and why
- Key features and capabilities
- Use cases and examples
- Integration points
- Customization options
- Next steps

**Start here if:** You want to understand the system design

---

#### `README.md`
**xpu_launcher documentation**
- Launcher CLI reference
- Scheduler integration
- Auto-retry and failover
- Accelerator backends (XPU/CUDA/ROCm)
- Package layout

**See also:** Original xpu_launcher documentation

---

#### `README_launcher_torchtitan.md`
**TorchTitan integration details**
- Launch path overview
- Model selection
- Usage examples
- Environment variables
- AGPT 2B checkpoint loading

**See also:** TorchTitan integration specifics

---

## 📊 Summary Report Structure

The automatically generated `TRAINING_SUMMARY.md` contains:

```
TRAINING_SUMMARY.md
├── Training Summary Report (title and timestamp)
├── Training Duration
│   ├── Elapsed Time (HH:MM:SS)
│   └── Start Time
├── Loss Metrics
│   ├── Observations (count)
│   ├── Mean/Min/Max Loss
│   └── Trend (improving/degrading/stable)
├── Accuracy Metrics
│   ├── Mean/Min/Max Accuracy
│   └── Trend
├── System Resource Consumption
│   ├── Peak Memory (GB and %)
│   ├── Max CPU %
│   └── Memory Timeline
├── Output Paths
│   ├── Checkpoints
│   ├── Logs
│   └── TensorBoard Events
├── Errors (if any)
├── Warnings (if any)
└── Additional Information
    ├── Report generation time
    └── Log directory path
```

---

## 🔧 Usage Patterns

### Pattern 1: Basic Training
```bash
./start_training.sh --model-path /path/to/Llama-3.1-8B
```
- Single-node, 5-hour run
- Default configuration
- Auto-summary generation

### Pattern 2: Quick Test
```bash
./start_training.sh --model-path /path/to/model --training-steps 100
```
- ~2 minute run
- Verify setup works
- Check summary generation

### Pattern 3: Multi-Node
```bash
./start_training.sh --mode multi --hostfile nodes.txt \
  --model-path /path/to/model
```
- Multi-node training
- Hostfile required
- Auto-retry enabled

### Pattern 4: Custom Configuration
```bash
SEQ_LEN=8192 NPROC_PER_NODE=6 \
./start_training.sh --model-path /path/to/model \
  --training-steps 5000
```
- Custom sequence length
- Custom processes per node
- Specific training steps

### Pattern 5: Resume Training
```bash
CKPT=/path/to/checkpoint/step-10000 \
./start_training.sh --model-path /path/to/model \
  --training-steps 8000
```
- Resume from checkpoint
- Continue training
- Different config allowed

### Pattern 6: Preview Command
```bash
./start_training.sh --model-path /path/to/model --dry-run
```
- See exact command
- No execution
- Validate configuration

---

## 📖 Reading Guide

### "I just want to train a model"
1. Read: **QUICK_START.txt** (5 min)
2. Run: `./start_training.sh --model-path /your/model`
3. Monitor: `tail -f outputs/xpu_torchtitan_*/logs/*.log`
4. Review: Open `TRAINING_SUMMARY.md` when done

### "I need detailed instructions"
1. Read: **SETUP_SUMMARY.md** (10 min)
2. Read: **TRAINING_GUIDE.md** (15 min)
3. Choose your scenario
4. Run the appropriate command
5. Check results in summary

### "I want to understand the system"
1. Read: **SETUP_SUMMARY.md** (system overview)
2. Review: **start_training.sh** (script logic)
3. Review: **train_monitor.py** (summary generation)
4. Read: **TRAINING_GUIDE.md** (usage details)

### "I need to troubleshoot something"
1. Check: **QUICK_START.txt** troubleshooting section
2. Check: **TRAINING_GUIDE.md** troubleshooting section
3. Run: `./start_training.sh --model-path /path --dry-run`
4. Check: Log files in output directory
5. See: xpu_launcher README.md for launcher issues

---

## 🎯 Command Reference

### Essential Commands

Start training:
```bash
./start_training.sh --model-path /path/to/model
```

Preview command:
```bash
./start_training.sh --model-path /path/to/model --dry-run
```

Quick test:
```bash
./start_training.sh --model-path /path/to/model --training-steps 100
```

Multi-node:
```bash
./start_training.sh --mode multi --hostfile nodes.txt \
  --model-path /path/to/model
```

Get help:
```bash
./start_training.sh --help
```

### Monitoring Commands

Watch logs in real-time:
```bash
tail -f outputs/xpu_torchtitan_*/logs/*.log
```

Check progress (grep for loss):
```bash
watch -n 5 'tail -20 outputs/xpu_torchtitan_*/logs/*.log | grep -i loss'
```

Monitor system (XPU):
```bash
watch xpu-smi
```

Monitor system (CUDA):
```bash
watch nvidia-smi
```

### Summary Commands

Generate summary manually:
```bash
python train_monitor.py outputs/xpu_torchtitan_<timestamp>
```

View summary:
```bash
cat outputs/xpu_torchtitan_<timestamp>/TRAINING_SUMMARY.md
```

---

## ⚙️ Configuration Reference

### Required Arguments
- `--model-path PATH` : Model/tokenizer directory (REQUIRED)

### Optional Arguments
- `--mode {single|multi}` : Training mode (default: single)
- `--hostfile FILE` : Hostfile for multi-node (required for multi)
- `--module NAME` : TorchTitan module (default: llama3)
- `--config NAME` : Config callable (default: llama3_debugmodel)
- `--dataset NAME` : Dataset name (default: pg19_multinews)
- `--training-steps N` : Training steps (default: 18000)
- `--log-dir DIR` : Output directory (default: auto-generated)
- `--dry-run` : Preview without execution
- `--help` : Show help message

### Environment Variables
- `MODEL_PATH` : Model directory (alternative to --model-path)
- `CKPT` : Checkpoint path for resume
- `SEQ_LEN` : Sequence length (default: 16384)
- `NPROC_PER_NODE` : Processes per node (default: 4)
- `NNODES` : Number of nodes
- `AUTO_RETRY` : Enable auto-retry (default: 1 for multi)
- `SPARE_NODES` : Spare nodes for failover
- `XPU_SHOW_WARNINGS` : Show warnings (default: 0 = silent)

---

## 🎓 Example Scenarios

### Scenario 1: First-Time Training
```bash
# 1. Prepare
ls -la /path/to/Llama-3.1-8B/  # Verify path exists

# 2. Preview
./start_training.sh --model-path /path/to/Llama-3.1-8B \
  --dry-run

# 3. Quick test (2 minutes)
./start_training.sh --model-path /path/to/Llama-3.1-8B \
  --training-steps 100

# 4. Check test results
cat outputs/xpu_torchtitan_*/TRAINING_SUMMARY.md

# 5. Full training (5 hours)
./start_training.sh --model-path /path/to/Llama-3.1-8B
```

### Scenario 2: Multi-Node on HPC
```bash
# Inside PBS job
./start_training.sh --mode multi \
  --model-path /path/to/model \
  --training-steps 18000
```

### Scenario 3: Custom Configuration
```bash
# AGPT 2B with custom settings
./start_training.sh \
  --model-path /path/to/agpt-2b \
  --module agpt \
  --config agpt_2b \
  --dataset PG19 \
  --training-steps 5000
```

### Scenario 4: Resume from Checkpoint
```bash
CKPT=/previous/run/checkpoint/step-10000 \
./start_training.sh \
  --model-path /path/to/model \
  --training-steps 8000
```

---

## 📞 Support & Help

| Topic | Resource |
|-------|----------|
| Training script | `TRAINING_GUIDE.md` |
| Quick reference | `QUICK_START.txt` |
| System design | `SETUP_SUMMARY.md` |
| xpu_launcher | `README.md` |
| TorchTitan | `README_launcher_torchtitan.md` |
| Command options | Run `./start_training.sh --help` |

---

## ✅ Checklist Before Training

- [ ] Read QUICK_START.txt or TRAINING_GUIDE.md
- [ ] Verify model path exists: `ls -la /path/to/model`
- [ ] Check model contains config.json and tokenizer files
- [ ] Test with dry-run: `./start_training.sh --model-path /path --dry-run`
- [ ] For multi-node: prepare hostfile with node names
- [ ] Ensure sufficient disk space for checkpoints and logs
- [ ] Verify compute resources available

---

## 📈 What You'll Get

After 5-hour training, in `outputs/xpu_torchtitan_<timestamp>/`:

```
✓ TRAINING_SUMMARY.md
  ├─ Elapsed time
  ├─ Loss metrics (min/max/mean/trend)
  ├─ Accuracy metrics
  ├─ Memory usage
  ├─ Checkpoint locations
  ├─ Log file paths
  └─ Errors & warnings

✓ checkpoint/
  ├─ step-1000/
  ├─ step-2000/
  └─ ...

✓ logs/
  ├─ rank-0.log
  ├─ rank-1.log
  └─ ...

✓ events.out.tfevents.* (TensorBoard)
```

---

## 🔗 Key Paths

| Item | Path |
|------|------|
| Training script | `./start_training.sh` |
| Summary generator | `./train_monitor.py` |
| Output directory | `./outputs/xpu_torchtitan_YYYYMMDD_HHMMSS/` |
| Generated summary | `./outputs/xpu_torchtitan_*/TRAINING_SUMMARY.md` |
| Checkpoints | `./outputs/xpu_torchtitan_*/checkpoint/` |
| Logs | `./outputs/xpu_torchtitan_*/logs/` |

---

## 🚦 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| "Model directory does not exist" | Check path: `ls -la /your/model/path` |
| "Cannot find xpu launcher" | Check: `xpu --help` or reinstall |
| "Command not found" | Run from xpu_launcher directory |
| "Multi-node fails" | Ensure hostfile exists and is readable |
| "Summary not generated" | Manually run: `python train_monitor.py <log_dir>` |
| "Training too slow" | Check GPU/XPU utilization with monitoring tools |

---

## 📝 Notes

- **Timing**: Default 18,000 steps ≈ 5 hours (varies by hardware)
- **Summary**: Generated automatically, no manual intervention needed
- **Checkpoints**: Saved every 1,000 steps (configurable)
- **Logs**: Collected in real-time during training
- **Resources**: Peak memory typically 45-80 GB depending on model size

---

**Last Updated**: 2026-09-25

For the latest information, see:
- **SETUP_SUMMARY.md** - System overview
- **TRAINING_GUIDE.md** - Comprehensive guide
- **QUICK_START.txt** - Quick reference
