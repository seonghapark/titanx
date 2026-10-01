# 🚀 Training Script Update - Start Here

**Date**: 2026-09-26  
**Status**: ✅ Ready for use

---

## What's New?

Your `start_training_with_conversion.sh` script has been **enhanced to work without explicit hostfile in multi-node mode**. It now automatically uses `PBS_NODEFILE` when running inside a PBS job, just like `./xpu_torchtitan/run_train_torchtitan.sh`.

---

## 📖 Quick Links

| Document | Purpose |
|----------|---------|
| **REFERENCE_CARD.txt** | Quick commands & examples (start here!) |
| **TRAINING_WITH_CONVERSION_USAGE.md** | Comprehensive usage guide |
| **SCRIPT_UPDATE_SUMMARY.md** | Technical details of changes |
| **QUICK_START_TRAINING.sh** | Copy & customize this script |
| **COMPARISON_REPORT.md** | Analysis of your last 2 training runs |

---

## ⚡ Fastest Way to Start

### Option 1: Simple PBS Job

```bash
cd /lus/flare/projects/datascience/seonghapark/xpu_launcher

qsub << 'EOF'
#!/bin/bash
#PBS -l select=9:ngpus=1
#PBS -l walltime=12:00:00

./start_training_with_conversion.sh \
  --model-path /lus/flare/projects/datascience/seonghapark/xpu_launcher/xpu_torchtitan/torchtitan_repo/outputs/xpu_torchtitan_20260925_160731/checkpoint/step-2062 \
  --mode multi \
  -- --training.steps 400000 --training.seq_len 16384
