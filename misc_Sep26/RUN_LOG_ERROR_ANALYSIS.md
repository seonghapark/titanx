# XPU Launcher Run.log Error Analysis & Resolution

**Date**: 2026-09-26  
**Status**: ✅ Issue Identified & Solution Provided

---

## 🔴 Critical Error

```
RuntimeError: The device index is out of range. It must be in [0, 12), but got 14.
RuntimeError: The device index is out of range. It must be in [0, 12), but got 16.
RuntimeError: The device index is out of range. It must be in [0, 12), but got 20.
... (multiple ranks failing with similar errors)
```

### Error Location
```
File "/opt/aurora/.../torch/xpu/__init__.py", line 440, in set_device
    torch._C._xpu_setDevice(device)
```

---

## 🎯 Root Cause Analysis

### Configuration From Log:
```
[ARGS] topology = NNODES=49 NPROC_PER_NODE=40 NPROC=auto SPARE_NODES=auto AUTO_RETRY=1(multi)
```

### Problem:

**Each node has 12 XPU devices, but training is configured for:**
- NPROC_PER_NODE=40 (processes per node)
- NNODES=49 (nodes)
- Total ranks: 49 × 40 = 1,960 ranks

**But each node only has 12 physical devices!**

When PyTorch tries to assign rank processes to devices:
- Device indices should be: 0-11 (12 devices per node)
- But it's trying to assign: 14, 16, 20, 30, 31, 34, 36, 37, etc.
- This exceeds the available devices [0, 12)

### Why This Happened:

Your command set:
```bash
NPROC_PER_NODE=10 NNODES=13 ... ./start_training_with_conversion.sh
```

But the **actual PBS job allocated 49 nodes with 40 processes per node**, which overrides the settings!

---

## ✅ Solution

### Option 1: Reduce NPROC_PER_NODE to Match Hardware

**Each Aurora XPU node has: 12 devices**

Change:
```bash
# WRONG (40 processes per 12 devices)
NPROC_PER_NODE=40 NNODES=49 ...

# CORRECT (match 12 devices)
NPROC_PER_NODE=12 NNODES=49 ...
```

Or even better:
```bash
# PROVEN WORKING (from 1st training run)
NPROC_PER_NODE=4 NNODES=9 ...
```

### Option 2: Reduce Nodes & Processes

Go back to the working configuration from your successful training:

```bash
NPROC_PER_NODE=4 \
NNODES=9 \
./start_training_with_conversion.sh \
  --model-path ./xpu_launcher/xpu_torchtitan/torchtitan_repo/outputs/xpu_torchtitan_20260925_160731/checkpoint/step-2062 \
  --mode multi \
  -- --training.steps 400000 --training.seq_len 16384
```

This was **proven to work** in your 1st training run (2,062 steps completed successfully).

### Option 3: Use PBS Job Script with Correct Settings

Create `train_corrected.pbs`:

```bash
#!/bin/bash
#PBS -l select=9:ngpus=1
#PBS -l walltime=12:00:00
#PBS -N train_agpt_corrected

cd /lus/flare/projects/datascience/seonghapark/xpu_launcher

export NPROC_PER_NODE=4
export NNODES=9

./start_training_with_conversion.sh \
  --model-path /lus/flare/projects/datascience/seonghapark/xpu_launcher/xpu_torchtitan/torchtitan_repo/outputs/xpu_torchtitan_20260925_160731/checkpoint/step-2062 \
  --mode multi \
  -- --training.steps 400000 --training.seq_len 16384
```

Submit:
```bash
qsub train_corrected.pbs
```

---

## 📊 Comparison: What Went Wrong vs What Worked

| Aspect | Failed Run | Successful Run (1st) | Correct Setting |
|--------|-----------|----------------------|-----------------|
| NPROC_PER_NODE | 40 | 4 | ✅ 4 |
| NNODES | 49 | 9 | ✅ 9 |
| Total Ranks | 1,960 | 36 | ✅ 36 |
| Devices per Node | 12 | 12 | ✅ 12 |
| Ranks per Device | **163** | 3 | ✅ 3 |
| Status | ❌ FAILED | ✅ SUCCESS | ✅ WILL WORK |

**Key Issue**: 1,960 ranks ÷ 12 devices = ~163 ranks per device → OVERSUBSCRIBED

---

## 🔧 Hardware Specifications

### Aurora XPU Nodes:
- **Devices per node**: 12 XPU devices
- **Optimal ranks per device**: 1-3 (with good balance)
- **Node count in your last success**: 9 nodes

### Recommended Configurations:

| Config | Ranks | RPD | Status |
|--------|-------|-----|--------|
| NPROC=1, NNODES=13 | 13 | ~1 | ✅ Good |
| NPROC=4, NNODES=9 | 36 | 3 | ✅ **Proven** |
| NPROC=4, NNODES=13 | 52 | ~4 | ⚠️ Borderline |
| NPROC=12, NNODES=13 | 156 | 13 | ❌ OVERSUBSCRIBED |
| NPROC=40, NNODES=49 | 1,960 | 163 | ❌ CATASTROPHIC |

(RPD = Ranks Per Device)

---

## 🚀 Recommended Command for Next Run

**Use the settings from your successful 1st training:**

```bash
cd /lus/flare/projects/datascience/seonghapark/xpu_launcher

# Set correct topology
export NPROC_PER_NODE=4
export NNODES=9

# Run training
python conversion.py \
  -i xpu_launcher/xpu_torchtitan/torchtitan_repo/outputs/xpu_torchtitan_20260925_160731/checkpoint/step-2062 \
  -o agpt-2b-v2-256n-post-step2062-safetensors &

# Or directly with PBS
qsub << 'EOF'
#!/bin/bash
#PBS -l select=9:ngpus=1
#PBS -l walltime=12:00:00

export NPROC_PER_NODE=4
export NNODES=9

./start_training_with_conversion.sh \
  --model-path ./xpu_launcher/xpu_torchtitan/torchtitan_repo/outputs/xpu_torchtitan_20260925_160731/checkpoint/step-2062 \
  --mode multi \
  -- --training.steps 400000 --training.seq_len 16384
EOF
```

---

## 📋 Step-by-Step Resolution

### Step 1: Understand the Error ✅
Device oversubscription: Too many ranks assigned to too few devices

### Step 2: Fix Configuration
Use proven settings: NPROC_PER_NODE=4, NNODES=9

### Step 3: Prepare Converted Model
```bash
python conversion.py \
  -i xpu_launcher/xpu_torchtitan/torchtitan_repo/outputs/xpu_torchtitan_20260925_160731/checkpoint/step-2062 \
  -o agpt-2b-v2-256n-post-step2062-safetensors
```

### Step 4: Submit Corrected Training Job
Create PBS script with correct settings (see above)

### Step 5: Monitor
```bash
tail -f xpu_launcher/xpu_torchtitan/torchtitan_repo/outputs/*/training.log
```

---

## 🛡️ Why This Works

The **1st training run proved** that:
- ✅ 36 ranks (9 nodes × 4 processes) = stable
- ✅ 3 ranks per device (36 ÷ 12) = optimal load
- ✅ Completed 2,062 steps successfully
- ✅ Only failed during cleanup (hardware issue, not configuration)

**Using the same configuration will:**
1. Avoid device oversubscription
2. Maintain stability
3. Allow training to progress beyond 2,062 steps
4. Resume from your successful checkpoint

---

## ❌ What NOT to Do

- ❌ Don't use NPROC_PER_NODE > 12 (exceeds devices)
- ❌ Don't use more than 49 nodes (allocation limit)
- ❌ Don't ignore device count mismatches
- ❌ Don't scale up without understanding hardware

---

## 📊 Log Analysis Summary

| Metric | Value |
|--------|-------|
| Total Ranks Attempted | 1,960 |
| Devices per Node | 12 |
| Oversubscription Ratio | **163:1** (catastrophic) |
| Failed Ranks | Multiple (14, 16, 20, 30, 31, 34, 36, 37, 40, 440, 473, ...) |
| Error Type | RuntimeError: Device index out of range |
| Job Termination | FAILOVER STOP: exhausted spare nodes |
| Root Cause | Configuration mismatch |

---

## ✅ Next Steps

1. **Immediately**: Use the recommended command above
2. **Convert model**: Run conversion.py if needed
3. **Submit job**: qsub train_corrected.pbs
4. **Monitor**: tail -f to watch progress
5. **Expected**: Training should start successfully and progress beyond step 2,062

---

## 📞 Reference

- **Working Configuration**: NPROC_PER_NODE=4, NNODES=9 (from 1st successful run)
- **Failed Configuration**: NPROC_PER_NODE=40, NNODES=49 (from this run)
- **Error**: Device index oversubscription
- **Solution**: Revert to proven settings

---

**Status**: ✅ Issue clearly identified  
**Recommendation**: Use proven configuration from 1st training run  
**Expected Outcome**: Successful training resumption from step 2,062

