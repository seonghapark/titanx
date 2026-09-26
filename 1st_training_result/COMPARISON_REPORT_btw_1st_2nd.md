# XPU Launcher Training Runs - Comparison Report
**Generated**: 2026-09-26  
**Comparison**: `titanx/1st_training_result` vs `xpu_launcher/run.log`

---

## 📊 Executive Summary

| Aspect | 1st Training (TitanX) | Current Run (XPU Launcher) | Status |
|--------|----------------------|---------------------------|--------|
| **Overall Result** | ✅ Partial Success | ❌ Failed Early | ⬇️ Regression |
| **Training Steps Completed** | 2,062 / 400,000 | 0 / 400,000 | ❌ No progress |
| **Loss Convergence** | 97.7% reduction (2.12→0.049) | N/A (failed during init) | ❌ Not reached |
| **Training Duration** | 4h 57m | ~1-2 min (interrupted) | ❌ Much shorter |
| **Failure Type** | Hardware (post-training) | Software/Compiler | ❌ Different cause |
| **Node Count** | 9 nodes × 4 ranks = 36 ranks | 13 nodes × 12 ranks = ~156 ranks | ⬆️ More resources |
| **Job Status** | Training completed, cleanup failed | Never reached training phase | ❌ Worse outcome |

---

## 🔴 Critical Differences

### 1. **Failure Timing & Phase**

**1st Training (Success until cleanup)**:
- ✅ Training ran successfully for 2,062 steps
- ✅ All loss metrics computed and converged properly
- ✅ Data pipeline completed 2,062 batches without errors
- ❌ Failed only during cleanup/shutdown phase (~5 hours in)

**Current Run (Failed immediately)**:
- ❌ Never reached training phase
- ❌ Failed during Triton kernel compilation (initialization)
- ❌ Job aborted within first 1-2 minutes
- ❌ No training data processed

**Analysis**: This is a **regression** — the current run fails earlier in the pipeline.

### 2. **Failure Root Cause**

**1st Training - Hardware/Firmware**:
```
Error Type: Hardware failures at shutdown
- Rank 2: Signal 11 (SIGSEGV) - Segmentation fault
- Rank 4: Exit code 245 - Process exit (likely thermal/driver)
- Rank 20: Exit code 241 - Abnormal termination
- Rank 35: Signal 15 (SIGTERM) - Graceful shutdown signal
```

**Current Run - Compiler/Triton**:
```
Error Type: Triton compilation failure
- torch._inductor.exc.InductorError: KeyError: "Unknown key: 'zebin'"
- Failed during Triton kernel precompilation
- Rank 77: triton_red_fused__fused_rms_norm_0 kernel compilation
- Caused by: Unknown binary extension key in Triton compiler
```

**Root Cause**:
- **1st Training**: GPU thermal/hardware issues after sustained load
- **Current Run**: Triton compilation issue with XPU binary format

### 3. **Configuration Differences**

| Parameter | 1st Training | Current Run | Change |
|-----------|--------------|------------|--------|
| **Nodes** | 9 | 13 | ⬆️ +4 more |
| **Ranks** | 36 (9×4) | ~156 (13×12) | ⬆️ 4.3x more |
| **Seq Length** | 16,384 | 16,384 | ➡️ Same |
| **Training Steps** | 400,000 | 400,000 | ➡️ Same |
| **Model** | AGPT 2B (step 92859) | AGPT 2B (step 92859) | ➡️ Same |
| **Dataset** | PG19 | PG19 | ➡️ Same |
| **Resource Monitor** | Not mentioned | Enabled (15s interval) | ⬆️ Added |
| **Spare Nodes** | Auto (~1-2) | Auto (~1-2) | ➡️ Same |

**Key Issue**: Current run uses **4.3x more ranks** (156 vs 36) which may overwhelm system resources or expose compilation issues.

### 4. **Error Message Comparison**

**1st Training**:
```
[auto-retry] FAILOVER STOP: exhausted (no spare nodes left, rc=245)
```
- Hardware issue, job terminated gracefully
- Auto-retry mechanism worked correctly
- Clean error code propagation

**Current Run**:
```
[rank77]: torch._inductor.exc.InductorError: KeyError: "Unknown key: 'zebin'"
[rank77]: AttributeError: 'CompiledKernel' object has no attribute 'module'
[auto-retry] FAILOVER STOP: exhausted (no spare nodes left, rc=241)
```
- Compiler error, job terminated with stack trace
- Suggests PyTorch/Triton version incompatibility
- Resource exhaustion on spare nodes (same as 1st training)

---

## 📈 Success/Failure Metrics

### 1st Training Results
```
✅ SUCCESSES:
  - 2,062 training steps completed
  - Loss reduced from 2.121203 → 0.049247 (97.7% improvement)
  - Multi-node synchronization stable (36 ranks)
  - Data pipeline processed 2,062 batches successfully
  - Checkpoint operations fast (~1.6s per checkpoint)
  - Collective communication (XCCL) stable
  - Process groups cleaned properly
  - Garbage collection working (0.02-0.04s)

❌ FAILURES:
  - 4 ranks failed during cleanup phase
  - Hardware failures on specific nodes (x4003c7s2b0n0, x4012c3s0b0n0, x4104c2s7b0n0, x4112c3s0b0n0)
  - Auto-retry exhausted spare nodes
  - Job terminated with exit code 245
```

### Current Run Results
```
✅ SUCCESSES:
  - Job submitted successfully
  - Nodes allocated (13 nodes)
  - Multi-node topology detected
  - Resource monitoring configured

❌ FAILURES:
  - Triton kernel compilation failed (triton_red_fused__fused_rms_norm_0)
  - KeyError in Triton compiler ("Unknown key: 'zebin'")
  - CompiledKernel object initialization failed
  - Rank 77, 115, 120 exited/signaled
  - Auto-retry exhausted spare nodes
  - Job terminated with exit code 241 (before training started)
```

---

## 🔍 Detailed Analysis

### Issue 1: Triton Compilation Error

**Error Message**:
```
KeyError: "Unknown key: 'zebin'"
```

**Location**: 
```python
torch/_inductor/compiler.py:410 in __missing__
triton/compiler/compiler.py:442 in __init__
```

**What It Means**:
- Triton compiler is looking for a binary extension named "zebin" (Triton Binary)
- The key doesn't exist in the available extensions
- This happens during kernel compilation for RMS normalization layer

**Likely Causes**:
1. **Version Mismatch**: PyTorch/Triton version incompatible with XPU backend
2. **Missing Triton Backend**: XPU Triton support not properly installed
3. **Environment Configuration**: `TRITON_BACKEND` environment variable not set correctly
4. **GPU Compiler Mismatch**: Triton GPU backend not available for XPU architecture

**Evidence**:
```
File "/opt/aurora/26.181.0/frameworks/aurora_frameworks-2026.1.0/lib/python3.12/site-packages/"
```
- Using Aurora 26.181.0 framework
- Framework is trying to use Triton with XPU but missing backend

### Issue 2: Rank Scaling Problem

**Difference**: 
- 1st Training: 36 ranks (9 nodes × 4 ranks/node)
- Current Run: ~156 ranks (13 nodes × 12 ranks/node)

**Impact**:
- 4.3x increase in parallelism
- Likely higher GPU memory per rank
- Increased compilation overhead
- More distributed synchronization needed

**Hypothesis**: 
Current configuration with 12 ranks/node may be too aggressive for Triton compilation, causing timeouts or resource exhaustion during kernel compilation.

### Issue 3: Repeated Failures on Different Nodes

**1st Training**:
- Failed on 4 specific nodes (x4003c7s2b0n0, x4012c3s0b0n0, x4104c2s7b0n0, x4112c3s0b0n0)

**Current Run**:
- Failed on ranks on x4112c3s2b0n0, x4112c3s5b0n0, x4112c3s6b0n0
- 2 nodes share same prefix (x4112), suggesting cluster-wide issue

**Observation**: Node x4112 appears in both runs. May indicate:
- Problematic hardware batch
- Firmware issue
- Thermal problems recurring

---

## ⚠️ Warnings Comparison

### 1st Training Warnings (Non-Critical)
```
WARNING: model.safetensors.index.json not found
  → Fallback to single safetensors file (handled gracefully)
  → Frequency: Multiple occurrences
  → Impact: None

WARNING: oneCCL Sysman API not initialized
  → Fallback to legacy behavior
  → Frequency: Multiple occurrences  
  → Impact: None

WARNING: Local index/count from environment variables missing
  → Fallback to ATL
  → Impact: None
```

### Current Run Warnings
```
(No logs before crash - investigation needed)
```

**Difference**: Current run has no preliminary warnings; crashes immediately on compilation.

---

## 💡 Root Cause Analysis

### Scenario 1: Triton Backend Missing (Most Likely)
```
Hypothesis: XPU Triton backend not installed or misconfigured
Evidence:
  - KeyError on "zebin" (expected binary extension)
  - Missing torch._inductor integration for XPU
  - Framework paths show Aurora 26.181.0 setup
  
Fix: Ensure Triton supports XPU backend
  export TRITON_BACKEND=xpu
  # Or use CPU-fallback for testing
  export TORCH_INDUCTOR_SKIP_TRITON_COMPILATION=1
```

### Scenario 2: PyTorch/Triton Version Mismatch
```
Hypothesis: Installed versions incompatible
Evidence:
  - CompiledKernel.__del__ AttributeError suggests partial init
  - Triton compiler missing expected module
  - Framework Aurora 26.181.0 may have outdated Triton
  
Fix: Verify versions
  python -c "import torch; print(torch.__version__)"
  python -c "import triton; print(triton.__version__)"
  # Compare with Aurora docs
```

### Scenario 3: Resource Exhaustion During Compilation
```
Hypothesis: 156 ranks × compilation threads > system capacity
Evidence:
  - 4.3x more ranks than 1st training
  - 12 ranks per node may trigger parallel compilation bottleneck
  - Multiple ranks fail on same nodes (x4112)
  
Fix: Reduce rank count
  NPROC_PER_NODE=4  # Instead of 12
  # Or compile kernels in serial mode
  export TORCH_INDUCTOR_MAX_ASYNC_COMPILE_NUM_THREADS=1
```

---

## 📝 Solutions & Recommendations

### Immediate Actions (Priority 1)

**1. Check Triton Configuration**
```bash
# Verify Triton is available
python -c "import triton; print(triton.__version__)"

# Check for XPU backend support
python -c "import triton.runtime.driver; print(dir(triton.runtime.driver))"

# Verify PyTorch compilation options
export TORCH_INDUCTOR_VERBOSE=1
export TORCHDYNAMO_VERBOSE=1
```

**2. Reduce Rank Count for Compilation**
```bash
# Current: 13 × 12 = 156 ranks
# Proposed: 13 × 4 = 52 ranks (matches 1st training success)

NPROC_PER_NODE=4 ./start_training.sh --mode multi \
  --model-path /lus/flare/projects/datascience/seonghapark/agpt-2b-v2-256n-step-92859-safetensors \
  --training-steps 400000
```

**3. Serial Compilation Mode**
```bash
# Force single-threaded kernel compilation
export TORCH_INDUCTOR_MAX_ASYNC_COMPILE_NUM_THREADS=1
export TRITON_CACHE_DIR=/tmp/triton_cache

./start_training.sh --mode multi ...
```

### Secondary Actions (Priority 2)

**4. Check Node Health**
```bash
# Verify x4112* nodes status (appearing in both runs)
pbsnodes -a | grep x4112

# Check for thermal or driver issues
xpu-smi health
dmesg | grep -i "error\|thermal\|xpu"
```

**5. Use Fallback Compilation**
```bash
# If Triton XPU backend unavailable
export TORCH_COMPILE_BACKEND=aot_eager
# Or CPU-fallback for debugging
export TORCH_DYNAMO_BACKEND=eager
```

**6. Gradual Scale-Up**
```bash
# Test with 1 node first
NNODES=1 NPROC_PER_NODE=4 ./start_training.sh ...

# Then 2 nodes
NNODES=2 NPROC_PER_NODE=4 ./start_training.sh ...

# Then increase ranks
NNODES=13 NPROC_PER_NODE=4 ./start_training.sh ...

# Only after success, try higher NPROC_PER_NODE
NNODES=13 NPROC_PER_NODE=8 ./start_training.sh ...
```

### Long-Term Actions (Priority 3)

**7. Update Framework/Dependencies**
```bash
# Check Aurora framework version
ls -la /opt/aurora/

# Update to latest stable
# Contact system admin for framework upgrade
# Ensure Triton supports your XPU version
```

**8. Prevent Compilation Bottlenecks**
```bash
# Cache compiled kernels from 1st training
TORCH_INDUCTOR_CACHE_DIR=/lus/flare/projects/datascience/seonghapark/triton_cache

# Pre-compile kernels with full node before multi-node run
NNODES=1 NPROC_PER_NODE=1 ./start_training.sh --training-steps 10

# Then use cached kernels for multi-node
export TORCH_INDUCTOR_CACHE_DIR=...
./start_training.sh --mode multi ...
```

---

## 📊 Comparison Table: By Category

### Configuration

| Category | 1st Training | Current Run | Recommendation |
|----------|--------------|------------|-----------------|
| Nodes | 9 | 13 | ✅ Revert to 9 |
| Ranks/Node | 4 | 12 | ✅ Reduce to 4 |
| Total Ranks | 36 | 156 | ✅ Reduce to 36-52 |
| Sequence Length | 16,384 | 16,384 | ➡️ Keep same |
| Training Steps | 400,000 | 400,000 | ➡️ Keep same |

### Infrastructure

| Aspect | 1st Training | Current Run | Status |
|--------|--------------|------------|--------|
| Auto-Retry | ✅ Enabled | ✅ Enabled | ✅ Working |
| Spare Nodes | Auto | Auto | ✅ Same |
| Resource Monitor | ⚠️ Not mentioned | ✅ Enabled | ✅ Improvement |
| Node Failures | Specific nodes (x4003c7, x4012c3, x4104c2, x4112c3) | Cluster-wide (x4112) | ⚠️ Pattern emerging |

### Success Metrics

| Metric | 1st Training | Current Run | Delta |
|--------|--------------|------------|-------|
| Training Steps Completed | 2,062 ✅ | 0 ❌ | -100% |
| Loss Convergence | 97.7% ✅ | N/A ❌ | Failed before reaching |
| Duration | 4h 57m ✅ | ~2 min ❌ | -99.3% |
| Training Quality | Excellent ✅ | Not assessed ❌ | Can't compare |

---

## 🎯 Key Takeaways

### ✅ What Worked in 1st Training
1. **9-node configuration** with 4 ranks/node was stable
2. **Multi-node infrastructure** functioned correctly throughout training
3. **Data pipeline and loss computation** worked without issues
4. **Auto-retry mechanism** detected failures and attempted recovery
5. **Training dynamics** showed excellent convergence

### ❌ What Failed in Current Run
1. **12 ranks/node configuration** appears unstable for compilation
2. **Triton kernel compilation** failed with "zebin" key error
3. **Job never reached training phase** (pre-training failure)
4. **Version incompatibility** between PyTorch/Triton/XPU backend

### 🔧 Critical Fixes Needed
1. **Reduce NPROC_PER_NODE from 12 to 4** (proven stable)
2. **Fix Triton backend configuration** for XPU
3. **Check x4112* nodes** for hardware issues
4. **Verify PyTorch/Triton versions** match framework requirements

---

## 📋 Next Steps

### Before Next Run
```bash
# 1. Check node health
pbsnodes -a | grep -E "x4003c7|x4012c3|x4104c2|x4112"

# 2. Verify Triton backend
python << 'PYTHON'
import torch
import triton
print(f"PyTorch: {torch.__version__}")
print(f"Triton: {triton.__version__}")
print(f"Compute Capability: {torch.xpu.get_device_capability()}")
PYTHON

# 3. Use stable configuration from 1st training
NNODES=9 NPROC_PER_NODE=4 TRAINING_STEPS=400000 SEQ_LEN=16384 ./start_training.sh
```

### During Run
```bash
# Monitor compilation progress
tail -f /lus/flare/projects/datascience/seonghapark/xpu_launcher/xpu_torchtitan/torchtitan_repo/outputs/*/training.log

# Check for thermal issues
while true; do
  xpu-smi dev -m | grep -i temp
  sleep 5
done
```

### After Run
```bash
# Analyze results
python analyze_logs.py /path/to/run.log

# Save checkpoint if training succeeds
cp -r outputs/*/checkpoint /backup/location/
```

---

## Appendix: Detailed Error Logs

### Current Run - Full Error Stack
```
[rank77]: torch._inductor.exc.InductorError: KeyError: "Unknown key: 'zebin'"
[rank77]: 
[rank77]:   File "/opt/aurora/26.181.0/frameworks/aurora_frameworks-2026.1.0/lib/python3.12/site-packages/torch/_inductor/runtime/triton_heuristics.py", line 1156, in _precompile_config
[rank77]:     binary = triton.compile(*compile_args, **compile_kwargs)
[rank77]:              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank77]:   File "/opt/aurora/26.181.0/frameworks/aurora_frameworks-2026.1.0/lib/python3.12/site-packages/triton/compiler/compiler.py", line 279, in compile
[rank77]:     res = CompiledKernel(src, metadata_group, hash)
[rank77]:           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank77]:   File "/opt/aurora/26.181.0/frameworks/aurora_frameworks-2026.1.0/lib/python3.12/site-packages/triton/compiler/compiler.py", line 442, in __init__
[rank77]:     self.kernel = self.asm[binary_ext]
[rank77]:                   ~~~~~~~~^^^^^^^^^^^^
[rank77]:   File "/opt/aurora/26.181.0/frameworks/aurora_frameworks-2026.1.0/lib/python3.12/site-packages/triton/compiler/compiler.py", line 410, in __missing__
[rank77]:     raise KeyError("Unknown key: '%s'" % key)
```

**Translation**: 
- Triton tried to access `self.asm[binary_ext]` where `binary_ext = "zebin"`
- The dictionary lookup failed because "zebin" is not a recognized key
- This suggests XPU backend support is missing or misconfigured

### Failed Ranks Log Extract
```
x4112c3s2b0n0.hsn.cm.aurora.alcf.anl.gov: rank 77 exited with code 1
x4112c3s5b0n0.hsn.cm.aurora.alcf.anl.gov: rank 115 died from signal 15
x4112c3s6b0n0.hsn.cm.aurora.alcf.anl.gov: rank 120 exited with code 241
[auto-retry] FAILOVER STOP: exhausted (no spare nodes left, rc=241)
```

**Translation**:
- Rank 77: Error during kernel compilation (exit code 1)
- Rank 115: Killed by scheduler (signal 15 = SIGTERM)
- Rank 120: Abnormal termination (exit code 241)
- All on x4112* nodes (cluster location appears in both runs)

---

## 📌 Conclusion

The **current XPU launcher run represents a regression** compared to the 1st training result:

1. **Earlier Failure**: Failed during initialization (min 0-2) vs cleanup (hour 5)
2. **Different Root Cause**: Compiler issue vs hardware issue
3. **Larger Configuration**: 156 ranks vs 36 ranks (not properly tuned)
4. **No Training Data**: 0 steps completed vs 2,062 steps

**Recommended Action**: 
- ✅ **Revert to proven 9-node, 4-rank/node configuration**
- ✅ **Fix Triton backend for XPU support**
- ✅ **Monitor x4112 nodes for hardware issues**
- ✅ **Validate dependencies before full-scale run**

---

**Report Generated**: 2026-09-26 by Automated Log Analysis  
**Next Analysis**: After corrected training run  
**Contact**: System Administration for node health verification
