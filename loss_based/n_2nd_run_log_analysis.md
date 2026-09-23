# Run Log Analysis - Errors and Solutions

## Summary
Training started successfully, HF model loaded, but failed during the state_dict conversion with a **tensor sharding/redistribution error**.

---

## CRITICAL ERROR

### Error: "Cannot unflatten unevenly sharded tensor"

**Type**: RuntimeError (Tensor Sharding Incompatibility)

**Frequency**: All 48 ranks failed identically

**Error Message**:
```
RuntimeError: Cannot unflatten unevenly sharded tensor: 
output dimension 0 (size 16) is not evenly divisible by mesh dimension 0 (size 48). 
Please redistribute the tensor before this operation.

Sharding propagation failed for aten.view.default(
  Spec(f32[2048, 2048](S(0))), [16, 64, 2, 2048]
) on DeviceMesh((fsdp=48), 'xpu', stride=(1,)))
```

**Stack Trace Location**:
```
torchtitan/models/llama3/state_dict_adapter.py:52 in _permute()
  w.view(n_heads_arg, dim1 // n_heads_arg // 2, 2, dim2)
```

**Root Cause**:
The error occurs during HF model loading in the state_dict adapter. The code tries to reshape a tensor from `[2048, 2048]` to `[16, 64, 2, 2048]`, but:

1. The tensor is sharded across 48 ranks with a single sharding dimension (S(0))
2. The output dimension 0 (size 16) cannot be evenly divided by the mesh dimension 0 (size 48)
3. **16 is not divisible by 48** → Cannot redistribute tensor for this view operation
4. The `.view()` operation requires the tensor to be resharded before it can execute

---

## WARNINGS

### Warning: "model.safetensors.index.json not found"

**Frequency**: ~40+ instances across ranks

**Message**:
```
WARNING - model.safetensors.index.json not found at hf_assets_path: 
.../model.safetensors.index.json
Defaulting to saving a single safetensors file if checkpoint is saved in HF format
```

**Severity**: LOW (Informational)

**Root Cause**: 
The HF model directory doesn't have the standard `model.safetensors.index.json` metadata file, which would indicate a sharded model. Instead, it has individual safetensors files.

**Impact**: 
None on loading (the model loads fine). Only affects saving - if checkpoints are saved in HF format, they'll be saved as a single file instead of sharded files.

---

## Timeline of Events

1. ✓ **16:56:59** - All 48 ranks initialized successfully
2. ✓ **17:57:32** - HF model loading started: "Loading HF model from --model.hf_assets_path"
3. ✓ **17:57:33** - Checkpoint initialization: "Trainer is initialized..."
4. ✓ **17:57:33** - Model loading message: "Loading the checkpoint from .../agpt-2b-v2-256n-step-92859-safetensors"
5. ✗ **17:57:33** - **CRASH** during state_dict conversion in `_permute()` method

**Duration**: ~34 seconds from start to crash (mostly model I/O and initialization)

---

## Why This Error Occurred

### Technical Details

The problem is in the **state_dict adapter for Llama3 models**. When loading HF checkpoints, the code needs to:

1. Load tensor weights from HF safetensors format
2. Permute/reshape tensors to match TorchTitan's internal format
3. Handle RoPE (Rotary Position Embedding) weights specially

The issue occurs at step 2 - the `_permute()` function in `llama3/state_dict_adapter.py`:

```python
# Line 52 in state_dict_adapter.py
def _permute(value, n_heads):
    # Trying to reshape [2048, 2048] → [16, 64, 2, 2048]
    w.view(n_heads_arg, dim1 // n_heads_arg // 2, 2, dim2)
```

### Why It Fails

The tensor comes in as:
- **Shape**: `[2048, 2048]` (original HF format)
- **Sharding**: Replicated across 48 ranks with FSDP sharding on dimension 0
- **Sharded format**: `f32[2048, 2048](S(0))` - Sharded on first dimension

The reshape requires:
- **New shape**: `[16, 64, 2, 2048]`
- **First dimension**: 16

**The problem**: 
- 2048 (original dim 0) sharded across 48 ranks = each rank has 2048/48 elements per step
- Trying to view as 16 = needs to redistribute to have 16 as the sharding unit
- **16 is not divisible by 48** → impossible redistribution

---

## Solution

The state_dict adapter needs to **reshard the tensor before reshaping** or **reshape before sharding**.

### Option 1: Reshard to Replicated Before View (Recommended)

**File**: `torchtitan/models/llama3/state_dict_adapter.py`

**Change at line 50-52**:
```python
def _permute(self, value: torch.Tensor, n_heads: int) -> torch.Tensor:
    # Convert distributed tensor to replicated (non-sharded)
    if isinstance(value, DTensor):
        value = value.to_local()  # Gather to all ranks
    
    # Now perform the reshape on the local tensor
    w = value.view(
        n_heads,
        value.shape[0] // n_heads // 2,
        2,
        value.shape[1],
    )
    return w.transpose(1, 2).reshape(value.shape)
```

### Option 2: Use Different Sharding Strategy

Ensure tensor is replicated across all ranks before state_dict conversion:

```python
from torch.distributed.tensor import Replicate, distribute_tensor

def _permute(self, value: torch.Tensor, n_heads: int) -> torch.Tensor:
    if isinstance(value, DTensor):
        # Redistribute to replicate on all ranks
        value = value.redistribute([Replicate()])
    
    # Perform reshape on replicated tensor
    w = value.view(...)
    ...
```

### Option 3: Modify Checkpoint Loading to Avoid Distributed Tensors

In `checkpoint.py`, don't use distributed tensors during HF load:

```python
# In dcp_load() method, before calling sd_adapter.to_hf()
if from_hf:
    # Load as local tensor (non-distributed)
    state_dict = load_hf_checkpoint(...)  # Load locally
    hf_state_dict = self.sd_adapter.to_hf(state_dict)  # Convert local
    # Then broadcast/synchronize across ranks
```

---

## Immediate Action

The most straightforward fix is **Option 1**: Convert distributed tensors to local before the view operation.

**Changes needed**:
1. Import `from torch.distributed.tensor import DTensor`
2. In `_permute()` method, check if tensor is distributed and convert to local
3. Perform reshape on local tensor
4. Re-distribute if needed

**Risk Level**: LOW - This is a safe operation (gather to all ranks is normal in distributed training)

**Testing**:
```bash
LOSS_STD_TERMINATION_ENABLED=1 \
MODEL=/path/to/hf/model \
./run_train_torchtitan.sh single
```

Should load model without errors and start training.

---

## Why This Happens Now

The refactoring you did:
- ✓ Removed DCP checkpoint support
- ✓ Forced HF model loading
- **Exposed an existing bug** in the HF state_dict adapter for Llama3

The adapter was written for a different distribution strategy or didn't account for FSDP-sharded tensors during the reshape operation.

---

## Summary Table

| Issue | Type | Severity | Root Cause | Solution |
|-------|------|----------|------------|----------|
| Cannot unflatten tensor | RuntimeError | CRITICAL | Tensor dimension mismatch during reshape with FSDP sharding | Reshard to replicated before view |
| model.safetensors.index.json missing | WARNING | LOW | HF model uses single file instead of sharded | No action needed (saves as single file) |

---

## Next Steps

1. **Fix state_dict adapter** (Option 1 recommended)
2. **Test on single node** first
3. **Test on multi-node** (48 ranks)
4. **Monitor for other tensor operations** that might have similar issues
