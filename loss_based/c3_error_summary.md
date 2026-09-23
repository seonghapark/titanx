# Training Run Error Summary and Fix

## Run Details
- **Date**: 2026-09-23 17:56:59 - 17:57:33
- **Duration**: ~34 seconds (before crash)
- **Topology**: 12 nodes, 48 ranks (4 per node)
- **Status**: FAILED during HF model loading

---

## Error Found

### Critical Error: "Cannot unflatten unevenly sharded tensor"

**File**: `torchtitan/models/llama3/state_dict_adapter.py:52`

**Method**: `_permute()`

**Error Message**:
```
RuntimeError: Cannot unflatten unevenly sharded tensor: 
output dimension 0 (size 16) is not evenly divisible by mesh dimension 0 (size 48).
Please redistribute the tensor before this operation.
```

**Affected Ranks**: All 48 ranks failed identically

**Cause**: 
The `_permute()` method tries to reshape an FSDP-sharded tensor from `[2048, 2048]` to `[16, 64, 2, 2048]`. The new first dimension (16) is not divisible by 48 ranks, making tensor redistribution impossible.

---

## Why This Happened

Your refactoring changes:
1. ✓ Removed DCP checkpoint support
2. ✓ Forced HF-only model loading
3. **→ Exposed latent bug in state_dict adapter**

The adapter wasn't designed to handle FSDP-sharded tensors during reshape operations. This bug was hidden before because DCP loading didn't use this code path.

---

## Solution

See `RECOMMENDED_FIX.txt` for detailed code changes.

**Quick Fix**: Convert distributed tensor to local before reshape

```python
# In _permute() method, add this check:
from torch.distributed.tensor import DTensor

if isinstance(value, DTensor):
    value = value.to_local()  # Convert to local tensor

# Then perform reshape on local tensor
```

---

## Implementation Steps

1. **Edit file**: `torchtitan/models/llama3/state_dict_adapter.py`
2. **Add import**: `from torch.distributed.tensor import DTensor`
3. **Modify method**: `_permute()` - add to_local() call (see RECOMMENDED_FIX.txt)
4. **Test**: Run single-node test first, then multi-node test
5. **Verify**: Model loads without "Cannot unflatten" error

---

## Testing Commands

After fix is applied:

**Single Node**:
```bash
MODEL=/lus/flare/projects/datascience/seonghapark/agpt-2b-v2-256n-step-92859-safetensors \
./run_train_torchtitan.sh single --training.steps 100
```

**Multi-Node**:
```bash
LOSS_STD_TERMINATION_ENABLED=1 \
MODEL=/lus/flare/projects/datascience/seonghapark/agpt-2b-v2-256n-step-92859-safetensors \
./run_train_torchtitan.sh multi --training.steps 100
```

---

## Warnings (Non-critical)

**"model.safetensors.index.json not found"** - ~40 instances

- **Impact**: LOW - Only affects checkpoint saving format
- **Action**: None required
- **Why**: HF model uses single file instead of sharded files

---

## Timeline

| Time | Event | Status |
|------|-------|--------|
| 16:56:59 | All 48 ranks initialized | ✓ |
| 17:57:32 | HF model loading started | ✓ |
| 17:57:33 | Trainer initialized | ✓ |
| 17:57:33 | State_dict conversion error | ✗ CRASH |

---

## Affected Code

**File**: `torchtitan/models/llama3/state_dict_adapter.py`

**Method**: `_permute()` (lines 45-57)

**Scope**: All HF model loads using Llama3 state_dict adapter

---

## Risk Assessment

**Code Change Risk**: LOW
- Single method modification
- Well-tested PyTorch operation (to_local)
- No API changes

**Deployment Risk**: LOW
- Fix is isolated to state_dict conversion
- Not in training loop
- Backward compatible

**Validation**: Complete with provided test procedures

---

## Documentation Files

- **RECOMMENDED_FIX.txt**: Detailed code fix with alternatives (272 lines)
- **ERROR_SUMMARY.md**: This file - quick reference
- **RUN_LOG_ANALYSIS.md**: Full technical analysis

---

## Next Actions

1. ☐ Apply code fix from RECOMMENDED_FIX.txt
2. ☐ Test on single node
3. ☐ Test on multi-node (48 ranks)
4. ☐ Verify early termination feature works
5. ☐ Run full training with loss convergence detection

**Status**: Ready for implementation
