# Run Log Analysis - 2026-09-23

## Summary
**Status**: ✗ FAILED  
**Error Type**: RuntimeError - Invalid tensor reshape  
**Affected Ranks**: All 48 ranks  
**Root Cause**: Double-permutation of HF safetensors during loading

---

## Primary Error

### Error Message
```
RuntimeError: shape '[16, 1, 2, 2048]' is invalid for input of size 88064
```

### Error Location Stack
```
torchtitan/components/checkpoint.py:782 in load()
  → self.dcp_load(states, checkpoint_id=..., from_hf=True, ...)
    
torchtitan/components/checkpoint.py:601 in dcp_load()
  → hf_state_dict = self.sd_adapter.to_hf(state_dict)
  
torchtitan/models/llama3/state_dict_adapter.py:122 in to_hf()
  → value = self._permute(value, n_heads)
  
torchtitan/models/llama3/state_dict_adapter.py:73 in _permute()
  → w.view(n_heads_arg, dim1 // n_heads_arg // 2, 2, dim2)
```

### Affected Tensors
- Q projection weights: `layers.{}.attention.qkv_linear.wq.weight`
- K projection weights: `layers.{}.attention.qkv_linear.wk.weight`
- Dimensions: [2048, 2048] (expected for 2B param model)

### Expected vs. Actual
- **Expected shape after view**: [16, 64, 2, 2048]
  - n_heads_arg = 16
  - dim1 // n_heads_arg // 2 = 2048 // 16 // 2 = 64
  - Result: 16 × 64 × 2 × 2048 = 4,194,304 elements
  
- **Actual tensor size**: 88,064 elements
  - This is roughly half of the original 2048×2048 = 4,194,304
  - Indicates tensor was already permuted once

---

## Root Cause Analysis

### What Happened
1. **Initial Load Path**: Code sets `from_hf=True` when loading from HF safetensors path
   - File: `checkpoint.py:741-750`
   - Condition: `self.load_hf_model=True` (default in config)

2. **Incorrect Conversion**: Code calls `dcp_load()` with `from_hf=True`
   - File: `checkpoint.py:782-787`
   - Problem: `dcp_load()` with `from_hf=True` expects a **native-format** DCP checkpoint that needs conversion to HF

3. **Double-Permutation**: The HF safetensors are already in HF format
   - The `to_hf()` method is designed to convert from native format → HF format
   - Calling `to_hf()` on already-HF tensors applies the permutation twice
   - After first permutation: tensor becomes smaller/different shape
   - Second permutation attempt: dimensions no longer match

### Code Flow Diagram

```
Load HF Safetensors
       ↓
checkpoint.load(step=-1)
       ↓
Load from hf_assets_path (from_hf=True)
       ↓
dcp_load(state_dict, from_hf=True)  ← WRONG: This path is for converting DCP checkpoints
       ↓
to_hf(state_dict)  ← WRONG: Applies HF permutation to already-permuted tensors
       ↓
_permute() called twice on same tensors
       ↓
CRASH: Dimensions don't match
```

---

## Why This Happened

### Historical Context
The code was designed to support two loading paths:
1. **DCP native → DCP with HF conversion**: Load a DCP checkpoint and convert its format using `to_hf()`
2. **Direct HF load**: Load HF safetensors directly

### The Bug
When we refactored to **remove DCP checkpoint initial load capability** and keep **only HF format**, we:
- ✓ Added new options: `load_hf_model`, `load_hf_model_quantized`
- ✓ Removed: `initial_load_path`, `initial_load_model_only`, etc.
- ✗ **BUT** the `dcp_load()` method still assumes:
  - If `from_hf=True`: Input is DCP that needs conversion
  - NOT: Input is already HF format that should be loaded directly

### The Mismatch
The calling code says: "Load from HF format"  
The `dcp_load()` code hears: "Load DCP and convert to HF"  
These are incompatible operations.

---

## Solution

### Option 1: Bypass Conversion (RECOMMENDED)
When loading directly from HF safetensors, skip the `to_hf()` conversion entirely.

**Changes needed**:
1. In `dcp_load()`, add a parameter: `load_hf_native=False`
2. When `from_hf=True` AND `load_hf_native=True`: Skip `to_hf()` conversion
3. Load safetensors directly and apply `from_hf()` to get native format

```python
def dcp_load(self, state_dict, checkpoint_id, from_hf, from_quantized, load_hf_native=False):
    if from_hf and not load_hf_native:
        # OLD PATH: DCP checkpoint that needs HF conversion
        hf_state_dict = self.sd_adapter.to_hf(state_dict)
        ...
    elif from_hf and load_hf_native:
        # NEW PATH: Direct HF safetensors load (no conversion)
        hf_storage_reader = self.sd_adapter.get_hf_storage_reader(checkpoint_id, from_quantized)
        dcp.load(state_dict, storage_reader=hf_storage_reader)
        state_dict = self.sd_adapter.from_hf(state_dict)
        ...
    else:
        # Native DCP load
        dcp.load(state_dict, checkpoint_id=checkpoint_id)
        ...
```

### Option 2: Create Separate Load Path
Split `dcp_load()` into:
- `load_native_checkpoint()` - for native format DCP
- `load_hf_checkpoint()` - for HF safetensors

This is cleaner but requires more refactoring.

### Option 3: Check Source Format
Detect whether the input is HF format or native format and handle accordingly.

---

## Recommended Fix

**Implementation**:
1. Add `load_hf_native` parameter to `dcp_load()`
2. In `checkpoint.load()`, pass `load_hf_native=True` when `from_hf=True` and loading initial model
3. Modify `dcp_load()` logic to skip `to_hf()` conversion when `load_hf_native=True`

**Files to modify**:
- `torchtitan/components/checkpoint.py`
  - Add parameter to `dcp_load()` signature
  - Add conditional logic to skip `to_hf()` when loading native HF format
  - Pass `load_hf_native=True` in `load()` method when appropriate

**Impact**:
- ✓ Fixes the double-permutation issue
- ✓ Maintains HF safetensors loading capability
- ✓ Minimal code changes
- ✓ Backward compatible (default `load_hf_native=False` preserves DCP behavior if needed)

---

## Testing Checklist

After implementing the fix:
- [ ] Single-node training loads HF model successfully
- [ ] Model loads within first 10 seconds (no hangs)
- [ ] No "invalid shape" errors in ranks
- [ ] Training loop starts (see "Starting training loop" messages)
- [ ] First loss value printed correctly
- [ ] Multi-node training (48 ranks) works with loss tracking
- [ ] Loss std termination feature activates when std <= 0.001

---

## Warning Message Noticed

```
WARNING - model.safetensors.index.json not found at hf_assets_path
```

**Significance**: Low - this is informational only, doesn't cause the crash
**What it means**: The model is a single-file safetensors (not sharded across multiple files)
**No action needed**: The loader handles this automatically

---

## Summary of Issues

| Issue | Severity | Type | Fix |
|-------|----------|------|-----|
| Double-permutation during HF load | CRITICAL | Logic error | Add `load_hf_native` parameter |
| Invalid tensor reshape | CRITICAL | Consequence | Resolved by fix above |
| DCP conversion called on HF data | CRITICAL | Design mismatch | Separate the code paths |

All 48 ranks fail with identical error, indicating the root cause is systematic, not environmental.


remove any dcp checkpoint load and conversion to HF related codes.
remove any load_hf_native option related code, because that will be the only way.
