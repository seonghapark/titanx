Read a newly generated run.log. Find errors, warnings, why they heppened, and what are solutions to resolve the errors and warnings. Double check if there are unresolved errors from previous runs.

# Run Log Analysis v2 - 2026-09-23

## Summary
**Status**: ✗ FAILED  
**Error Type**: Missing key in checkpoint state_dict  
**Affected Ranks**: All 48 ranks  
**Root Cause**: HF state dict is empty after dcp.load() with HF storage reader

---

## Primary Error

### Error Message
```
RuntimeError: Missing key in checkpoint state_dict: layers.0.attention.qkv_linear.wk.weight
```

### Error Stack
```
torchtitan/components/checkpoint.py:755 in load()
  → self.hf_load(states, checkpoint_id=..., from_quantized=...)

torchtitan/components/checkpoint.py (hf_load method - line 593):
  → dcp.load(state_dict, storage_reader=hf_storage_reader)
  → state_dict = self.sd_adapter.from_hf(state_dict)
  → self.states[MODEL].load_state_dict(state_dict)

torch/distributed/checkpoint/state_dict_loader.py:281 in local_step():
  → local_plan = planner.create_local_plan()
  
torch/distributed/checkpoint/default_planner.py:495 in create_default_local_load_plan():
  → RuntimeError: Missing key in checkpoint state_dict: layers.0.attention.qkv_linear.wk.weight
```

### Location in Code
The error occurs in the DCP planner when trying to load the model state dict. The key `layers.0.attention.qkv_linear.wk.weight` is expected (it's in the native TorchTitan format), but it's missing from the state_dict that was populated by `dcp.load()`.

---

## Root Cause Analysis

### What Happened

1. **hf_load() calls dcp.load()**
   ```python
   dcp.load(state_dict, storage_reader=hf_storage_reader)
   ```
   
2. **Expected behavior**: 
   - `state_dict` should be populated with HF format keys
   - Keys like "model.layers.0.self_attn.k_proj.weight"
   
3. **Then from_hf() is called**:
   ```python
   state_dict = self.sd_adapter.from_hf(state_dict)
   ```
   - Should convert HF keys to native format
   - "model.layers.0.self_attn.k_proj.weight" → "layers.0.attention.qkv_linear.wk.weight"
   
4. **The Problem**:
   - `dcp.load()` with HF storage reader is NOT populating the state_dict correctly
   - The state_dict remains empty or with wrong keys
   - `from_hf()` tries to process empty/wrong dict
   - Results in missing keys when loading into model

### Why dcp.load() Didn't Work

The issue is likely:
1. The `hf_storage_reader` returned by `self.sd_adapter.get_hf_storage_reader()` expects a certain state_dict structure
2. But we're passing an empty or pre-structured state_dict expecting it to be populated

**The mismatch**: 
- We call `dcp.load(state_dict, storage_reader=hf_storage_reader)`
- But `state_dict` should already have the model keys defined (sharded DTensors)
- Instead, we're passing the raw state dict that needs to be populated from HF

### Code Path Issue

In the old `dcp_load()` method (before removal):
```python
hf_state_dict = self.sd_adapter.to_hf(state_dict)  # Create HF key structure
hf_storage_reader = self.sd_adapter.get_hf_storage_reader(checkpoint_id, from_quantized)
dcp.load(hf_state_dict, storage_reader=hf_storage_reader)  # Populate with HF values
state_dict = self.sd_adapter.from_hf(hf_state_dict)  # Convert back to native
```

In the new `hf_load()` method:
```python
hf_storage_reader = self.sd_adapter.get_hf_storage_reader(checkpoint_id, from_quantized)
dcp.load(state_dict, storage_reader=hf_storage_reader)  # Try to populate state_dict
state_dict = self.sd_adapter.from_hf(state_dict)  # But state_dict is empty!
```

**The issue**: We removed the `to_hf()` call that was creating the HF key structure for dcp.load() to populate!

---

## The Bug in Our Recent Fix

### What We Removed

We removed this line that was actually necessary:
```python
hf_state_dict = self.sd_adapter.to_hf(state_dict)
```

**Why this matters**:
- `to_hf()` creates a template state dict with **HF format keys**
- The keys it creates define WHERE dcp.load() should put data
- Without this template, dcp.load() doesn't know what keys to populate
- Result: state_dict stays empty

### The Original Bug

The original bug was that `to_hf()` was being called but:
1. It applied permutation (for saving native→HF)
2. But we were loading HF, not converting from native
3. This caused the tensor shape issues we saw earlier

### Why Our Fix Made It Worse

Our fix completely removed `to_hf()` thinking it was the problem, but:
- `to_hf()` has two purposes:
  1. **For saving**: Convert native format tensors TO HF format
  2. **For loading HF**: Create HF key structure for dcp.load()
  
We needed the second purpose but not the first!

---

## Correct Solution

### The Real Fix

We need to:
1. Create HF key structure (but DON'T apply permutation)
2. Use that structure for dcp.load()
3. Convert HF keys back to native with from_hf()

### Proper Implementation

Option 1: Create a separate method for creating HF key template:
```python
def _create_hf_state_dict_template(self, state_dict):
    """Create state dict with HF key structure but no values yet."""
    # For each key in native state_dict, create corresponding HF key
    # without applying any transformations
    hf_dict = {}
    to_hf_map = {v: k for k, v in self.from_hf_map.items() if v is not None}
    
    for key in state_dict:
        abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
        if abstract_key in to_hf_map:
            hf_key = to_hf_map[abstract_key]
            hf_dict[hf_key] = state_dict[key]
    return hf_dict
```

Option 2: Use to_hf() but WITHOUT the permutation:
```python
# Create HF template (just key mapping, no tensor transformation)
hf_state_dict = {}
to_hf_map = {v: k for k, v in self.from_hf_map.items() if v is not None}
for key in state_dict:
    abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
    if abstract_key in to_hf_map:
        hf_state_dict[to_hf_map[abstract_key]] = state_dict[key]

# Load HF data into this template
dcp.load(hf_state_dict, storage_reader=hf_storage_reader)

# Convert back to native format
state_dict = self.sd_adapter.from_hf(hf_state_dict)
```

---

## Why Previous Run Failed Too

The log shows the previous error was also happening - the double-permutation issue. But THAT error happened DURING the reshape in `_permute()`. This error is happening BEFORE that - when trying to load the dict structure itself.

This suggests the recent DCP removal changes broke the loading path, but in a different place than the original double-permutation bug.

---

## Summary of Issues

| Issue | Root Cause | Previous Status | Current Status |
|-------|-----------|-----------------|-----------------|
| Double-permutation during reshape | `to_hf()` called on HF data | FIXED by removing to_hf() | N/A - not reached |
| Empty state dict after dcp.load() | Removed `to_hf()` that created structure | N/A | **NEW - BROKEN** |
| Missing HF keys in loading | Incomplete implementation of hf_load() | N/A | **CURRENT BLOCKER** |

---

## Required Changes

**File**: `torchtitan/components/checkpoint.py`  
**Method**: `hf_load()`

**Issue**: The method doesn't create the HF key template before calling dcp.load()

**Fix**: Restore HF key template creation but WITHOUT applying tensor transformations

**Implementation**:
1. Create HF state dict with HF format keys (from from_hf_map)
2. But do NOT apply any permutations or reshaping
3. Call dcp.load() with this structure
4. Call from_hf() to convert keys and apply permutations

---

## Testing After Fix

Once fixed, test with:
```bash
MODEL=/path/to/hf/model ./run_train_torchtitan.sh single -- --training.steps 10
```

Expected:
- ✓ Model loads from HF safetensors
- ✓ No missing key errors
- ✓ No shape mismatch errors
- ✓ Training loop starts

---

## Lessons Learned

1. `to_hf()` serves TWO purposes:
   - When SAVING: Transform tensor values and keys to HF format
   - When LOADING: Create HF key structure for dcp.load()
   
2. We can't completely remove `to_hf()` when loading HF
   - We need the key mapping part
   - But not the value transformation part
   
3. The original double-permutation bug occurred because:
   - We were calling `to_hf()` on already-HF-format data
   - Not because the method itself was wrong
   - The fix should have been to skip `to_hf()` only when input is already HF format
   - Not to remove the HF key structure creation entirely

---

## Recommendation

Revert the hf_load() method to properly create HF key template, and handle the double-permutation issue differently:
- Detect if input is already HF format
- Skip permutation in `_permute()` if already HF
- Or, properly construct HF template without permutation
