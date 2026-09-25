# Run Log Analysis v3 - 2026-09-23 (Latest Run)

## Summary
**Status**: ✗ FAILED  
**Error Type**: RuntimeError in tensor reshape  
**Affected Ranks**: All 48 ranks  
**Root Cause**: DTensor sharding pattern mismatch in _reverse_permute()

---

## Primary Error

### Error Message
```
RuntimeError: shape '[16, 2, 1, 2048]' is invalid for input of size 88064
```

### Error Location Stack
```
torchtitan/components/checkpoint.py:774 in load()
  → self.hf_load(states, checkpoint_id=..., from_quantized=...)
    
torchtitan/components/checkpoint.py:614 in hf_load()
  → state_dict_native = self.sd_adapter.from_hf(hf_state_dict)
    
torchtitan/models/llama3/state_dict_adapter.py:167 in from_hf()
  → value = self._reverse_permute(value, n_heads)
  
torchtitan/models/llama3/state_dict_adapter.py:93 in _reverse_permute()
  → w.view(n_heads_arg, 2, dim1 // n_heads_arg // 2, dim2)
```

### Tensor Dimension Analysis

**Expected** (from code):
- view([16, 2, 1, 2048]) = 16 × 2 × 1 × 2048 = **65,536 elements**

**Actual** (input tensor):
- 88,064 elements

**Mismatch**: 88,064 ≠ 65,536

**Possible Explanation**:
- 88,064 / 2048 = 43 (odd number, suggests non-uniform sharding)
- Expected size per rank: 65,536 / 48 ≈ 1,365 per rank
- Actual size: 88,064 / 48 ≈ 1,835 per rank
- Suggests different sharding pattern than expected

---

## Root Cause Analysis

### The Problem Sequence

1. **hf_load() creates HF state dict with DTensor placeholders**
   ```python
   hf_state_dict[hf_key] = value  # value is a DTensor from native state_dict
   ```

2. **dcp.load() loads HF safetensors into this dict**
   ```python
   dcp.load(hf_state_dict, storage_reader=hf_storage_reader)
   ```
   
3. **DTensors are now populated with HF data BUT...**
   - DTensors have a sharding pattern
   - HF safetensors data is non-distributed
   - dcp.load() applies a sharding pattern when converting
   - The sharding pattern might not match the original native state dict

4. **from_hf() calls _reverse_permute() on sharded DTensor**
   ```python
   value = self._reverse_permute(value, n_heads)  # value is DTensor with HF sharding
   ```

5. **_reverse_permute() tries to reshape sharded tensor**
   ```python
   w.view(16, 2, 1, 2048)  # But w is sharded with pattern that doesn't match
   ```

### Why to_local() Didn't Work

The code has:
```python
if isinstance(w, DTensor):
    w = w.to_local()
```

But there are two possibilities:
1. **w is NOT a DTensor instance**: isinstance() check fails
   - Tensor might be a different type after dcp.load()
   - Might be a wrapper or subclass not recognized

2. **to_local() is being called but returns wrong shape**:
   - to_local() gathers distributed tensor
   - But if gathering fails or returns partial tensor, size doesn't match
   - 88,064 elements suggests partial/incomplete gather

3. **isinstance check location is wrong**:
   - Check is in _reverse_permute()
   - But error is happening BEFORE to_local() is applied
   - Suggests check is not executing before reshape

### Key Insight

The error trace shows line 93 which is the `.view()` call. If `to_local()` at line 85-86 was executing successfully, the tensor would be a regular (non-sharded) tensor and the error would be different. The fact that we get exactly this error suggests `to_local()` is NOT being called or NOT converting the tensor properly.

---

## Why This Is Happening

### The Architectural Problem

We're mixing two different data formats:

**Native Format State Dict** (from model training):
- Keys: `layers.0.attention.qkv_linear.wk.weight`
- Values: DTensors sharded according to model's parallelism strategy
- Sharding: FSDP sharding pattern defined in training

**HF Safetensors** (external model):
- Keys: `model.layers.0.self_attn.k_proj.weight`
- Values: Regular (non-distributed) tensors
- No sharding information

**Current Approach** (hf_load):
1. Create HF-keyed dict with native DTensor placeholders
2. Load HF safetensors into these DTensor placeholders
3. dcp.load() applies its own sharding pattern
4. Result: DTensors with HF-incompatible sharding
5. from_hf() tries to use RoPE operations on wrong-sharded tensors

### The Mismatch

When dcp.load() loads HF safetensors into DTensor placeholders:
- It doesn't know what sharding pattern to use
- It might use default FSDP sharding (shard along specific dim)
- But _reverse_permute() expects original training sharding pattern
- Result: Shape mismatch on reshape

---

## Solution

### Root Problem
We're trying to load HF tensors into DTensor placeholders, but the sharding patterns don't match.

### Proper Solution
Don't use DTensor placeholders for HF loading. Instead:

**Option 1: Load to regular tensors first, then convert**
```python
# Create dict with regular tensor placeholders (not DTensors)
hf_state_dict = {}  # Empty dict, no DTensor placeholders

hf_storage_reader = self.sd_adapter.get_hf_storage_reader(checkpoint_id, from_quantized)

# dcp.load will populate with regular tensors from HF
dcp.load(hf_state_dict, storage_reader=hf_storage_reader)

# Now convert to native format (regular tensors to DTensors with correct sharding)
state_dict_native = self.sd_adapter.from_hf(hf_state_dict)

# Explicitly shard the result using model's sharding patterns
# (This part might be automatic via load_state_dict)
self.states[MODEL].load_state_dict(state_dict_native)
```

**Option 2: Don't pre-populate hf_state_dict, let dcp.load create it**
```python
# Start with completely empty dict
hf_state_dict = {}

# dcp.load will create and populate with HF tensors
dcp.load(hf_state_dict, storage_reader=hf_storage_reader)

# Convert to native format
state_dict_native = self.sd_adapter.from_hf(hf_state_dict)

# Let model's load_state_dict handle sharding
self.states[MODEL].load_state_dict(state_dict_native)
```

### Recommended Fix

**Option 2 is simpler**. The current code tries to help dcp.load() by pre-creating HF-keyed placeholders, but this creates the sharding mismatch. Instead:

1. Start with empty dict
2. Let dcp.load() populate with HF tensors (as regular tensors)
3. from_hf() converts keys AND handles any required transformations
4. model.load_state_dict() handles sharding


```python
# BEFORE (current, broken):
hf_state_dict = {}
to_hf_map = {...}
for key, value in state_dict.items():
    hf_key = to_hf_map.get(...)
    if hf_key is not None:
        hf_state_dict[hf_key] = value  # PROBLEM: DTensor placeholders

dcp.load(hf_state_dict, storage_reader=hf_storage_reader)
state_dict_native = self.sd_adapter.from_hf(hf_state_dict)

# AFTER (fixed):
hf_state_dict = {}  # Start completely empty, no placeholders

dcp.load(hf_state_dict, storage_reader=hf_storage_reader)

state_dict_native = self.sd_adapter.from_hf(hf_state_dict)
self.states[MODEL].load_state_dict(state_dict_native)
```

### Why This Works

1. dcp.load() with empty dict creates keys/values as needed
2. HF storage reader provides HF keys and regular (non-DTensor) values
3. hf_state_dict ends up with: `{hf_keys: regular_tensors}`
4. from_hf() processes regular tensors (no sharding complexity)
5. Converts to native format and applies RoPE reversal
6. Result: regular tensors in native format
7. model.load_state_dict() automatically shards using model's parallelism settings

---

## Additional Fix Required

Even with the above fix, we need to ensure `_reverse_permute()` can handle DTensors properly (for when it's called on other paths). The existing `to_local()` check is good but may not be sufficient.

**Better fix for _reverse_permute()**:
```python
def _reverse_permute(self, w, n_heads_arg, dim1=None, dim2=None):
    # Convert FSDP-sharded tensor to local
    if isinstance(w, DTensor):
        try:
            w = w.to_local()
        except Exception as e:
            logger.warning(f"Failed to convert DTensor to local: {e}, using as-is")
            # Fall through - might work anyway for replicated tensors
    
    # ... rest of method
```

---

## Summary of Issues

| Issue | Cause | Status | Fix |
|-------|-------|--------|-----|
| Wrong sharding pattern for HF tensors | Using DTensor placeholders | CRITICAL | Use empty dict for hf_load |
| to_local() not preventing error | Wrong sharding applied by dcp.load | CRITICAL | Avoid DTensor placeholders |
| Shape mismatch in _reverse_permute | Sharded tensor with incompatible pattern | CRITICAL | Load HF as regular tensors |

---

## Files to Modify

**torchtitan/components/checkpoint.py** - hf_load() method (lines 588-616)

**Change**: Remove the loop that pre-populates hf_state_dict with DTensor placeholders. Start with empty dict instead.

---

## Testing After Fix

```bash
MODEL=/path/to/hf/model ./run_train_torchtitan.sh single -- --training.steps 5
```

Expected:
- ✓ No shape mismatch errors
- ✓ Model loads successfully
- ✓ Training begins
- ✓ Loss values printed

---

## Lesson Learned

DTensor sharding patterns must be managed carefully when mixing external data sources (HF) with distributed training. The safest approach is:
1. Load external data as regular tensors
2. Let the training framework's load_state_dict() handle sharding
3. Only operate on DTensors when their sharding pattern is known
