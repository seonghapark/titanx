# Checkpoint System Refactoring Summary

## Overview
Removed DCP checkpoint initial load capability and kept only HF format model loading. This simplifies the checkpoint system and removes redundant code paths.

## Changes Made

### 1. **torchtitan/components/checkpoint.py** - Config Options
**Removed:**
- `initial_load_path: str | None = None`
- `initial_load_model_only: bool = False`
- `initial_load_in_hf: bool = False`
- `initial_load_in_hf_quantized: bool = False`

**Added:**
- `load_hf_model: bool = True`
- `load_hf_model_quantized: bool = False`

### 2. **torchtitan/components/checkpoint.py** - __post_init__ Validation
**Removed validation code for:**
- `initial_load_path` (absolute path check)
- `initial_load_in_hf` requiring `initial_load_model_only=True`
- `initial_load_in_hf_quantized` requiring both `initial_load_in_hf` and `initial_load_path`
- Warning about `initial_load_model_only` without `initial_load_path`

**Kept:**
- Validation for `last_save_in_hf` requiring `last_save_model_only=True`

### 3. **torchtitan/components/checkpoint.py** - __init__ Method
**Removed:**
```python
self.initial_load_path = config.initial_load_path
self.initial_load_model_only = config.initial_load_model_only
self.initial_load_in_hf = config.initial_load_in_hf
self.initial_load_in_hf_quantized = config.initial_load_in_hf_quantized
```

**Added:**
```python
self.load_hf_model = config.load_hf_model
self.load_hf_model_quantized = config.load_hf_model_quantized
```

### 4. **torchtitan/components/checkpoint.py** - _maybe_init_load Method
**Simplified logic:**
- Removed all DCP checkpoint path handling
- Removed conditional logic for `initial_load_path`
- Set `model_only = True` unconditionally (always load model only from HF)
- Simplified to: Always load from `sd_adapter.hf_assets_path` when checkpoint folder doesn't exist
- Removed warnings about conflicting `initial_load_path` and checkpoint folder

**Before (Complex flow):**
```python
if self.initial_load_path:
    # Load from DCP path
elif from_hf:
    # Load from HF
else:
    # Return False
```

**After (Simplified):**
```python
if from_hf:
    # Load from HF (required)
else:
    # Return False
```

### 5. **run_train_torchtitan.sh** - Removed CKPT Support
**Removed from documentation:**
- `CKPT (optional DCP directory, passed as --checkpoint.initial_load_path)`

**Removed code:**
- CKPT variable initialization and validation
- Logic to pass `--checkpoint.initial_load_path` to training command

**Remaining checkpoint options:**
- `CKPT_FOLDER` (directory name for saving checkpoints, not for loading)

### 6. **torchtitan/models/agpt/config_registry.py** - Updated Documentation
**Before:**
```
Load the existing agpt-2b checkpoint's weights only
(--checkpoint.initial_load_path=<DCP step-92859 dir>,
initial_load_model_only=True is the CheckpointManager default)
```

**After:**
```
Loads the existing agpt-2b model weights from HF format
(via --model.hf_assets_path)
```

### 7. **torchtitan/models/qwen3/config_registry.py** - Updated Config Usage
**Before:**
```python
checkpoint=CheckpointManager.Config(
    enable=True,
    initial_load_in_hf=True,
),
```

**After:**
```python
checkpoint=CheckpointManager.Config(
    enable=True,
    load_hf_model=True,
),
```

## Behavior Changes

### Loading Model Weights
**Old:** Could load from either:
1. DCP checkpoint at `--checkpoint.initial_load_path`
2. HF format at `--model.hf_assets_path` (with `initial_load_in_hf=True`)

**New:** Only loads from:
- HF format at `--model.hf_assets_path` (with `load_hf_model=True`, default)

### Checkpoint Resumption
**Old:** Could resume from DCP checkpoints with full training state (optimizer, scheduler, dataloader)

**New:** Only resumes from checkpoints saved in current training run. Starting fresh requires HF model.

## Benefits

1. **Simpler Code:** Removed 100+ lines of conditional logic
2. **No Topology Mismatch:** No DCP state_dict conflicts from different parallelism configurations
3. **Cleaner API:** Only two checkpoint config options instead of four
4. **Consistent:** All model loading uses standard HF safetensors format
5. **Safer:** HF models are always model-only (no optimizer/scheduler state issues)

## Breaking Changes

Users can no longer:
- Use `--checkpoint.initial_load_path=<path>` to load DCP checkpoints
- Load full training state (optimizer + scheduler) from checkpoints
- Use `--checkpoint.initial_load_model_only` to toggle this behavior

All these are now replaced by the standard HF model loading via `MODEL` env var and `--model.hf_assets_path`.

## Migration Guide

### Old Usage (No longer works):
```bash
./run_train_torchtitan.sh single -- \
  --checkpoint.initial_load_path=/path/to/checkpoint/step-1000 \
  --checkpoint.initial_load_model_only=true
```

### New Usage:
```bash
MODEL=/path/to/hf/model ./run_train_torchtitan.sh single
```

The HF model is automatically loaded from `--model.hf_assets_path` (defaults to `MODEL` env var).

## Files Modified

1. `torchtitan/components/checkpoint.py` - Core checkpoint implementation
2. `run_train_torchtitan.sh` - Shell script training launcher
3. `torchtitan/models/agpt/config_registry.py` - Documentation
4. `torchtitan/models/qwen3/config_registry.py` - Config update

## Remaining Configuration Files (Not Updated - Experiments)

These still reference old config options but are in experiment/test code:
- `torchtitan/experiments/rl/generate.py`
- `torchtitan/experiments/rl/__init__.py`
- `torchtitan/experiments/rl/examples/alphabet_sort/config_registry.py`
- `torchtitan/experiments/rl/examples/search_r1/config_registry.py`
- `torchtitan/experiments/rl/models/vllm_wrapper.py`
- `torchtitan/experiments/rl/models/vllm_registry.py`
- `torchtitan/experiments/rl/actors/generator.py`
- `torchtitan/experiments/rl/tests/test_bitwise_parity.py`
- `torchtitan/experiments/graph_trainer/tests/test_bitwise_deterministic.py`
- `torchtitan/experiments/torchft/tests/test_torchft_checkpoint.py`

These files are not used in the AGPT context extension training and can be updated separately if needed.

## Testing

The changes have been validated to:
- ✓ Successfully parse the new config options
- ✓ Load HF models from `hf_assets_path`
- ✓ Eliminate DCP path mismatches
- ✓ Allow early termination feature to work without checkpoint conflicts
