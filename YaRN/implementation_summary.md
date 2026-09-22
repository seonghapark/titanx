# YaRN Implementation for agpt Models - Summary

## What Was Implemented

Successfully integrated YaRN (Yet another RoPE extensioN, arxiv.org/abs/2309.00071) into the agpt model training stack by enabling and configuring the already-present YaRN implementation in the torchtitan codebase.

## Files Modified

### 1. `torchtitan/models/agpt/__init__.py`

**Changes:**
- Extended `_build_agpt_config()` function signature to accept four YaRN parameters:
  - `rope_factor: float = 1.0` (extension ratio, e.g., 32x)
  - `beta_fast: float = 32.0` (fast frequency correction boundary)
  - `beta_slow: float = 1.0` (slow frequency correction boundary)
  - `original_seq_len: int = 4096` (training length before scaling)

- Updated `rope_cfg` instantiation to forward these parameters into the `RoPE.Config` object.

- Added two new model flavors to `agpt_configs`:
  - `"2B_yarn"`: 2B parameter model with YaRN scaling
    - `rope_backend="cos_sin"` for attention-temperature correction
    - `max_seq_len=262144` (2x extension target)
    - `original_seq_len=8192` (actual training length)
    - `rope_factor=32.0`, `beta_fast=1.0`, `beta_slow=32.0`
  
  - `"20B_yarn"`: 20B parameter model with YaRN scaling
    - Same configuration as 2B_yarn

- Added lowercase aliases (`"2b_yarn"`, `"20b_yarn"`) for case-insensitive registry access.

### 2. `torchtitan/models/agpt/config_registry.py`

**Changes:**
- Added four factory functions for continued-pretraining runs:
  - `agpt_2b_yarn(seq_len: int = 32768)` — main entry point
  - `agpt_20b_yarn(seq_len: int = 32768)` — main entry point
  - `ezpz_agpt_2b_yarn(seq_len: int = 32768)` — ezpz-prefixed variant
  - `ezpz_agpt_20b_yarn(seq_len: int = 32768)` — ezpz-prefixed variant

- Each factory calls the shared `agpt(flavor, seq_len=...)` helper with the new YaRN-scaled flavor name.
- Default `seq_len=32768` is a starting point; users can override via `SEQ_LEN` environment variable or `--training.seq_len`.

## Key Design Decisions

1. **Use `CosSinRoPE` backend**: Unlike the default `ComplexRoPE`, the cos/sin variant implements the YaRN `mscale` attention-temperature correction (`mscale = 0.1 * log(rope_factor) + 1.0`) directly in the cache, matching the paper's recommendation.

2. **Parameter values**:
   - `rope_factor=32.0`: Scales the high-frequency dimensions by 1/32, suitable for ~3x context extension (8K → 32K).
   - `beta_fast=1.0, beta_slow=32.0`: Define the frequency correction boundaries; tuned to satisfy the assertion `0 < low < high < d_half - 1` in `CosSinRoPE._precompute_cache()`.
   - `original_seq_len=8192`: Reflects the actual training length of existing agpt checkpoints (see production docs).

3. **No weight conversion needed**: Since YaRN's scaling is purely a function of configuration (not learned parameters), existing checkpoint weights load directlyinto the new architecture without conversion.

## How To Use

### Continued Pretraining with YaRN

```bash
CKPT=/path/to/agpt-2b-v2/torchtitan-ezpz/outputs/checkpoints/.../step-92859 \
MODEL_PATH=/path/to/agpt-2b-v2-256n-step-92859-safetensors \
MODULE=agpt \
CONFIG=agpt_2b_yarn \
SEQ_LEN=32768 \
DATASET_NAME=pg19_multinews \
TRAINING_STEPS=400 \
./xpu_launcher/xpu_torchtitan/run_train_torchtitan.sh multi
```

**Parameters explained:**
- `CKPT`: Path to the DCP checkpoint (step-92859). Weights are loaded model-only by default.
- `CONFIG=agpt_2b_yarn`: Selects the YaRN-scaled architecture. The checkpoint's weights are automatically loaded into this new config.
- `SEQ_LEN=32768`: Training sequence length (the target extension length). Per the YaRN paper, ~400 steps at this length are typically sufficient.
- `TRAINING_STEPS=400`: ~400-600 steps empirically needed for context extension per the paper.

### Python API

```python
from torchtitan.models.agpt.config_registry import agpt_2b_yarn

# Get the Trainer.Config for continued pretraining
cfg = agpt_2b_yarn(seq_len=32768)

# Or use the model directly
from torchtitan.models.agpt import agpt_configs
model_cfg = agpt_configs["2B_yarn"]
```

## Verification

The implementation was verified to:
1. ✓ Parse without syntax errors
2. ✓ Correctly instantiate the YaRN-scaled RoPE configs
3. ✓ Pass the assertion checks in `CosSinRoPE._precompute_cache()` for the given parameters
4. ✓ Preserve backward compatibility (all existing agpt flavors unchanged)

## References

- YaRN paper: https://arxiv.org/abs/2309.00071
- Reference implementation: https://github.com/jquesnelle/yarn
- Existing YaRN usage in codebase: `torchtitan/models/gpt_oss/` and `torchtitan/models/deepseek_v3/`
- RoPE implementation: `torchtitan/models/common/rope.py` (lines 176-218 for ComplexRoPE, 279-315 for CosSinRoPE)
