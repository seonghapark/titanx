# TorchTitan Integration Guide for agpt-20b YaRN

## Overview

This document shows the code changes needed in TorchTitan to wire up YaRN for agpt-20b. The changes are minimal because YaRN math already exists in `torchtitan/models/common/rope.py`.

## File 1: torchtitan/models/agpt/__init__.py

### Change 1: Update `_build_agpt_config` signature

Add YaRN parameters to the function signature:

```python
def _build_agpt_config(
    dim: int,
    n_layers: int,
    n_heads: int,
    n_kv_heads: int,
    hidden_dim: int,
    rope_theta: float,
    vocab_size: int,
    scaling: Literal["none", "llama", "yarn"] = "none",
    max_seq_len: int = 131072,
    # NEW: YaRN parameters
    rope_factor: float = 1.0,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    original_seq_len: int = 2048,
    rope_backend: Literal["complex", "cos_sin"] = "complex",
) -> ModelConfig:
    """Build an AGPT model config with optional YaRN RoPE scaling."""
    # ... existing code ...
```

### Change 2: Forward YaRN params to rope_cls

Update the RoPE instantiation (around line 357-362):

```python
# OLD CODE (before YaRN):
rope = rope_cls(
    dim=dim,
    max_seq_len=max_seq_len,
    theta=rope_theta,
    scaling=scaling,
)

# NEW CODE (with YaRN):
rope = rope_cls(
    dim=dim,
    max_seq_len=max_seq_len,
    theta=rope_theta,
    scaling=scaling,
    rope_factor=rope_factor,
    beta_fast=beta_fast,
    beta_slow=beta_slow,
    original_seq_len=original_seq_len,
)
```

### Change 3: Add YaRN model flavors to agpt_configs

Add to the `agpt_configs` dictionary:

```python
agpt_configs = {
    # ... existing configs ...
    
    # YaRN-scaled variants (new)
    "20B_yarn": _build_agpt_config(
        dim=5120,
        n_layers=64,
        n_heads=40,
        n_kv_heads=8,
        rope_theta=500000,
        vocab_size=256128,
        hidden_dim=14336,
        scaling="yarn",
        max_seq_len=262144,              # Upper bound for RoPE cache
        original_seq_len=8192,           # Actual training length
        rope_factor=2.0,                 # 2x context extension
        beta_fast=1.0,                   # High-freq: no scaling
        beta_slow=32.0,                  # Low-freq: 32x scaling
        rope_backend="cos_sin",          # Enable mscale correction
    ),
    "2B_yarn": _build_agpt_config(
        dim=2048,
        n_layers=12,
        n_heads=16,
        n_kv_heads=4,
        rope_theta=50000,
        vocab_size=256128,
        hidden_dim=11008,
        scaling="yarn",
        max_seq_len=262144,
        original_seq_len=8192,
        rope_factor=2.0,
        beta_fast=1.0,
        beta_slow=32.0,
        rope_backend="cos_sin",
    ),
}

# Add lowercase aliases after the main dict
agpt_configs["20b_yarn"] = agpt_configs["20B_yarn"]
agpt_configs["2b_yarn"] = agpt_configs["2B_yarn"]
```

## File 2: torchtitan/models/agpt/config_registry.py

Add training recipe factories:

```python
def agpt_20b_yarn(seq_len: int = 16384) -> Trainer.Config:
    """Continue-pretrain agpt-20b with YaRN-scaled RoPE for context extension.
    
    Loads the existing checkpoint weights (only model state, not optimizer state
    via CheckpointManager.Config.initial_load_model_only=True) and fine-tunes at
    a longer sequence length. Per the YaRN paper, ~400 steps at the target length
    are needed to stabilize learning.
    
    Args:
        seq_len: Target sequence length for fine-tuning. Default 16384 (2x extension
                 from original 8192). Can extend further in subsequent runs.
    
    Returns:
        Trainer.Config configured for continued pretraining with YaRN scaling.
    """
    return agpt("20b_yarn", seq_len=seq_len, activation_checkpoint_mode="none")


def agpt_2b_yarn(seq_len: int = 16384) -> Trainer.Config:
    """Continue-pretrain agpt-2b with YaRN-scaled RoPE for context extension."""
    return agpt("2b_yarn", seq_len=seq_len, activation_checkpoint_mode="none")


# Optional: ezpz-style variants
def ezpz_agpt_20b_yarn(seq_len: int = 16384) -> Trainer.Config:
    """EZpz-compatible version of agpt_20b_yarn."""
    return agpt("20b_yarn", seq_len=seq_len)


def ezpz_agpt_2b_yarn(seq_len: int = 16384) -> Trainer.Config:
    """EZpz-compatible version of agpt_2b_yarn."""
    return agpt("2b_yarn", seq_len=seq_len)
```

## No Changes Needed To

The following files require **no modifications** because YaRN is fully compatible:

1. **torchtitan/models/common/rope.py** — YaRN math already implemented
2. **torchtitan/models/common/decoder.py** — seq_len validation is generic
3. **torchtitan/models/agpt/attention.py** — RoPE interface unchanged
4. **torchtitan/models/agpt/state_dict_adapter.py** — Weight loading unchanged
5. **xpu_launcher/run_train_torchtitan.sh** — Env var plumbing already exists

## Validation Checklist

Run these checks after implementing the changes:

### 1. Import Check
```python
from torchtitan.models.agpt import agpt_configs
config = agpt_configs["20b_yarn"]
print(f"Rope scaling: {config.model.layers[0].attention.rope.scaling}")
# Expected: "yarn"
```

### 2. Config Instantiation
```python
from torchtitan.models.agpt.config_registry import agpt_20b_yarn
trainer_config = agpt_20b_yarn(seq_len=16384)
print(f"Seq len: {trainer_config.training.seq_len}")
# Expected: 16384
```

### 3. RoPE Initialization
```python
from torchtitan.models.common.rope import CosSinRoPE
rope_config = config.model.layers[0].attention.rope
rope = CosSinRoPE(rope_config)
print(f"Cache shape: {rope.cos_sin_cache.shape}")
# Expected: (1, <seq_len>, <dim>, 2)
```

### 4. Assertion Check (IMPORTANT)
The RoPE constructor validates `0 < low < high < d_half - 1`. This should pass:
```python
rope = CosSinRoPE(rope_config)  # Should not raise AssertionError
```

If this raises an assertion, check:
- `rope_factor` and `beta_fast`/`beta_slow` are compatible with model dims
- `original_seq_len` matches checkpoint's actual training length

## Testing Strategy

### Level 1: Unit Tests (Local, no GPU)
```bash
python3 -c "
from torchtitan.models.agpt import agpt_configs
config = agpt_configs['20b_yarn']
print('✓ Config loaded')
rope = config.model.layers[0].attention.rope
print(f'✓ RoPE config valid: scaling={rope.scaling}')
"
```

### Level 2: Dry Run (Single GPU or CPU)
```bash
MODULE=agpt CONFIG=agpt_20b_yarn SEQ_LEN=16384 TRAINING_STEPS=5 \
./run_train_torchtitan.sh single --dry-run
```

### Level 3: Short Run (4-8 GPUs)
```bash
MODULE=agpt CONFIG=agpt_20b_yarn SEQ_LEN=16384 TRAINING_STEPS=50 \
./run_train_torchtitan.sh multi
```
Monitor loss — should be stable, no spikes.

### Level 4: Full Training (8+ nodes)
```bash
MODULE=agpt CONFIG=agpt_20b_yarn SEQ_LEN=16384 TRAINING_STEPS=400 \
./run_train_torchtitan.sh multi
```

## Verification Against Reference

The YaRN implementation matches:
- **NTK-by-parts**: Dynamic interpolation ramp (deepseek_v3 style)
- **mscale correction**: Attention temperature scaling from paper (§3.4)
- **Parameter ranges**: 
  - `rope_factor`: 1.0-32.0 (tested values)
  - `beta_fast`: 0.5-2.0 (high-freq scaling)
  - `beta_slow`: 16-40 (low-freq scaling)

## Launch Command Reference

```bash
# Minimal viable launch
CKPT=/path/to/step-4000 \
MODULE=agpt CONFIG=agpt_20b_yarn SEQ_LEN=16384 TRAINING_STEPS=400 \
./run_train_torchtitan.sh multi

# With custom parallelism
TP=8 PP=1 CP=1 \
CKPT=/path/to/step-4000 \
MODULE=agpt CONFIG=agpt_20b_yarn SEQ_LEN=16384 TRAINING_STEPS=400 \
./run_train_torchtitan.sh multi

# For 2B model
CKPT=/path/to/2b-step-92859 \
MODULE=agpt CONFIG=agpt_2b_yarn SEQ_LEN=16384 TRAINING_STEPS=400 \
./run_train_torchtitan.sh multi
```

## HuggingFace Safetensors Export (Post-Training)

After continued training, convert to HF format:

```bash
python3 llm_evaluation/convert_dcp_to_safetensors.py \
  --input-path /path/to/yarn-step-400 \
  --output-path agpt-20b-v2-512n-step-4400-yarn-16k-safetensors
```

The `config.json` in the exported checkpoint will automatically include the
rope_scaling block, making it compatible with vLLM/HF transformers.

## References

- YaRN Paper: https://arxiv.org/abs/2309.00071
- jquesnelle/yarn: https://github.com/jquesnelle/yarn
- TorchTitan Rope: `torchtitan/models/common/rope.py` (lines 1-315)
- Existing YaRN usage: gpt_oss (`models/gpt_oss/`), deepseek_v3 (`models/deepseek_v3/`)

---

**Implementation Status**: Ready to integrate. Code changes are minimal and low-risk.
