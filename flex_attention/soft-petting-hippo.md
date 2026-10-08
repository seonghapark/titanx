# FlexAttention Implementation Plan for AGPT Models

## Context

The AGPT model family currently supports two inner attention backends:
1. **SDPA** (Scaled Dot Product Attention) - default, via `XPUScaledDotProductAttention` / `EzpzScaledDotProductAttention`
2. **Softcapped FlexAttention** - limited to models with logit_softcap (e.g., "2B_softcap")

FlexAttention is a newer PyTorch primitive (`torch.nn.attention.flex_attention`) that:
- Provides fused attention kernels with custom score modification functions
- Supports `score_mod` for softcapping, position biases, and other attention customizations
- Requires `torch.compile` for peak performance
- Supports Context Parallel (CP) for distributed training
- Is already fully implemented in TorchTitan's `torchtitan/models/common/attention.py`

**Goal:** Enable FlexAttention as a standard attention backend option for all AGPT models (not just those with logit_softcap), making it available for benchmarking and experimentation.

---

## Current State

### Existing FlexAttention Support

**AGPT already has partial FlexAttention support:**
- `SoftcappedFlexAttention` class in `torchtitan/models/agpt/__init__.py` (lines 117-170)
  - Wraps `torch.nn.attention.flex_attention.flex_attention` 
  - Applies tanh softcapping via score_mod (Gemma2-style)
  - Requires `torch.compile`

- Standard `FlexAttention` fully implemented in `torchtitan/models/common/attention.py` (lines 208-339)
  - Supports arbitrary score_mod functions (or None)
  - Handles block_mask, GQA, scale, transpose conventions
  - Already used by Llama3 when `attn_backend="flex"`

### Config Infrastructure

**Model building functions** in `torchtitan/models/agpt/__init__.py`:
- `_build_agpt_layers()` (lines 259-327): Builds per-layer configs
  - Accepts `attn_backend` parameter (default: "sdpa")
  - When `logit_softcap is not None`, uses `SoftcappedFlexAttention` 
  - Otherwise uses `_ezpz_get_attention_config(attn_backend)` for "sdpa" or delegates to upstream

- `_build_agpt_config()` (lines 330-397): Top-level config builder
  - Accepts `attn_backend` parameter (default: "sdpa")
  - Passes it through to `_build_agpt_layers()`

- `_ezpz_get_attention_config()` (lines 239-256): Backend selector
  - Handles "sdpa" backend specially (uses XPU-optimized classes)
  - Delegates other backends to upstream `get_attention_config()`
  - Already supports "flex" backend via upstream

**Existing AGPT configs** in `agpt_configs` dict (lines 400+):
- `"debugmodel_flex_attn"` (line 410): debug model with `attn_backend="flex"`
- `"2B_flex_attn"` (line 481): 2B model with `attn_backend="flex"`
- `"20B_flex_attn"` (line 520): 20B model with `attn_backend="flex"`
- `"2B_softcap"` (line 449): 2B model with logit_softcap (uses SoftcappedFlexAttention internally)

---

## Implementation Plan

### Phase 1: Enable FlexAttention for All AGPT Sizes

#### 1.1 Add FlexAttention Config Variants
**File:** `torchtitan/models/agpt/__init__.py` (in `agpt_configs` dict)

For each model size, add a `"_flex_attn"` variant that uses FlexAttention:
- Copy existing config
- Set `attn_backend="flex"`
- Keep all other parameters identical

**Configs to add:**
```python
"debugmodel_flex_attn"          # Already exists
"2B_flex_attn"                  # Already exists
"2B_qknorm_flex_attn"           # NEW: QK norm + FlexAttention
"2B_softcap_flex_attn"          # NEW: Compare softcap via SoftcappedFlexAttention
"2B_kitchen_sink_flex_attn"     # NEW: All tweaks + FlexAttention
"7B_flex_attn"                  # NEW
"8B_flex_attn"                  # NEW
"20B_flex_attn"                 # Already exists
"50B_flex_attn"                 # NEW
"50B_wide_flex_attn"            # NEW
"70B_wide_flex_attn"            # NEW
"80B_flex_attn"                 # NEW
"80B_wide_flex_attn"            # NEW
"80B_deep_flex_attn"            # NEW
```

#### 1.2 Leverage Existing Upstream FlexAttention (VERIFIED OPTION)
**File:** `torchtitan/models/agpt/__init__.py`

**NO NEW CLASS NEEDED!** The upstream `FlexAttention` class already exists and is production-ready:
- Location: `torchtitan/models/common/attention.py:208-339`
- Already supports all needed features:
  - Arbitrary `score_mod` functions (or None)
  - Proper Q, K, V transpose handling
  - GQA support via `enable_gqa` parameter
  - Block-wise attention with configurable `block_size`
  - Context Parallel support
  - Kernel options and Triton autotuning

**Why use upstream's FlexAttention?**
- Already implements all AGPT needs (no score_mod needed for baseline)
- Used by Llama3 models successfully
- Comes with torch.compile + Triton optimization setup
- Consistent with TorchTitan architecture

**How AGPT will get FlexAttention:**
- Modify `_ezpz_get_attention_config(backend)` to handle "flex" → `FlexAttention.Config()`
- Delegates to upstream's `get_attention_config("flex")` already returns `FlexAttention.Config`
- Pattern: `attn_backend="flex"` → `_ezpz_get_attention_config("flex")` → `FlexAttention.Config()`

#### 1.3 Update Config Building Logic
**File:** `torchtitan/models/agpt/__init__.py`, function `_build_agpt_layers()` (lines 259-327)

**CURRENT LOGIC** (lines 274-279):
```python
if logit_softcap is not None:
    inner_attention = SoftcappedFlexAttention.Config(logit_cap=logit_softcap)
else:
    inner_attention = _ezpz_get_attention_config(attn_backend)
```

**The logic is ALREADY CORRECT:**
- When `attn_backend="flex"` is passed, `_ezpz_get_attention_config("flex")` calls upstream's `get_attention_config("flex")`
- Upstream returns `FlexAttention.Config()` which is exactly what we need
- When `logit_softcap` is set, it uses `SoftcappedFlexAttention` which wraps FlexAttention with logit capping

**NO CHANGES NEEDED** to the building logic! The infrastructure already supports it.

### Phase 2: Add Configuration Functions to Config Registry

**File:** `torchtitan/models/agpt/config_registry.py`

Add wrapper functions that call the builder with `attn_backend="flex"`:

```python
def agpt_2b_flex_attn() -> Trainer.Config:
    return agpt("2B", attn_backend="flex")

def agpt_7b_flex_attn() -> Trainer.Config:
    return agpt("7B", attn_backend="flex")

def agpt_8b_flex_attn() -> Trainer.Config:
    return agpt("8B", attn_backend="flex")

def agpt_20b_flex_attn() -> Trainer.Config:
    return agpt("20B", attn_backend="flex")

# ... one per model size (9 total configs)
```

These allow users to request via:
```bash
CONFIG=agpt_7b_flex_attn ./run_train_torchtitan.sh ...
```

**Note:** The `agpt()` builder function (line 107) already has `attn_backend="flex"` support via parameter (line 128, not shown in that snippet but part of `_base_config()` chain)

### Phase 3: Testing & Verification

#### 3.1 Unit Tests
- Verify FlexAttention configs are correctly built for each model size
- Check that forward pass works with FlexAttention backend
- Verify checkpoint compatibility (state dicts should be identical format)

#### 3.2 Integration Tests
- Run training with a small model (debugmodel or 2B) using FlexAttention
- Verify loss curves match SDPA baseline (not identical, but within ~2% range)
- Test with different sequence lengths (to exercise block_size logic)

#### 3.3 Distributed Tests
- Verify FlexAttention works with FSDP (if enabled)
- Verify FlexAttention works with Tensor Parallel (if enabled)
- Verify Context Parallel compatibility (use existing CP test suite)

#### 3.4 Documentation
- Add a note in AGPT config_registry explaining FlexAttention variants
- Note any performance characteristics or stability considerations
- Document which PyTorch version is required (2.5+)

---

## Files to Modify

| File | Change | Priority | Complexity |
|------|--------|----------|------------|
| `torchtitan/models/agpt/__init__.py` | Add 10+ new config variants to `agpt_configs` dict (lines 400+) | **HIGH** | **LOW** (copy-paste with `attn_backend="flex"`) |
| `torchtitan/models/agpt/config_registry.py` | Add 9 wrapper functions for flex_attn configs | **MEDIUM** | **LOW** (simple one-liners) |
| `tests/` (if exists) | Add integration tests for FlexAttention variants | **MEDIUM** | **MEDIUM** |
| Documentation | Update AGPT docs with FlexAttention options | **LOW** | **LOW** |

**Total New Code:** ~50 lines (mostly config definitions, no custom classes needed)

---

## Implementation Details

### No Custom Class Needed!

The upstream `FlexAttention` class at `torchtitan/models/common/attention.py:208-339` already has everything needed:

**FlexAttention Features:**
- Supports `score_mod=None` (baseline, no custom score modifications)
- `torch.compile` with Triton kernel generation
- Configurable `block_size` for sparse patterns
- Full `enable_gqa` support
- Context Parallel compatible
- Proper Q, K, V transpose handling (B,L,N,H ↔ B,N,L,H)

**How it gets used in AGPT:**
1. `attn_backend="flex"` passed to `_build_agpt_config()`
2. → `_build_agpt_layers()` calls `_ezpz_get_attention_config("flex")`
3. → delegates to upstream's `get_attention_config("flex")`
4. → returns `FlexAttention.Config()`
5. → passed to `make_gqa_config(inner_attention=FlexAttention.Config())`
6. → stored in each layer's `GQAttention.Config.inner_attention`
7. → at runtime: `self.inner_attention = config.inner_attention.build()` instantiates the FlexAttention module

### Config Variants in agpt_configs Dict

Each new variant in `agpt_configs` follows the pattern. Copy existing config and add `attn_backend="flex"`:

```python
# NEW: 7B_flex_attn (line ~500 in agpt_configs dict)
"7B_flex_attn": _build_agpt_config(
    dim=4096,
    n_layers=32,
    n_heads=32,
    n_kv_heads=8,
    rope_theta=10000,
    vocab_size=32000,
    hidden_dim=11008,
    attn_backend="flex",  # NEW: set to "flex"
),
```

**Configs to Add to agpt_configs** (all variants from line 400+):
- `"debugmodel_flex_attn"` - Already exists (line 410-419)
- `"2B_flex_attn"` - Already exists (line 481-490)
- `"2B_qknorm_flex_attn"` - NEW
- `"2B_softcap_flex_attn"` - NEW (compare softcap via SoftcappedFlexAttention)
- `"2B_kitchen_sink_flex_attn"` - NEW
- `"7B_flex_attn"` - NEW
- `"8B_flex_attn"` - NEW
- `"20B_flex_attn"` - Already exists (line 520-529)
- `"50B_flex_attn"` - NEW
- `"50B_wide_flex_attn"` - NEW
- `"70B_wide_flex_attn"` - NEW
- `"80B_flex_attn"` - NEW
- `"80B_wide_flex_attn"` - NEW
- `"80B_deep_flex_attn"` - NEW

---

## Verification Strategy

### End-to-End Test
```bash
# Test with small model first
MODULE=agpt CONFIG=debugmodel_flex_attn \
  ./run_train_torchtitan.sh single --dry-run

# Test actual training (few steps)
MODULE=agpt CONFIG=2B_flex_attn \
  TRAINING_STEPS=10 \
  ./run_train_torchtitan.sh single
```

### Checkpoint Compatibility
- Train with SDPA version 10 steps, save checkpoint
- Load checkpoint in FlexAttention version, train 1 more step
- Verify loss is reasonable (continuity)

### Loss Curve Comparison
- Run debugmodel with both SDPA and FlexAttention for 100 steps
- Compare loss curves (should be within 1-2% variation due to numerical differences)

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| FlexAttention less stable on some hardware | Start testing on known-good hardware (GPU); test XPU separately |
| torch.compile overhead at startup | Document compilation time; note it only happens once per session |
| Correctness regression (loss divergence) | Add loss curve comparison to CI |
| Missing features vs SDPA (e.g., mask_type) | Verify attention_masks are BlockMask type (already done) |

---

## Detailed File Changes

### 1. `torchtitan/models/agpt/__init__.py` - Add Config Variants

**Location:** Lines 400-620 in `agpt_configs` dict

**Change:** Add new config entries with `attn_backend="flex"` parameter to `_build_agpt_config()` calls.

**Example additions (insert in appropriate alphabetical position):**

```python
agpt_configs = {
    "debugmodel": _build_agpt_config(...),
    "debugmodel_flex_attn": _build_agpt_config(..., attn_backend="flex"),  # Already exists
    "2B": _build_agpt_config(...),
    "2B_flex_attn": _build_agpt_config(..., attn_backend="flex"),  # Already exists
    "2B_qknorm": _build_agpt_config(..., qk_norm=True),
    "2B_qknorm_flex_attn": _build_agpt_config(          # NEW
        dim=2048, n_layers=12, n_heads=16, n_kv_heads=4,
        rope_theta=50000, vocab_size=256128, hidden_dim=11008,
        qk_norm=True,
        attn_backend="flex",
    ),
    # ... similar additions for other variants
}
```

**No changes needed to code logic.** The infrastructure already supports `attn_backend="flex"`.

### 2. `torchtitan/models/agpt/config_registry.py` - Add Trainer Config Functions

**Location:** Lines 85+ (near existing flex_attn configs)

**Change:** Add wrapper functions for each new flex_attn variant:

```python
def agpt_2b_qknorm_flex_attn() -> Trainer.Config:
    return agpt("2B_qknorm", attn_backend="flex")

def agpt_7b_flex_attn() -> Trainer.Config:
    return agpt("7B", attn_backend="flex")

def agpt_8b_flex_attn() -> Trainer.Config:
    return agpt("8B", attn_backend="flex")

# ... continue for all new variants
```

These enable users to select configs via:
```bash
CONFIG=agpt_7b_flex_attn ./run_train_torchtitan.sh single
```

---

## Success Criteria

✅ All AGPT model sizes have corresponding `_flex_attn` config variants (12+ new configs)  
✅ FlexAttention configs build without errors  
✅ Training runs (forward + backward) with debugmodel_flex_attn  
✅ Loss curves with FlexAttention are within 2% of SDPA baseline  
✅ Users can select FlexAttention via `CONFIG=agpt_7b_flex_attn`  
✅ Documentation explains the variants and when to use FlexAttention  
