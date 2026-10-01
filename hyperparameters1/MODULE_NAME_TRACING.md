# Module Name Tracing: agpt_2b_yarn Call Flow

## Summary
When `MODULE=agpt CONFIG=agpt_2b_yarn` is specified, here's the complete call flow from command-line to model configuration.

## Is Module Name Currently Logged?

### ❌ **NOT in Python Trainer Logs**
The module name is **not logged** in the `_log_training_hyperparameters()` method in `trainer.py`.

### ✅ **YES in Shell Script Output**
The module name **is logged** in `run_train_torchtitan.sh`:
- **Line 271**: Printed as `Module/Config = llama3 / llama3_debugmodel`

### ✨ **RECOMMENDATION**
The module name should be added to the Python trainer logs for complete transparency. It would be helpful to log it in the `[MODEL]` section.

---

## Complete Call Flow: `agpt_2b_yarn`

### Step 1: User Input
```bash
MODEL=/path/to/model CONFIG=agpt_2b_yarn MODULE=agpt ./run_train_torchtitan.sh single
```

### Step 2: Shell Script (run_train_torchtitan.sh)
```bash
MODULE="${MODULE:-llama3}"  # Line 170
CONFIG="${CONFIG:-llama3_debugmodel}"  # Line 171

# Passes to trainer as:
TRAIN_CMD=(
  "$TRAIN_PYTHON_BIN" "${SCRIPT_DIR}/titan_train.py"
  "--module" "$MODULE"    # "agpt"
  "--config" "$CONFIG"    # "agpt_2b_yarn"
  ...
)
```

**Logged at line 271**:
```
[MODEL & DATASET]
  Module/Config          = agpt / agpt_2b_yarn
```

### Step 3: Trainer Entry Point (titan_train.py)
Passes command-line args to torchtitan.train.main()

### Step 4: TorchTitan Config Manager
The `ConfigManager` parses CLI args:
- `--module agpt`
- `--config agpt_2b_yarn`

Uses `tyro` to dynamically load the config function.

### Step 5: Config Registry Lookup

**Module**: `agpt`  
**File**: `torchtitan/models/agpt/config_registry.py`

When `--config agpt_2b_yarn` is specified:

```python
# config_registry.py line 404
def agpt_2b_yarn(seq_len: int = 32768) -> Trainer.Config:
    """Continue-pretrain agpt-2b with YaRN-scaled RoPE for context extension."""
    return agpt("2b_yarn", seq_len=seq_len, activation_checkpoint_mode="none")
```

**Calls**: `agpt()` function with flavor="2b_yarn"

### Step 6: Main Config Builder (agpt function)

**File**: `config_registry.py` line 107

```python
def agpt(
    flavor: str,  # "2b_yarn" passed here
    local_batch_size: int = 1,
    activation_checkpoint_mode: Literal["none", "full"] = "full",
    seq_len: int = 8192,
    ...
) -> Trainer.Config:
    cfg = _base_config(flavor)  # Line 127
    ...
    return cfg
```

**Calls**: `_base_config("2b_yarn")`

### Step 7: Base Config Setup

**File**: `config_registry.py` line 154

```python
def _base_config(flavor: str) -> Trainer.Config:
    return Trainer.Config(
        model_spec=model_registry(flavor),  # Line 157: "2b_yarn"
        tokenizer=HuggingFaceTokenizer.Config(),
        ...
    )
```

**Calls**: `model_registry("2b_yarn")`

### Step 8: Model Registry Lookup

**File**: `torchtitan/models/agpt/__init__.py` line 688

```python
def model_registry(
    flavor: str,  # "2b_yarn"
    attn_backend: str = "sdpa",
) -> ModelSpec:
    from torchtitan.distributed.pipeline_parallel import pipeline_llm
    
    config = agpt_configs[flavor]  # Line 694: Gets "2B_yarn" config
    
    return ModelSpec(
        name="agpt",
        flavor=flavor,  # "2b_yarn"
        model=config,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=None,
        state_dict_adapter=Llama3StateDictAdapter,
    )
```

### Step 9: Config Dictionary Lookup

**File**: `__init__.py` line 400 (agpt_configs definition)

```python
agpt_configs = {
    "debugmodel": _build_agpt_config(...),
    "2B": _build_agpt_config(...),
    ...
    "2B_yarn": _build_agpt_config(                    # Line 630
        dim=2048,
        n_layers=12,
        n_heads=16,
        n_kv_heads=4,
        rope_theta=50000,
        vocab_size=256128,
        hidden_dim=11008,
        rope_backend="cos_sin",
        scaling="yarn",                # ← YaRN enabled here!
        max_seq_len=262144,
        original_seq_len=8192,
        rope_factor=32.0,
        beta_fast=1.0,
        beta_slow=32.0,
    ),
    ...
}
```

**Also** (line 684):
```python
agpt_configs["2b_yarn"] = agpt_configs["2B_yarn"]  # Lowercase alias
```

### Step 10: Build Config via _build_agpt_config

```python
def _build_agpt_config(
    dim: int,
    n_layers: int,
    n_heads: int,
    n_kv_heads: int | None,
    rope_theta: int,
    vocab_size: int,
    hidden_dim: int,
    rope_backend: str = "complex",
    scaling: str | None = None,
    max_seq_len: int = 32768,
    original_seq_len: int | None = None,
    rope_factor: float | None = None,
    beta_fast: float | None = None,
    beta_slow: float | None = None,
    ...
) -> Transformer.Config:
    """Returns Transformer.Config with specified architecture."""
    
    # Creates RoPE config with YaRN scaling:
    # scaling="yarn" enables YaRN rope scaling
    # max_seq_len=262144, original_seq_len=8192
```

---

## Key Architecture Parameters for agpt_2b_yarn

| Parameter | Value | Notes |
|-----------|-------|-------|
| **dim** | 2048 | Hidden dimension |
| **n_layers** | 12 | Number of transformer layers |
| **n_heads** | 16 | Number of attention heads |
| **n_kv_heads** | 4 | Number of KV heads (GQA) |
| **vocab_size** | 256128 | Vocabulary size |
| **hidden_dim** | 11008 | FFN intermediate dimension |
| **rope_theta** | 50000 | Base RoPE frequency |
| **rope_backend** | "cos_sin" | RoPE implementation (real-valued) |
| **scaling** | "yarn" | **YaRN rope scaling enabled** |
| **max_seq_len** | 262144 | Max sequence length with YaRN |
| **original_seq_len** | 8192 | Original pre-training seq_len |
| **rope_factor** | 32.0 | YaRN scaling factor |
| **beta_fast** | 1.0 | YaRN fast-dim scaling |
| **beta_slow** | 32.0 | YaRN slow-dim scaling |

---

## Module Name Detection

Looking at the call flow, the **module name** (`agpt`) and **config name** (`agpt_2b_yarn` or `2b_yarn`) are available at multiple points:

1. **Shell Script** - Has both (uses them)
2. **titan_train.py** - Has both (in sys.argv)
3. **ConfigManager** - Has both (parses CLI args)
4. **Trainer.__init__** - Has access via `config.model_spec`:
   - `config.model_spec.name` = "agpt"
   - `config.model_spec.flavor` = "2b_yarn"

Currently logged in `_log_training_hyperparameters()`:
- ✅ `config.model_spec.name` (line 798) - logs as "Model Name"
- ✅ `config.model_spec.flavor` (line 799) - logs as "Model Flavor"

So the information is already available! The module name is implicitly the **model name** in the current logging.

---

## What's Currently Logged

The `[MODEL]` section in trainer.py already logs:
```python
logger.info(f"  Model Name: {config.model_spec.name}")        # "agpt"
logger.info(f"  Model Flavor: {config.model_spec.flavor}")    # "2b_yarn"
```

This is **equivalent** to:
- Module: agpt
- Config: agpt_2b_yarn (or 2b_yarn)

---

## Enhancement Suggestion

To make it **even clearer** and match the shell script output exactly, we could log it as:

```python
logger.info("\n[MODEL]")
logger.info(f"  Module: {config.model_spec.name}")           # "agpt"
logger.info(f"  Config: {config.model_spec.flavor}")         # "2b_yarn"
logger.info(f"  Architecture: {self.model_config.__class__.__name__}")
```

This would make the Python trainer output match the shell script format:
- **Shell**: `Module/Config = agpt / agpt_2b_yarn`
- **Trainer**: `Module: agpt` + `Config: 2b_yarn`

