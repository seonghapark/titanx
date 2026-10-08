# AGPT Parameter Calculator Implementation Summary

## Overview

Successfully implemented two new scripts that extract and generalize AGPT training parameters from `train_agpt_*_venv.sh` scripts, making them reusable with the flexible `run_train_torchtitan.sh` launcher.

## What Was Created

### 1. `calculate_agpt_params.sh` - Parameter Calculator Script
**Location:** `/lus/flare/projects/datascience/seonghapark/xpu_launcher/xpu_torchtitan/calculate_agpt_params.sh`

**Purpose:** Calculate pre-optimized AGPT training parameters based on model size and cluster topology.

**Key Features:**
- ✅ Supports 2B, 20B, and 80B models with correct model-specific defaults
- ✅ Auto-detects node count from PBS_NODEFILE (for PBS jobs) or accepts --nnodes
- ✅ Calculates Global Batch Size (GBS): `(NGPUS × LBS × GAS) / (TP × PP × CP)`
- ✅ Calculates Training Steps: `TRAIN_TOKENS / (GBS × SEQ_LEN)`
- ✅ Outputs environment variables with `AGPT_` prefix
- ✅ Supports shell and JSON output formats
- ✅ Includes dry-run mode for validation
- ✅ Full parameter override capability
- ✅ Comprehensive error handling and validation

**Model-Specific Defaults:**

| Model | TP | PP | CP | LBS | GAS | Reason |
|-------|----|----|----|----|-----|--------|
| 2B | 1 | 1 | 1 | 2 | 1 | Memory efficient, pure data parallelism |
| 20B | 1 | 1 | 1 | 2 | 1 | Memory efficient, pure data parallelism |
| 80B | 2 | 1 | 1 | 1 | 1 | Requires TP=2 to fit in GPU memory |

**Verified Outputs (8-node test):**
```
2B:  GBS=192, TRAINING_STEPS=2,971,509
20B: GBS=192, TRAINING_STEPS=2,971,509  
80B: GBS=48,  TRAINING_STEPS=11,886,037  (different TP/LBS!)
```

### 2. `run_train_agpt.sh` - Wrapper Script
**Location:** `/lus/flare/projects/datascience/seonghapark/xpu_launcher/xpu_torchtitan/run_train_agpt.sh`

**Purpose:** Simple user-facing wrapper around `run_train_torchtitan.sh` for AGPT training.

**Key Features:**
- ✅ Auto-calculates parameters via `calculate_agpt_params.sh`
- ✅ Automatically detects NNODES from PBS_NODEFILE
- ✅ Validates model path before launching
- ✅ Supports single-node and multi-node training modes
- ✅ Works with and without hostfile
- ✅ Passes through extra training arguments
- ✅ Comprehensive configuration summary before launch
- ✅ Dry-run mode for verification

**Usage Examples:**
```bash
# Single-node 2B training
./run_train_agpt.sh --model 2b single

# Multi-node 20B on 32 nodes with hostfile
./run_train_agpt.sh --model 20b --nnodes 32 multi /path/to/hosts.txt

# In PBS job (auto-detects nodes)
./run_train_agpt.sh --model 80b multi

# With parameter overrides
./run_train_agpt.sh --model 2b --nnodes 8 --tp 2 --lbs 1 single

# Dry-run to verify
./run_train_agpt.sh --model 2b single --dry-run
```

### 3. `AGPT_TRAINING.md` - Comprehensive User Documentation
**Location:** `/lus/flare/projects/datascience/seonghapark/xpu_launcher/xpu_torchtitan/AGPT_TRAINING.md`

**Contents:**
- Quick start examples
- Detailed parameter calculation formulas
- Model-specific default explanations
- Common usage patterns (1-node, 8-node, 512-node)
- Advanced parameter customization
- PBS job submission examples
- Troubleshooting guide
- Parameter reference tables
- Integration with run_train_torchtitan.sh

## Technical Implementation

### Architecture

```
User invokes run_train_agpt.sh
    ↓
Wrapper parses arguments (--model, --nnodes, --tp, etc.)
    ↓
Calls calculate_agpt_params.sh with arguments
    ↓
Calculator:
  - Auto-detects NNODES from PBS_NODEFILE if available
  - Loads model-specific defaults
  - Validates inputs
  - Calculates GBS and TRAINING_STEPS
    ↓
Calculator outputs AGPT_* environment variables
    ↓
Wrapper:
  - Sources variables
  - Sets MODEL, MODULE, CONFIG for run_train_torchtitan.sh
  - Builds command with calculated parameters
  - Prints configuration summary
    ↓
Executes: run_train_torchtitan.sh single/multi [hostfile] -- <args>
```

### Parameter Flow

```
Raw Inputs:
  --model 2b --nnodes 8 [--tp 1 --lbs 2 --gas 1 ...]
  
Calculator Processing:
  1. Validate inputs
  2. Load defaults: {2b: TP=1, LBS=2}, {80b: TP=2, LBS=1}
  3. Override with user values
  4. Calculate NGPUS = NNODES × 12
  5. Calculate GBS = (96 × 2 × 1) / (1 × 1 × 1) = 192
  6. Calculate TRAINING_STEPS = 4.67T / (192 × 8192) ≈ 2.97M

Output Environment Variables:
  AGPT_MODEL=2b
  AGPT_NNODES=8
  AGPT_NGPUS=96
  AGPT_TP=1
  AGPT_LBS=2
  AGPT_GBS=192
  AGPT_TRAINING_STEPS=2971509
  AGPT_CKPT_DIR=agpt-2b-sophiag-olmo-mix-1124-n8-gbs192
  ... (18 total variables)
```

## Verification Tests Performed

### Unit Tests - Parameter Calculations ✓

**Test 1: 2B on 8 nodes**
```
Expected: GBS=192, STEPS≈2.97M
Actual:   GBS=192, STEPS=2,971,509 ✓
```

**Test 2: 80B on 8 nodes (different defaults)**
```
Expected: TP=2, LBS=1, GBS=48, STEPS≈11.9M
Actual:   TP=2, LBS=1, GBS=48,  STEPS=11,886,037 ✓
```

**Test 3: Model-specific defaults**
```
2B:  TP=1, LBS=2 ✓
20B: TP=1, LBS=2 ✓
80B: TP=2, LBS=1 ✓
```

**Test 4: JSON output format**
```
./calculate_agpt_params.sh --model 2b --nnodes 4 --output json
Output: Valid JSON with all parameters ✓
```

### Integration Tests - Wrapper Script ✓

**Test 1: Dry-run with wrapper**
```
./run_train_agpt.sh --model 2b --model-path /tmp/test single --dry-run
Output: Configuration summary + command preview ✓
```

**Test 2: CONFIG variable set correctly**
```
Expected: CONFIG=agpt_2b
Actual:   Module/Config = agpt / agpt_2b ✓
```

**Test 3: Model path validation**
```
Error on missing path: "Model path not found: ..." ✓
```

### Edge Cases ✓

- Single-node (1 node, GBS=24) ✓
- Large scale (512 nodes, GBS=12,288) ✓
- Parameter override (--tp 4 --lbs 2) ✓
- Invalid inputs (error messages clear) ✓

## How It Solves the Original Problem

### Before Implementation
- AGPT parameters hardcoded in train_agpt_*_venv.sh scripts
- Only accessible via PBS jobs
- Must manually recreate calculations to use with run_train_torchtitan.sh
- No reuse across deployment methods

### After Implementation
- Parameters extracted into reusable calculator
- Works interactively or in PBS jobs
- Integrates seamlessly with run_train_torchtitan.sh via wrapper
- Supports all 3 models (2B, 20B, 80B)
- Fully customizable (any parameter can be overridden)
- Backward compatible (run_train_torchtitan.sh unchanged)

## Usage Recommendations

### For Quick Experiments
Use the wrapper:
```bash
./run_train_agpt.sh --model 2b single
./run_train_agpt.sh --model 2b multi hosts.txt
```

### For Maximum Flexibility
Use the calculator directly:
```bash
eval $(./calculate_agpt_params.sh --model 2b --nnodes 8)
# Then combine with run_train_torchtitan.sh and custom args
```

### For Custom Batch Sizes
Override in wrapper:
```bash
./run_train_agpt.sh --model 2b --nnodes 8 --lbs 4 --gas 2 multi hosts.txt
# GBS = (96 × 4 × 2) / 1 = 768
```

## Files Changed/Created

### New Files Created
1. ✅ `calculate_agpt_params.sh` (350 lines)
2. ✅ `run_train_agpt.sh` (310 lines)
3. ✅ `AGPT_TRAINING.md` (650 lines)
4. ✅ `IMPLEMENTATION_SUMMARY.md` (this file)

### Files Modified
- None. All changes are backward compatible additions.

### File Permissions
- ✅ `calculate_agpt_params.sh` - executable (755)
- ✅ `run_train_agpt.sh` - executable (755)

## Next Steps / Future Enhancements

### Optional Enhancements (Out of Scope)
1. Extend calculator to support other models (Llama, DeepSeek-MoE, etc.)
2. Add checkpoint resumption support with auto-detected step counts
3. Create automated benchmark suite comparing GBS across scales
4. Add WANDB integration templates
5. Create LaTeX output format for papers/reports

### Maintenance
- Keep calculator synchronized with new AGPT model sizes
- Update documentation as production parameters change (LR, optimizer tuning)
- Monitor for new parallelism techniques (EP, distributed CP) and add support

## Design Decisions

### 1. **Separate Calculator and Wrapper**
- **Decision:** Two scripts instead of one monolithic script
- **Rationale:** Allows independent use of calculator for research; wrapper provides convenience
- **Benefit:** Flexibility + simplicity

### 2. **Environment Variables with AGPT_ Prefix**
- **Decision:** `AGPT_GBS`, `AGPT_TRAINING_STEPS`, not generic names
- **Rationale:** Clear scope, no conflicts with other tools, self-documenting
- **Benefit:** Obvious what belongs to AGPT vs. other config

### 3. **No Changes to run_train_torchtitan.sh**
- **Decision:** New scripts wrap existing launcher, don't modify it
- **Rationale:** Backward compatible, reduces risk, existing users unaffected
- **Benefit:** Zero breaking changes

### 4. **Model-Specific Lookup Table**
- **Decision:** Hardcode TP/LBS per model rather than machine-learning them
- **Rationale:** Production values are hand-tuned and validated; rules are simple
- **Benefit:** Transparent, auditable, matches original scripts exactly

## QA Sign-Off

✅ All unit tests pass  
✅ All integration tests pass  
✅ Model-specific defaults validated  
✅ Error handling tested  
✅ Documentation complete  
✅ Scripts executable  
✅ Backward compatible  
✅ Ready for production use  

