# Hyperparameter Analysis Summary

## Overview

Analyzed **66 hyperparameters** across all ML training categories. Current coverage:
- **✓ Fully tracked & printed: 23 parameters (34%)**
- **⚠️ Available but not pre-printed: 7 parameters (10%)**  
- **❌ Missing/Not tracked: 36 parameters (54%)**

## Scripts Created

### 1. `hyperparameters.py` (Full-featured)
- Requires TorchTitan environment
- Extracts all hyperparameters from Trainer.Config
- Detects RoPE backend and YaRN parameters
- Can load from JSON config
- Outputs as text or JSON

**Usage:**
```bash
python hyperparameters.py --missing-only          # Show what's not tracked
python hyperparameters.py --format json           # Output as JSON
```

### 2. `hyperparameters_standalone.py` (Works anywhere)
- No TorchTitan dependencies required
- Generates comprehensive status report
- Shows all hyperparameters with tracking status
- Provides recommendations for improvement

**Usage:**
```bash
python hyperparameters_standalone.py                    # Show missing only
python hyperparameters_standalone.py --full             # Full report
python hyperparameters_standalone.py --recommendations  # Recommendations only
```

## Currently Printed Hyperparameters (23 Total)

### Pre-Training Launch Config

From `run_train_torchtitan.sh` sections:

**[LAUNCH MODE]** (5)
- NNODES (number of nodes)
- NPROC_PER_NODE (processes per node)
- NPROC (total processes)
- Auto Retry (enabled/disabled)
- Spare Nodes (count)

**[MODEL & DATASET]** (5)
- Model Path
- Module/Config (e.g., agpt_yarn / agpt_2b_yarn)
- Dataset Name
- Dataset Path (if set)
- HF Assets Path

**[TRAINING HYPERPARAMETERS]** (5)
- Training Steps
- Sequence Length
- Loss Std Termination (enabled/disabled)
- Loss Std Threshold
- Loss Std Window

**[SYSTEM & PATHS]** (3)
- TorchTitan Root
- Log Directory
- Checkpoint Folder
- Python Binary

**[MONITORING]** (3)
- Resource Monitor (enabled/disabled)
- Resource Interval (seconds)
- Resource Output Directory

**[EXTRA ARGUMENTS]** (varies)
- Command-line arguments after `--`

### During Training (Printed by TorchTitan)

TorchTitan prints full config at startup, including:
- Learning rate (optimizer.lr)
- Optimizer type (AdamW)
- Betas, eps, weight_decay
- Model architecture (num_layers, hidden_size, num_heads, vocab_size)
- Parallelism settings (DP, TP, PP degrees)
- Precision settings (dtype, mixed precision)
- Training settings (batch size, max_norm, etc.)
- Checkpoint and compile settings

---

## NOT Currently Printed (36 Parameters)

### Optimization (7 missing)
❌ learning_rate — In config but not pre-printed  
❌ optimizer_type — Hardcoded as AdamW  
❌ momentum — Not exposed  
❌ betas — In default_adamw() but not pre-printed  
❌ weight_decay — In optimizer but not pre-printed  
❌ gradient_clipping_max_norm — In training.max_norm but NOT printed  
❌ optimizer_epsilon — Not printed pre-training

### Training Schedule (1 missing, 2 partial)
❌ epochs — Implicit via steps, no explicit epochs  
⚠️ gradient_accumulation_steps — Derived at runtime, not pre-computed  
⚠️ lr_scheduler_type — Fixed as linear, not shown

### Model Architecture (3 missing)
❌ activation_function — Hardcoded in model  
❌ dropout_rate — Hardcoded in model  
❌ normalization_type — RMSNorm, not printed

### Initialization (3 missing)
❌ weight_initialization_method — TorchTitan default  
❌ bias_initialization — TorchTitan default  
❌ random_seed — Not exposed in config

### Regularization (4 missing, 1 partial)
❌ dropout — Hardcoded in model  
❌ label_smoothing — Not implemented  
❌ data_augmentation — Dataset-specific  
⚠️ weight_decay — In optimizer, not pre-printed

### Loss (3 missing)
❌ loss_weights — Single loss, not used  
❌ class_weights — Not implemented  
❌ auxiliary_losses — Not in base config

### Data (5 missing)
❌ dataset_size — Not tracked  
❌ sampling_strategy — Dataset-specific  
❌ shuffle — Dataset-specific  
❌ train_val_split — Hardcoded (5% validation)  
❌ preprocessing/augmentation — Dataset-specific

### Batching (2 missing, 1 partial)
❌ padding_strategy — Dataset-specific  
❌ dynamic_batching — Not supported  
⚠️ global_batch_size — Computed at runtime

### Context Extension / RoPE (5 missing)
❌ rope_backend_type — Available but NOT printed  
❌ context_extension_technique — YaRN/Linear/None not printed  
❌ yarn_alpha — Available if YaRN used, NOT printed  
❌ yarn_beta — Available if YaRN used, NOT printed  
❌ rope_scaling_method — Available but NOT printed

### Model Provenance (3 missing, 1 partial)
❌ current_model_name — Set via MODULE/CONFIG, not printed  
❌ base_model_name — Implicit (Llama3), not printed  
❌ original_model_name — Not tracked  
⚠️ initialization_checkpoint — Set via CKPT env var, not printed

---

## Key Findings

### What's Available but Not Pre-Printed (7 parameters)

1. **Global Batch Size** — Computed as `local_batch_size × data_parallel_degree` at runtime
2. **Gradient Accumulation Steps** — Derived from global_batch_size / (local_batch_size × DP degree)
3. **Weight Decay** — In optimizer config, shown during training
4. **Intermediate Size** — Available in model.config
5. **LR Scheduler Type** — Fixed as linear in base config, not shown
6. **Initialization Checkpoint** — Passed via CKPT env var but not echoed
7. **Gradient Scaling** — Handled by torch.autocast, not explicit

### What's Hardcoded/Not Configurable (9 parameters)

1. **Activation Function** — ReLU/SiLU hardcoded in architecture
2. **Dropout Rate** — Hardcoded in model layers
3. **Normalization Type** — RMSNorm in Llama/AGPT models
4. **Weight Init Method** — TorchTitan's default kaiming/normal
5. **Bias Init Method** — TorchTitan's default (usually zeros)
6. **Random Seed** — Not exposed in current config
7. **Optimizer Type** — Fixed as AdamW in default_adamw()
8. **Label Smoothing** — Not implemented in CrossEntropyLoss
9. **Dynamic Batching** — Not supported

### Dataset-Specific (5 parameters)

These are defined at the DataLoader level, not in Trainer.Config:
1. Dataset Size
2. Sampling Strategy
3. Shuffle
4. Train/Val Split (hardcoded as 5%)
5. Augmentation/Preprocessing

---

## Current Model Information (Your AGPT-2B Setup)

From your command:
```bash
MODULE=agpt_yarn
CONFIG=agpt_2b_yarn
MODEL=/lus/flare/.../agpt-2b-v2-256n-step-92859-safetensors
```

### What's Printed ✓
- Model class: AgptModel
- Module/Config: agpt_yarn / agpt_2b_yarn
- Model path (HF assets)
- HF assets path
- Dataset: PG19
- Dataset path: /lus/flare/.../pg19/data

### What's NOT Printed ❌
- Base model: **Llama3** (implicit, not shown)
- Current model: **AGPT-2B** (known from config but not echoed)
- Original model: **AGPT-2B** (same as current, loaded from CKPT)
- Context extension: **YaRN** (should be detected from agpt_yarn, but not printed)
- YaRN parameters (alpha, beta) — Not printed

---

## Recommendations by Priority

### HIGH PRIORITY

1. **Create Model Metadata Tracking**
   ```bash
   # Add to launch config
   [MODEL PROVENANCE]
     Current Model Name      = AGPT-2B (from MODULE=agpt_yarn CONFIG=agpt_2b_yarn)
     Base Model              = Llama3 (AgptModel extends Llama3Model)
     Original Model Path     = /path/to/checkpoint
     Initialization          = Checkpoint (CKPT) or Random
   ```

2. **Add RoPE/Context Extension Detection**
   ```bash
   # Add to launch config
   [CONTEXT EXTENSION]
     RoPE Backend            = ComplexRoPE or CosSinRoPE
     Context Technique       = YaRN (from agpt_yarn) or None
     YaRN Alpha (if used)    = 32 (example)
     YaRN Beta (if used)     = 1 (example)
   ```

3. **Add Random Seed to Config**
   - Currently not exposed
   - Add to Trainer.Config
   - Print in launch config

### MEDIUM PRIORITY

4. **Print Optimizer Settings Pre-Training**
   ```bash
   # Add to launch config
   [OPTIMIZATION]
     Learning Rate           = 8e-4 (or override value)
     Optimizer Type          = AdamW
     Betas (momentum)        = (0.9, 0.95)
     Weight Decay            = 0.1 (example)
     Optimizer Epsilon       = 1e-8
     Max Norm (grad clip)    = 1.0
   ```

5. **Add Model Architecture Details**
   ```bash
   # Add to launch config
   [MODEL ARCHITECTURE]
     Activation Function     = SiLU
     Dropout Rate            = 0.0
     Normalization Type      = RMSNorm
     Vocabulary Size         = 128256 (or model-specific)
   ```

6. **Track Initialization Checkpoint**
   - Echo CKPT env var value if set
   - Show whether training from random init or loaded weights

### LOW PRIORITY

7. **Data Augmentation Documentation**
   - Dataset-specific, could add to launch config
   - Document in LOG_DIR/dataset_config.txt

8. **Batch Size Derivatives**
   - Pre-compute and print:
     - Global batch size
     - Gradient accumulation steps
     - Effective batch size per device

---

## Integration Guide

### For Your Immediate Use

1. **Run the standalone report:**
   ```bash
   python hyperparameters_standalone.py --full > hyperparams_report.txt
   ```

2. **Check what's missing for your AGPT training:**
   ```bash
   python hyperparameters_standalone.py | grep "rope\|yarn\|model_name"
   ```

3. **Generate before each training run:**
   ```bash
   python hyperparameters_standalone.py --full > $LOG_DIR/hyperparameter_status.txt
   ```

### For Systematic Improvement

1. **Phase 1** (Next): Add RoPE/YaRN detection to launch config
2. **Phase 2** (Next sprint): Add model provenance tracking (current/base/original names)
3. **Phase 3** (Next quarter): Extend Trainer.Config with optional metadata field
4. **Phase 4**: Add random seed and initialization method to config

---

## Files Generated

1. **`hyperparameters.py`**
   - Full-featured version with TorchTitan integration
   - Extracts from Trainer.Config
   - Detects RoPE/YaRN parameters
   - Location: `/lus/flare/projects/datascience/seonghapark/xpu_launcher/`

2. **`hyperparameters_standalone.py`**
   - Works without TorchTitan environment
   - Generates comprehensive status report
   - Ready to run with `python3`
   - Location: `/lus/flare/projects/datascience/seonghapark/xpu_launcher/`

3. **`HYPERPARAMETER_TRACKING_STATUS.md`**
   - Detailed tracking status document
   - Recommendations for each missing parameter
   - Implementation guide

4. **`HYPERPARAMETER_ANALYSIS_SUMMARY.md`** (this file)
   - Executive summary
   - Quick reference tables
   - Priority-ranked recommendations

---

## Quick Reference: 66 Hyperparameters Analyzed

```
✓  Fully tracked:        23 (34%)
⚠️  Partial/Available:    7 (10%)
❌ Missing/Not tracked:  36 (54%)
```

**Run this to see the full status:**
```bash
python3 hyperparameters_standalone.py
```
