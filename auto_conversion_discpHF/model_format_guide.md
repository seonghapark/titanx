# Model Format Conversion Guide

**For Training with xpu_launcher and TorchTitan**

Converting between distributed checkpoint (distcp/DCP) and Hugging Face (HF) formats for model training.

---

## 📋 Overview

This guide explains how to convert and use models in different formats with the xpu_launcher training system.

### Supported Formats

| Format | Name | Use Case | Status |
|--------|------|----------|--------|
| **HF** | Hugging Face | TorchTitan training (recommended) | ✅ Preferred |
| **distcp** | Distributed Checkpoint | Multi-node saves, resuming training | ✅ Supported |

---

## 🚀 Quick Start

### Option 1: Train with HF Format (Recommended)

If your model is already in Hugging Face format:

```bash
./start_training.sh --model-path /path/to/hf/model
```

### Option 2: Auto-Convert distcp to HF

If your model is in distcp format, use the wrapper that auto-converts:

```bash
./start_training_with_conversion.sh --model-path /path/to/dcp/model
```

### Option 3: Manual Conversion

Convert a model manually before training:

```bash
python convert_model_format.py --dcp /path/to/dcp/model --output /path/to/hf/model
./start_training.sh --model-path /path/to/hf/model
```

---

## 📁 Model Format Details

### Hugging Face (HF) Format

**Structure**:
```
model_directory/
├── config.json                 # Model configuration
├── model.safetensors           # Weights (preferred format)
│  or pytorch_model.bin         # or PyTorch weights
├── tokenizer.json              # Tokenizer (if available)
├── tokenizer_config.json       # Tokenizer config
├── special_tokens_map.json     # Special tokens
└── [optional assets]           # Other config files
```

**Characteristics**:
- ✅ Standard format for HuggingFace ecosystem
- ✅ Works directly with TorchTitan
- ✅ Easy to load with `transformers` library
- ✅ Compatible with model hub uploading
- ✅ Optimal for training resume/checkpointing

**Example paths**:
- `/path/to/Llama-3.1-8B/`
- `/path/to/agpt-2b/`
- `/lus/flare/projects/datascience/seonghapark/agpt-2b-v2-256n-step-92859-safetensors/`

---

### Distributed Checkpoint (distcp/DCP) Format

**Structure**:
```
checkpoint_directory/
├── __0_0/                      # Shard 0, replica 0
│   ├── model_state_dict-00001.pt
│   ├── model_state_dict-00002.pt
│   └── ...
├── __0_1/                      # Shard 0, replica 1 (if applicable)
├── metadata.json               # Checkpoint metadata
└── [optional config files]     # Config if available
```

**Characteristics**:
- ✅ Used for distributed/multi-node training saves
- ✅ Supports fault tolerance and resume
- ✅ Sharded format for large models
- ❌ Not directly compatible with HF/TorchTitan
- ❌ Requires conversion for training resume

**Typical sources**:
- Direct saves from distributed training
- Fault-tolerant checkpoint systems
- Prior Aurora/XPU training runs

---

## 🔄 Format Conversion

### Auto-Conversion During Training

The **recommended** approach: use the conversion wrapper script.

```bash
./start_training_with_conversion.sh \
  --model-path /path/to/dcp/or/hf/model \
  [training options...]
```

**What happens**:
1. Script detects model format (distcp or HF)
2. If distcp: automatically converts to HF
3. If HF: uses as-is
4. Starts training with the final model

**Advantages**:
- ✅ Automatic format detection
- ✅ Seamless conversion
- ✅ Single command for all cases
- ✅ Optional cleanup after training

---

### Manual Conversion

For more control, convert models separately:

#### Check Format

```bash
python convert_model_format.py --check-format /path/to/model
```

Output:
- `✓ .../model is in Hugging Face format` → Ready to train
- `✗ .../model is in DCP format (needs conversion)` → Needs conversion
- `? .../model format is unknown` → Check manually

#### Convert Single Model

```bash
python convert_model_format.py \
  --dcp /path/to/dcp/model \
  --output /path/to/hf/model \
  --verbose
```

#### Batch Convert Directory

Convert all models in a directory:

```bash
python convert_model_format.py \
  --batch /source/directory \
  --output /destination/directory \
  --pattern "agpt*" \
  --verbose
```

---

## 📊 Conversion Details

### What Gets Converted

1. **Model Weights**
   - Loads from DCP shards or single checkpoint
   - Normalizes state dict (removes distributed prefixes)
   - Saves as `.safetensors` (or `.bin` if safetensors unavailable)

2. **Configuration**
   - Extracts from `config.json` in DCP
   - Falls back to metadata if available
   - Creates minimal config if none found
   - Saves as `config.json` in HF format

3. **Tokenizer & Assets**
   - Copies `tokenizer.json`, `tokenizer_config.json`
   - Copies special tokens, vocab files
   - Preserves preprocessing config

### Conversion Safety

The converter includes several safety mechanisms:

- ✅ **Format validation**: Checks if output is valid HF format
- ✅ **Error handling**: Graceful failure with detailed errors
- ✅ **State dict normalization**: Handles distributed prefixes
- ✅ **Asset preservation**: Keeps tokenizers and configs
- ✅ **Dry-run mode**: Preview without making changes

---

## 🔧 Usage Examples

### Example 1: Train HF Model Directly

```bash
./start_training.sh \
  --model-path /path/to/Llama-3.1-8B \
  --training-steps 10000
```

### Example 2: Auto-Convert and Train

```bash
./start_training_with_conversion.sh \
  --model-path /path/to/checkpoint.dcp \
  --training-steps 10000
```

### Example 3: Convert Before Training

```bash
# Step 1: Convert
python convert_model_format.py \
  --dcp /path/to/model.dcp \
  --output ./models/model_hf

# Step 2: Train
./start_training.sh \
  --model-path ./models/model_hf \
  --training-steps 5000
```

### Example 4: Multi-Node with Conversion

```bash
./start_training_with_conversion.sh \
  --model-path /checkpoint/agpt_2b.dcp \
  --mode multi \
  --hostfile nodes.txt \
  --training-steps 50000
```

### Example 5: Batch Convert

```bash
# Convert all agpt models in directory
python convert_model_format.py \
  --batch /checkpoints/models \
  --output /data/models_hf \
  --pattern "agpt*"
```

### Example 6: Dry-Run (Preview)

```bash
# Preview conversion without executing
./start_training_with_conversion.sh \
  --model-path /path/to/model \
  --dry-run
```

---

## 📋 Script Options

### `start_training_with_conversion.sh`

```bash
./start_training_with_conversion.sh \
  --model-path PATH              # Model directory (required)
  --output-dir DIR               # Converted models directory
  --no-auto-convert              # Fail if distcp (don't convert)
  --delete-converted             # Delete converted model after training
  --dry-run                       # Preview without executing
  --verbose                       # Detailed logging
  --mode {single|multi}           # Training mode
  --hostfile FILE                # Multi-node hostfile
  --training-steps N              # Number of steps
  [... other training options]
```

### `convert_model_format.py`

```bash
python convert_model_format.py \
  --dcp PATH                     # DCP model to convert
  --output PATH                  # Output HF directory
  [--batch DIR]                  # Batch convert directory
  [--pattern "glob"]             # File pattern for batch
  [--skip-validation]            # Skip validation after conversion
  [--verbose]                    # Detailed output
  [--check-format PATH]          # Check model format and exit
```

---

## 🐛 Troubleshooting

### Issue: "Model is in DCP format but auto-convert is disabled"

**Solution**: Either convert manually or use the auto-conversion wrapper:

```bash
# Option 1: Use auto-conversion wrapper
./start_training_with_conversion.sh --model-path /dcp/model

# Option 2: Convert manually then train
python convert_model_format.py --dcp /dcp/model --output /hf/model
./start_training.sh --model-path /hf/model
```

### Issue: "Could not load DCP checkpoint"

**Possible causes**:
- DCP path is incorrect
- DCP format is corrupted
- Wrong checkpoint structure

**Debug**:
```bash
# Check what's in the directory
ls -la /path/to/dcp/

# Try verbose conversion
python convert_model_format.py --dcp /path/to/dcp --output /tmp/test --verbose
```

### Issue: "No model weights found"

**Possible causes**:
- DCP directory is empty
- Weights have unexpected file extension
- Configuration issue

**Debug**:
```bash
# Check for weight files
find /path/to/dcp -name "*.pt" -o -name "*.bin" -o -name "*.safetensors"

# Check format
python convert_model_format.py --check-format /path/to/dcp
```

### Issue: "Validation failed: Missing config.json"

**Solution**: Converter should create a minimal config, but if it fails:

```bash
# Use skip-validation to bypass
./start_training_with_conversion.sh \
  --model-path /path \
  # (validation is skipped during conversion if config creation works)

# Or check the output directory
ls -la /converted/models/model_hf/
```

### Issue: "Could not load config.json"

**Solution**: Config might be malformed or missing. Converter creates a minimal one automatically.

---

## ⚙️ Environment Variables

Control conversion behavior with environment variables:

```bash
# Use custom Python binary
PYTHON_BIN=/custom/python ./start_training_with_conversion.sh --model-path ...

# Control conversion output location
CONVERT_OUTPUT_DIR=/tmp/models ./start_training_with_conversion.sh --model-path ...

# Disable auto-conversion (fail on distcp)
AUTO_CONVERT=0 ./start_training_with_conversion.sh --model-path ...

# Don't keep converted models after training
KEEP_CONVERTED=0 ./start_training_with_conversion.sh --model-path ...

# Verbose logging during conversion
VERBOSE=1 ./start_training_with_conversion.sh --model-path ...
```

---

## 📊 Performance Notes

### Conversion Time

- **Small model (2B params)**: ~5-30 seconds
- **Medium model (8B params)**: ~30-60 seconds
- **Large model (70B params)**: ~2-5 minutes

*Depends on disk speed, model size, and system load*

### Disk Space Requirements

- **Input DCP**: Original size
- **Output HF**: Similar to DCP (maybe 5-10% larger due to safetensors overhead)
- **Both during conversion**: ~2x model size (need space for both)

**Example**: 2B model ~10GB → need ~20GB free space

---

## 🔍 Technical Details

### State Dict Normalization

The converter removes common distributed training prefixes:

```python
Prefix removal (in order):
- "module."           # DataParallel wrapper
- "_orig_mod."        # Compiled model wrapper
- "model."            # Common model wrapper
- "state_dict."       # Checkpoint wrapper
```

**Example**:
```
Input:  "module.transformer.layer.0.weight"
Output: "transformer.layer.0.weight"
```

### Config Inference

If no config is found, converter infers from state dict:

```python
- Hidden size: from embeddings or first weight shape
- Default architecture: transformer-like
- Default layers: 12 hidden layers
- Default heads: 12 attention heads
```

This creates a minimal but valid config. For best results, ensure DCP has `config.json`.

---

## 📚 Integration Points

### With Training Scripts

The conversion wrapper (`start_training_with_conversion.sh`) passes all arguments to the training script:

```bash
./start_training_with_conversion.sh \
  --model-path /model \
  --mode multi \           # ← passed to training script
  --hostfile nodes.txt \   # ← passed to training script
  --training-steps 5000    # ← passed to training script
```

### With TorchTitan

TorchTitan expects models in HF format via `--hf_assets_path`:

```bash
# Internally handled by training scripts
python torchtitan/train.py \
  --hf_assets_path /converted/model \  # Must be HF format
  --module llama3 \
  --config llama3_8b
```

---

## ✅ Validation Checklist

After conversion, verify the model is ready:

- [ ] `config.json` exists and is valid JSON
- [ ] Model weights present (`.safetensors` or `.bin`)
- [ ] Tokenizer files present (if applicable)
- [ ] Can load with: `torch.load()` or `safetensors.torch.load_file()`
- [ ] Config contains `hidden_size`, `num_layers` (for inference/eval)

**Quick validation**:
```bash
python convert_model_format.py --check-format /path/to/converted/model
```

---

## 🎓 Common Patterns

### Pattern 1: Resume Training from Checkpoint

```bash
# Checkpoint from prior run (in DCP format)
CKPT=/checkpoints/step-10000.dcp

# Convert if needed
python convert_model_format.py \
  --dcp "$CKPT" \
  --output ./ckpt_hf

# Resume training
./start_training.sh \
  --model-path /original/hf/model \
  --log-dir ./resume_run \
  -- --checkpoint.initial_load_path ./ckpt_hf
```

### Pattern 2: Fine-tune Converted Model

```bash
# Convert pre-trained checkpoint
python convert_model_format.py \
  --dcp /checkpoints/pretrained.dcp \
  --output ./models/pretrained_hf

# Fine-tune on new dataset
./start_training.sh \
  --model-path ./models/pretrained_hf \
  --dataset-name custom_dataset \
  --training-steps 1000
```

### Pattern 3: Distributed Training with Auto-Conversion

```bash
./start_training_with_conversion.sh \
  --model-path /checkpoints/model.dcp \
  --mode multi \
  --hostfile nodes.txt \
  --spare-nodes 2 \
  --training-steps 100000
```

---

## 📖 Related Documentation

- `TRAINING_GUIDE.md` - Complete training guide
- `start_training.sh` - Direct training (HF models)
- `start_training_with_conversion.sh` - Training with conversion
- `convert_model_format.py` - Detailed converter source code
- `README_launcher_torchtitan.md` - TorchTitan integration

---

## 🆘 Support

For issues or questions:

1. Check `Troubleshooting` section above
2. Review example use cases in "Usage Examples"
3. Check detailed converter script documentation
4. Consult training guides for model-specific issues

---

**Last Updated**: 2026-09-26  
**Supported Formats**: distcp → HF, HF (pass-through)  
**TorchTitan Compatibility**: ✅ Full support via HF format
