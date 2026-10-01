# Model Conversion: Distcp to Safetensors

**Script Location**: `/lus/flare/projects/datascience/seonghapark/conversion.py`

This script converts models from PyTorch distributed checkpoint (distcp) format to Hugging Face safetensors format, which is compatible with TorchTitan and other HF-based training frameworks.

---

## Quick Start

### Basic Usage

```bash
cd /lus/flare/projects/datascience/seonghapark

python conversion.py \
  --input /path/to/distcp/checkpoint \
  --output /path/to/output/safetensors
```

### Short Form

```bash
python conversion.py -i ./checkpoint -o ./converted
```

---

## Real-World Examples

### Example 1: Convert agpt-2b distcp to safetensors

```bash
cd /lus/flare/projects/datascience/seonghapark

python conversion.py \
  --input ./agpt-2b-v2-256n-step-92859 \
  --output ./agpt-2b-v2-256n-step-92859-safetensors
```

### Example 2: Using absolute paths

```bash
python conversion.py \
  --input /lus/flare/projects/datascience/seonghapark/agpt-2b-v2-256n-step-92859 \
  --output /lus/flare/projects/datascience/seonghapark/agpt-2b-v2-safetensors
```

### Example 3: Convert in a PBS job

```bash
qsub << 'EOF'
#!/bin/bash
#PBS -l select=1
#PBS -l walltime=1:00:00
#PBS -N model_conversion

cd /lus/flare/projects/datascience/seonghapark

python conversion.py \
  --input ./agpt-2b-v2-256n-step-92859 \
  --output ./agpt-2b-v2-safetensors

echo "✅ Conversion complete!"
EOF
```

---

## What the Script Does

The conversion process has 5 steps:

1. **Load Checkpoint** - Reads from distcp format (supports .pt, .bin, __0_0 structure)
2. **Normalize** - Removes distributed training prefixes (module., _orig_mod., etc.)
3. **Load Config** - Extracts or creates model config.json
4. **Save Safetensors** - Writes model.safetensors, config.json, and index file
5. **Summary** - Reports conversion success and output location

---

## Output Structure

After conversion, the output directory will contain:

```
output_dir/
├── model.safetensors           # Model weights in safetensors format
├── config.json                 # Model configuration
└── model.safetensors.index.json # Model index metadata
```

---

## Command-Line Options

```
-i, --input PATH       (Required) Path to distcp checkpoint directory
-o, --output PATH      (Required) Path where to save safetensors model
```

---

## Supported Input Formats

The script can read from multiple distcp formats:

- ✅ `.pt` files in root directory
- ✅ `.bin` files in root directory  
- ✅ Distributed checkpoint structure (`__0_0/` with shards)
- ✅ Safetensors files (if already present)
- ✅ Checkpoints with or without metadata wrapper

---

## Output Format

Generates Hugging Face compatible safetensors with:

- ✅ `model.safetensors` - Binary model weights
- ✅ `config.json` - Model configuration (extracted or minimal)
- ✅ `model.safetensors.index.json` - Metadata and size info

---

## Example Output

```
============================================================
Distcp → Safetensors Converter
============================================================

[1/5] Loading checkpoint...
📂 Loading from: /path/to/checkpoint
🔍 Looking for checkpoint files...
Found 1 checkpoint file(s)
  Loading checkpoint.pt...
    ✓ Extracted 'state_dict' (1000 params)
✅ Loaded 1000 total parameters

[2/5] Normalizing state dict...
🔄 Normalizing state dict...
✅ Normalized 1000 parameters

[3/5] Loading config...
📄 Found config: config.json
✅ Config loaded

[4/5] Saving safetensors...
💾 Saving model weights...
✅ Saved weights (5.20 GB)
💾 Saving config...
✅ Saved config to config.json
💾 Saving model index...
✅ Saved model index

[5/5] Conversion complete!

============================================================
✅ Successfully converted!
   Input:  /path/to/checkpoint
   Output: /path/to/output
============================================================
```

---

## Use After Conversion

### With start_training_with_conversion.sh

```bash
./start_training_with_conversion.sh \
  --model-path /path/to/converted/safetensors \
  --mode multi \
  -- --training.steps 400000
```

### With run_train_torchtitan.sh

```bash
MODEL=/path/to/converted/safetensors \
./xpu_torchtitan/run_train_torchtitan.sh multi
```

---

## Troubleshooting

### Error: Module 'safetensors' not found

**Fix**: Install safetensors in your Python environment
```bash
pip install safetensors
```

### Error: Checkpoint path does not exist

**Fix**: Use absolute path or verify the path exists
```bash
# Check if path exists
ls -la /lus/flare/projects/datascience/seonghapark/agpt-2b-v2-256n-step-92859

# Use absolute path
python conversion.py \
  --input /lus/flare/projects/datascience/seonghapark/agpt-2b-v2-256n-step-92859 \
  --output /lus/flare/projects/datascience/seonghapark/agpt-2b-v2-safetensors
```

### Error: No checkpoint found

**Cause**: Script couldn't find any checkpoint files in the directory

**Fix**: Verify checkpoint structure and format
```bash
# Check what's in the checkpoint directory
ls -la /path/to/checkpoint

# Look for .pt, .bin, or __0_0 subdirectory
find /path/to/checkpoint -name "*.pt" -o -name "*.bin" -o -type d -name "__0_0"
```

### Slow Conversion

**Note**: Conversion speed depends on:
- Model size (larger models take longer)
- Disk I/O performance (network storage slower than local)
- CPU available (torch operations use CPU)

**Typical Times**:
- Small model (1B params): 5-10 minutes
- Medium model (7B params): 15-30 minutes
- Large model (70B params): 30-60 minutes

---

## Advanced Usage

### Running in Background

```bash
# Run in background and save logs
nohup python conversion.py \
  --input /path/to/checkpoint \
  --output /path/to/output \
  > conversion.log 2>&1 &

# Monitor progress
tail -f conversion.log
```

### Using with Screen/Tmux

```bash
# Start screen session
screen -S conversion

# Run conversion
python conversion.py --input ... --output ...

# Detach with Ctrl+A then D
# Reattach later with: screen -r conversion
```

### Batch Conversion

```bash
# Create script: convert_all.sh
#!/bin/bash
for checkpoint in /path/to/checkpoints/*/; do
    output_name=$(basename "$checkpoint")_safetensors
    python conversion.py \
      --input "$checkpoint" \
      --output "./converted/$output_name"
done

# Run it
bash convert_all.sh
```

---

## Performance Notes

- Conversion is **CPU-based** (no GPU needed)
- Uses **single-threaded** loading (safe for all systems)
- Can be run on **login nodes** or in PBS jobs
- Output is **disk I/O bound** (writing safetensors takes time)

---

## File Locations

```
/lus/flare/projects/datascience/seonghapark/
├── conversion.py                          ← This script
├── agpt-2b-v2-256n-step-92859/            ← Example input
└── agpt-2b-v2-safetensors/                ← Example output
    ├── model.safetensors
    ├── config.json
    └── model.safetensors.index.json
```

---

## Related Scripts

- `start_training_with_conversion.sh` - Auto-converts and trains
- `convert_distcp_to_safetensors.py` - Alternative conversion script
- `run_train_torchtitan.sh` - Train with converted model

---

**Last Updated**: 2026-09-26
