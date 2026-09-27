# Dataset Path Configuration with local_files_only=True

**Date**: 2026-09-26  
**Topic**: How to provide dataset path when using `local_files_only=True`

---

## Important Clarification

**`local_files_only=True` applies ONLY to the tokenizer loading**, not the dataset.

The dataset is handled separately and independently of the tokenizer's `local_files_only` setting.

---

## How to Provide Dataset Path

### Two Environment Variables

| Variable | Purpose | Required | Default |
|----------|---------|----------|---------|
| `DATASET_NAME` | Dataset identifier (name in HuggingFace Hub or config) | Yes | `pg19_multinews` |
| `DATASET_PATH` | Optional: Custom path or HuggingFace Hub path override | No | Empty (use DATASET_NAME) |

### Location in Script

**File**: `run_train_torchtitan.sh` (lines 173-174, 220-222)

```bash
DATASET_NAME="${DATASET_NAME:-pg19_multinews}"
DATASET_PATH="${DATASET_PATH:-}"

# ...later in command building:
TRAIN_CMD+=("--dataloader.dataset" "$DATASET_NAME")

if [[ -n "$DATASET_PATH" ]]; then
  TRAIN_CMD+=("--dataloader.dataset_path" "$DATASET_PATH")
fi
```

---

## Usage Examples

### Example 1: Default Dataset (pg19_multinews)
```bash
./xpu_torchtitan/run_train_torchtitan.sh multi ./hostfile.txt \
  -- --training.steps 100
# Uses DATASET_NAME=pg19_multinews (default)
# No custom DATASET_PATH
```

### Example 2: Specify Different Dataset Name
```bash
DATASET_NAME=c4 ./xpu_torchtitan/run_train_torchtitan.sh multi ./hostfile.txt \
  -- --training.steps 100
# Uses DATASET_NAME=c4
# Loads from HuggingFace Hub: allenai/c4
```

### Example 3: Custom Dataset Path
```bash
DATASET_NAME=c4 \
DATASET_PATH=allenai/c4 \
./xpu_torchtitan/run_train_torchtitan.sh multi ./hostfile.txt \
  -- --training.steps 100
# Uses DATASET_NAME=c4
# Overrides with custom path: allenai/c4
```

### Example 4: Local Path (Offline Mode)
```bash
DATASET_NAME=c4_test \
DATASET_PATH=/path/to/local/c4_test \
./xpu_torchtitan/run_train_torchtitan.sh multi ./hostfile.txt \
  -- --training.steps 100
# Uses DATASET_NAME=c4_test
# Loads from local filesystem: /path/to/local/c4_test
```

### Example 5: With local_files_only=True for Tokenizer
```bash
DATASET_NAME=c4 \
DATASET_PATH=/path/to/local/c4 \
./xpu_torchtitan/run_train_torchtitan.sh multi ./hostfile.txt \
  -- --training.steps 100
# Tokenizer loads with local_files_only=True (handled internally in trainer.py)
# Dataset loads from DATASET_PATH (independent of tokenizer setting)
```

---

## Available Datasets

### Built-in Datasets

| Name | Description | Source |
|------|-------------|--------|
| `pg19_multinews` | PG19 + MultiNews interleaved, streaming | HuggingFace Hub |
| `c4` | C4 (Common Crawl) dataset | HuggingFace Hub (allenai/c4) |
| `c4_test` | C4 offline bundled smoke sample | Local torchtitan bundled data |
| Custom | Your own dataset | Local filesystem or HF Hub |

### Example: Using C4 Test Dataset
```bash
DATASET_NAME=c4_test \
./xpu_torchtitan/run_train_torchtitan.sh multi ./hostfile.txt \
  -- --training.steps 100
# Uses bundled c4_test data (good for smoke testing)
```

---

## How It Works (Command Building)

```bash
# 1. Set DATASET_NAME (required)
DATASET_NAME="${DATASET_NAME:-pg19_multinews}"

# 2. Set DATASET_PATH (optional)
DATASET_PATH="${DATASET_PATH:-}"

# 3. Always pass DATASET_NAME to dataloader
TRAIN_CMD+=("--dataloader.dataset" "$DATASET_NAME")

# 4. If DATASET_PATH is provided, also pass it
if [[ -n "$DATASET_PATH" ]]; then
  TRAIN_CMD+=("--dataloader.dataset_path" "$DATASET_PATH")
fi

# Final command arguments passed to torchtitan:
# --dataloader.dataset pg19_multinews
# --dataloader.dataset_path /path/to/dataset  (if provided)
```

---

## local_files_only=True Details

**Only affects tokenizer loading**:
- Rank 0: `local_files_only=False` → Can download tokenizer from HuggingFace
- Ranks 1-N: `local_files_only=True` → Load tokenizer from local cache only

**Does NOT affect dataset loading**:
- Dataset loading uses HuggingFace datasets library
- Honors DATASET_PATH environment variable
- Can stream from HuggingFace or load from local path

---

## Full Example: Complete Command

```bash
# Multi-node training with custom dataset and local tokenizer caching
NNODES=9 \
DATASET_NAME=c4 \
DATASET_PATH=allenai/c4 \
MODEL=/lus/flare/projects/datascience/seonghapark/agpt-2b-v2-256n-step-92859-safetensors \
MODULE=agpt \
CONFIG=agpt_2b \
./xpu_torchtitan/run_train_torchtitan.sh multi ./hostfile.txt \
  --resource-monitor \
  --resource-interval 15 \
  -- \
  --training.steps 400000 \
  --training.seq_len 16384
```

**What happens**:
1. ✅ Tokenizer: Rank 0 downloads, ranks 1-N load with `local_files_only=True`
2. ✅ Dataset: All ranks load C4 from allenai/c4 (via DATASET_PATH)
3. ✅ Model: All ranks load AGPT-2B from provided path

---

## Log Output

When you run with DATASET_NAME and DATASET_PATH set, you'll see:

```
[ARGS] DATASET_NAME      = c4
[ARGS] DATASET_PATH      = allenai/c4
```

And in the command:
```
--dataloader.dataset c4 --dataloader.dataset_path allenai/c4
```

---

## Key Points

1. **`local_files_only=True` = Tokenizer only**
   - Does not affect datasets
   - Ensures rank 0 downloads first, then others use cache

2. **`DATASET_NAME` = Primary identifier**
   - Which dataset to use (c4, pg19_multinews, etc.)
   - Always passed to dataloader

3. **`DATASET_PATH` = Optional override**
   - Custom path on filesystem
   - HuggingFace Hub identifier (allenai/c4)
   - Only included if explicitly set

4. **Independent concerns**
   - Tokenizer caching: handled by trainer.py
   - Dataset loading: handled by dataloader config
   - No conflict between them

---

## Troubleshooting

### Issue: Dataset not found
```bash
# Error: Can't find c4 dataset
# Solution: Ensure DATASET_PATH is correct
DATASET_NAME=c4 \
DATASET_PATH=allenai/c4 \
./xpu_torchtitan/run_train_torchtitan.sh multi ...
```

### Issue: Tokenizer fails with local_files_only=True
```bash
# This is a tokenizer issue, not dataset issue
# Solution: Ensure rank 0 downloads complete first
# Check: Tokenizer files exist in HF_ASSETS_PATH
ls -la /lus/flare/projects/datascience/seonghapark/agpt-2b-v2-256n-step-92859-safetensors/
```

### Issue: Network timeout on dataset
```bash
# If dataset streaming from HuggingFace times out:
# Solution: Use local dataset path
DATASET_NAME=c4 \
DATASET_PATH=/local/path/to/c4 \
./xpu_torchtitan/run_train_torchtitan.sh multi ...
```

---

## Summary Table

| Aspect | local_files_only=True | Dataset Config | Notes |
|--------|----------------------|-----------------|-------|
| **Applies to** | Tokenizer only | Independent | Separate concerns |
| **Rank 0** | `local_files_only=False` | Loads from DATASET_PATH | Can download tokenizer |
| **Ranks 1-N** | `local_files_only=True` | Loads from DATASET_PATH | Cache-only tokenizer |
| **Control via** | trainer.py (automatic) | DATASET_NAME/DATASET_PATH | User-provided |
| **Default** | Enabled for multi-node | pg19_multinews | Built into script |

---

**Bottom Line**: Set `DATASET_NAME` and `DATASET_PATH` environment variables independently of `local_files_only=True`. They don't conflict — one handles tokenizer caching, the other handles dataset loading.
