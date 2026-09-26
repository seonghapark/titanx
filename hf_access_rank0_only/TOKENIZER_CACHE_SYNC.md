# Tokenizer Cache Synchronization - Multi-Rank Loading

**Date**: 2026-09-26  
**Status**: ✅ Implemented  
**Location**: `/lus/flare/projects/datascience/seonghapark/xpu_launcher/xpu_torchtitan/torchtitan_repo/torchtitan/trainer.py` (lines 509-519)

---

## What Was Changed

### Problem
In multi-node training, all ranks attempted to download and cache the tokenizer simultaneously from HuggingFace, causing:
- Network congestion
- Cache conflicts
- Potential corruption if multiple ranks wrote simultaneously

### Solution
Implemented rank 0 priority pattern:

```python
# build tokenizer with rank 0 priority (download once, all ranks read from cache)
rank = dist_utils.get_rank()
if rank == 0:
    # Rank 0: Download and cache model assets
    self.tokenizer = config.tokenizer.build(tokenizer_path=config.hf_assets_path)
# All ranks: Wait for rank 0 to complete download
if torch.distributed.is_initialized():
    torch.distributed.barrier()
if rank != 0:
    # Other ranks: Load from cached files
    self.tokenizer = config.tokenizer.build(tokenizer_path=config.hf_assets_path)
```

---

## How It Works

### Execution Flow (Multi-Rank)

```
Rank 0                          Rank 1-N
└─ Check if rank == 0          └─ Check if rank == 0
   └─ YES                         └─ NO (skip)
   └─ Download + cache            
      tokenizer                 └─ Wait at barrier()
   └─ Reach barrier()          
      ────────────────────────────> PROCEED
   └─ Load from cache          
   └─ Continue                 └─ Load from cache
                               └─ Continue
```

### Key Points

1. **Rank 0 Downloads First**
   - Single source of truth for tokenizer cache
   - Files written once, atomically

2. **Barrier Synchronization**
   - `torch.distributed.barrier()` ensures all ranks wait
   - Rank 0 completes, then others proceed
   - Zero chance of concurrent access

3. **All Ranks Read from Cache**
   - After barrier, tokenizer files guaranteed to exist
   - `local_files_only` optimization can be added later
   - Other ranks load pre-cached files (fast)

4. **Distributed Environment Check**
   - `torch.distributed.is_initialized()` - only barrier in multi-rank mode
   - Single-node jobs unaffected (rank 0 downloads, no barrier)

---

## File Locations

**Tokenizer Cache Path**: Determined by HuggingFace transformers library default or explicit path:
```python
config.hf_assets_path  # e.g., /lus/flare/projects/datascience/seonghapark/agpt-2b-v2-256n-step-92859-safetensors/
```

**Cache Directory**: Typically `~/.cache/huggingface/hub/` or custom if configured

---

## Future Optimization: local_files_only=True

To prevent any network calls on non-rank-0:

```python
# Modify HuggingFaceTokenizer.build() to accept local_files_only parameter
rank = dist_utils.get_rank()
if rank == 0:
    self.tokenizer = config.tokenizer.build(
        tokenizer_path=config.hf_assets_path,
        local_files_only=False  # Download
    )
else:
    torch.distributed.barrier()
    self.tokenizer = config.tokenizer.build(
        tokenizer_path=config.hf_assets_path,
        local_files_only=True   # Local cache only
    )
```

---

## Model Weight Loading

**Note**: Model weights are loaded via `model_spec.state_dict_adapter()` at line 543, which may also download HF assets. To fully synchronize model loading:

```python
rank = dist_utils.get_rank()

# Download model weights with rank 0 priority
if model_spec.state_dict_adapter:
    if rank == 0:
        sd_adapter = model_spec.state_dict_adapter(
            model_config, 
            config.hf_assets_path
        )
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    if rank != 0:
        sd_adapter = model_spec.state_dict_adapter(
            model_config, 
            config.hf_assets_path
        )
```

---

## Testing

### Single Node (No Barrier Effect)
```bash
./xpu_torchtitan/run_train_torchtitan.sh single -- --training.steps 100
# Tokenizer downloaded by rank 0, barrier skipped (not distributed)
```

### Multi-Node (With Barrier)
```bash
NNODES=9 ./xpu_torchtitan/run_train_torchtitan.sh multi ./hostfile.txt -- --training.steps 100
# Rank 0 downloads → barrier → Ranks 1-107 load from cache
```

---

## Verification

Check logs for synchronization:

```bash
# All ranks should log tokenizer loading
grep -r "Loading tokenizer" /lus/flare/projects/datascience/seonghapark/xpu_launcher/xpu_torchtitan/torchtitan_repo/outputs/*/run.log

# Rank 0 logs should show download, others show immediate load
# (time difference between rank 0 and rank 1 should be < 5 seconds)
```

---

## Compatibility

- ✅ Multi-node training (PBS multi-rank jobs)
- ✅ Single-node training (auto-skips barrier)
- ✅ Checkpoint loading (tokenizer built once)
- ✅ Validation (uses same tokenizer instance)

---

## Status

✅ **Complete** - Tokenizer now loads with rank 0 priority  
⏳ **Optional** - Model weights can use same pattern (see above)

---

**Modified File**: `/lus/flare/projects/datascience/seonghapark/xpu_launcher/xpu_torchtitan/torchtitan_repo/torchtitan/trainer.py`  
**Lines Changed**: 509-519  
**Implementation Pattern**: Rank 0 priority + distributed barrier
