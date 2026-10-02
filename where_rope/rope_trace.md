# RoPE Call Trace in TorchTitan Training

## Overview
When running `./run_train_torchtitan.sh`, RoPE (Rotary Position Embedding) is called during the forward pass of each attention layer for every training step and batch.

## Call Stack

### 1. **Training Entry Point**
```
run_train_torchtitan.sh
├─ Parses configuration and environment variables
└─ Calls: python titan_train.py [args]
```
📍 **File:** `run_train_torchtitan.sh` (lines 191-226)

### 2. **TorchTitan Main Entry**
```
titan_train.py::main()
├─ Normalizes distributed environment (MPI/PBS setup)
├─ Installs XCCL workaround if needed
└─ Calls: torchtitan.train.main()
```
📍 **File:** `titan_train.py` (lines 111-119)

### 3. **Training Loop**
```
torchtitan/train.py::main()
├─ Initializes model, optimizer, distributed setup
├─ Enters training loop
└─ For each training step:
   └─ Calls: trainer.train_step()
```
📍 **File:** `torchtitan/trainer.py::train()` (line 1027)

### 4. **Training Step**
```
trainer.py::train_step()
├─ Loads batch from dataloader
├─ For each microbatch:
│  └─ Calls: forward_backward_step()
└─ Synchronizes loss and gradients
```
📍 **File:** `torchtitan/trainer.py::train_step()` (line 904-960)

### 5. **Forward-Backward Step**
```
trainer.py::forward_backward_step()
├─ Calls: model.forward(batch_data, positions, attention_masks)
├─ Computes loss
└─ loss.backward()
```
📍 **File:** `torchtitan/trainer.py::forward_backward_step()` (line 727-783)

### 6. **Model Forward Pass**
```
Llama3Model::forward()
  (inherits from Decoder)
├─ Iterates through each transformer layer
└─ For each layer:
   └─ Calls: Llama3TransformerBlock.forward()
```
📍 **File:** `torchtitan/models/llama3/model.py::Llama3TransformerBlock.forward()` (line 42-50)

### 7. **Transformer Block Forward**
```
Llama3TransformerBlock.forward()
├─ Applies attention_norm
├─ Calls: self.attention(normalized_x, attention_masks, positions)
└─ Applies FFN
```
📍 **File:** `torchtitan/models/llama3/model.py` (line 48)

### 8. **Attention Forward ⭐ RoPE Called Here**
```
Attention.forward()
├─ Projects input to Q, K, V: xq_BLNH, xk_BLNH, xv_BLNH
├─ Applies optional QK normalization
├─ **ROPE CALL:**
│  └─ xq_BLNH, xk_BLNH = self.rope(xq_BLNH, xk_BLNH, positions)
├─ Applies inner attention (SDPA/FlexAttention)
└─ Applies output projection
```
📍 **File:** `torchtitan/models/common/attention.py` (lines 910-936)
📍 **RoPE Call:** Line 924

### 9. **RoPE Forward Pass**
```
RoPE.forward()
├─ Calls: reshaped_cache = self._reshape_cache(query, positions)
│  ├─ Loads precomputed RoPE cache
│  ├─ Reshapes/adjusts for current batch and position indices
│  └─ Returns aligned cache
├─ Calls: RoPE.apply_rotary_emb(query, key, reshaped_cache)
│  ├─ Applies rotation to Q and K tensors
│  └─ Returns rotated Q and K
└─ Returns: (rotated_query, rotated_key)
```
📍 **File:** `torchtitan/models/common/rope.py` (lines 117-125)

## When RoPE is Called

**Timing:** During the **forward pass** of every training step

### Detailed Timing:
1. **Per Training Step:** Once `trainer.train_step()` is invoked
2. **Per Microbatch:** In `forward_backward_step()` 
3. **Per Transformer Layer:** For each of the N layers in the model
4. **Per Attention Head:** Applied to Q and K projections (but called once per layer)

### Example for llama3_8b model:
- **Batch size:** 16
- **Sequence length:** 16,384
- **Number of layers:** 32
- **Number of heads:** 32
- **Head dimension:** 128

**RoPE is called:** `32 layers × 1 time per layer per batch = 32 times per forward pass`

If running with **100 training steps**:
- **Total RoPE calls:** 100 steps × 32 calls = **3,200 RoPE forward passes**

## Configuration & Cache

### RoPE Cache
- **Precomputed** at initialization: `RoPE._precompute_cache()` (line 73)
- **Stored** as a buffer (non-persistent): Line 71
- **Updated** when model's `max_seq_len` changes (adaptive sequence length)

### Scaling Methods
Supported via the `RoPE.Config` (lines 50-66):
- `scaling="none"` — Standard RoPE (no scaling)
- `scaling="llama"` — Llama 3 scaling with configurable factors
- `scaling="yarn"` — YaRN (Yet another RoPE extensioN) scaling

### Default Parameters
```python
Config defaults:
├─ dim: (from model config, e.g., 4096 for Llama 3 8B)
├─ max_seq_len: (from model, e.g., 8192 or adaptive to training.seq_len)
├─ theta: 10000.0
├─ scaling: "none"
├─ scaling_factor: 8.0 (for llama scaling)
└─ ...
```

## Variants

### ComplexRoPE
- Uses complex exponential caches
- Cache shape: `(max_seq_len, dim / 2)`
- More numerically stable

### CosSinRoPE  
- Uses concatenated cos/sin caches
- Cache shape: `(max_seq_len, dim)`
- Memory efficient

## Performance Considerations

1. **Cache Precomputation:** O(max_seq_len × dim) at model init
2. **Per-Step Cost:** O(batch_size × seq_len × num_heads × head_dim)
3. **CUDA Kernels:** Uses optimized fused kernels (ComplexRoPE) or element-wise ops (CosSinRoPE)
4. **Sharding:** RoPE respects the attention layer's sharding config for distributed training

## Files Involved

| File | Purpose |
|------|---------|
| `run_train_torchtitan.sh` | Training launch script |
| `titan_train.py` | MPI/PBS initialization → TorchTitan entry |
| `torchtitan/train.py` | Main training loop |
| `torchtitan/trainer.py` | `train_step()` and `forward_backward_step()` |
| `torchtitan/models/llama3/model.py` | Llama3Model and TransformerBlock |
| `torchtitan/models/common/attention.py` | Attention layer (where RoPE is **called**) |
| `torchtitan/models/common/rope.py` | RoPE implementation (**where it runs**) |
