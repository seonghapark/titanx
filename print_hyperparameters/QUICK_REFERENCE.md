# Quick Reference: Hyperparameter Logging

## What Changed?

Two key areas of the TorchTitan training pipeline now print comprehensive hyperparameter information:

1. **Launch Script** (`run_train_torchtitan.sh`) - Prints at startup
2. **Trainer** (`trainer.py`) - Prints when training begins

## When to Look for These Logs

### At Launch Time
```
$ MODEL=/models/llama ./run_train_torchtitan.sh single

📍 Look for: "TRAINING LAUNCH CONFIGURATION" section
```

### At Training Start
```
...
Training starts at step 1
📍 Look for: "TRAINING HYPERPARAMETERS AND CONFIGURATION" section
```

## Key Parameters You'll Find

### **Learning Rate**
```
[TRAINING]
  Learning Rate: 3e-05
```

### **Tensor Parallel Size**
```
[PARALLELISM]
  Tensor Parallel Size (TP): 4
```

### **Architecture Name (with YaRN detection)**
```
[MODEL]
  Architecture: Llama
  Rope Scaling Type: yarn        <- YaRN detected!
  Rope Theta: 10000000.0
```

### **Batch Sizes**
```
[TRAINING]
  Global Batch Size: 128
  Micro Batch Size: 4
  Gradient Accumulation Steps: 32
```

### **All Parallelism Dimensions**
```
[PARALLELISM]
  Tensor Parallel Size (TP): 4
  Pipeline Parallel Size (PP): 1
  Data Parallel Size (DP): 8
  Context Parallel Size (CP): 1
```

### **Model Architecture**
```
[ARCHITECTURE DETAILS]
  Hidden Dimension: 4096
  Number of Layers: 32
  Number of Attention Heads: 32
  Number of KV Heads: 8
  Vocabulary Size: 128256
  Intermediate Dimension (FFN): 14336
```

### **Optimizer Settings**
```
[OPTIMIZER]
  Optimizer Type: AdamW
  Learning Rate: 3e-05
  Weight Decay: 0.1
  Betas (Adam): (0.9, 0.95)
  Epsilon: 1e-08
```

## Common Search Patterns in Logs

### Finding Learning Rate
```bash
$ grep "Learning Rate:" training.log
Learning Rate: 3e-05
```

### Finding Parallelism Configuration
```bash
$ grep -A 6 "\[PARALLELISM\]" training.log
Tensor Parallel Size (TP): 4
Pipeline Parallel Size (PP): 1
Data Parallel Size (DP): 8
```

### Checking if YaRN is Enabled
```bash
$ grep "Rope Scaling" training.log
Rope Scaling Type: yarn
```

### Finding Batch Sizes
```bash
$ grep -E "Global Batch|Micro Batch|Gradient Accumulation" training.log
Global Batch Size: 128
Micro Batch Size: 4
Gradient Accumulation Steps: 32
```

## Use Cases

### ✅ Verify Configuration Applied Correctly
Run training with parameters, then check logs to confirm they were used.

### ✅ Quick Debugging
Instead of reading config files, look at the consolidated logs.

### ✅ Experiment Tracking
Save training logs for complete record of what was run.

### ✅ Batch Size Calculations
See both global and micro batch size to understand effective batch size:
```
Effective Batch Size = Global Batch / Data Parallel Size
                     = 128 / 8 = 16 per GPU
```

### ✅ Memory Estimation
With architecture details visible:
- Layer count
- Hidden dimension
- Number of heads
- You can better estimate memory usage

## Files Modified

| File | Lines | What |
|------|-------|------|
| `run_train_torchtitan.sh` | 258-297 | Launch config formatting |
| `trainer.py` | 787-886 | New logging method |
| `trainer.py` | 1028 | Call logging method |

## Example: Running and Finding Parameters

```bash
# Step 1: Run training
$ MODEL=/models/llama3-8b \
  TRAINING_STEPS=1000 \
  SEQ_LEN=8192 \
  ./run_train_torchtitan.sh single

# Step 2: Check launch config (visible immediately)
[TRAINING HYPERPARAMETERS]
  Training Steps         = 1000
  Sequence Length        = 8192

# Step 3: Once training starts, check full config
[TRAINING]
  Learning Rate: 3e-05
  Sequence Length: 8192
  Global Batch Size: 256

[PARALLELISM]
  Tensor Parallel Size (TP): 2
  Pipeline Parallel Size (PP): 1
  Data Parallel Size (DP): 16
```

## Troubleshooting

### "I don't see hyperparameter logs"
- Make sure training is actually running
- Logs appear after "Training starts at step 1" message
- Check both stdout and stderr (script outputs to stderr)

### "Learning Rate shows optimizer.lr instead of value"
- This shouldn't happen - if you see a literal like this, check for exceptions during logging
- The method uses `config.optimizer.lr` which should always be a number

### "Some architecture details are missing"
- This is normal! Not all models have all attributes
- The code safely skips missing attributes with `hasattr()` checks

## Key Takeaways

1. **Two sets of logs:** Launch-time config + Training-time comprehensive config
2. **Everything visible:** All requested parameters are logged
3. **YaRN detection:** Rope scaling modifications are visible
4. **Safe implementation:** Uses defensive programming (hasattr checks)
5. **No performance impact:** Logging only happens once at startup
6. **Easy to find:** Clear sections and formatting for quick scanning

---

**Need more details?** See:
- `HYPERPARAMETER_LOGGING_SUMMARY.md` - Technical details
- `BEFORE_AFTER_COMPARISON.md` - What changed and why
- `EXAMPLE_HYPERPARAMETER_OUTPUT.txt` - Full example output
