# YaRN Quick Start Guide for agpt-20b-v2-512n-step-4000

## What is YaRN?

YaRN (Yet Another RoPE Extension) extends the context length of language models using Rotary Position Embeddings. It works by:
- **Extrapolating low frequencies** (smooth continuation beyond training length)
- **Interpolating high frequencies** (compression within the extension factor)
- **Applying attention scaling** (mscale correction for stability)

Result: Train at 8K, extend to 16K/32K/64K+ with minimal fine-tuning (~400 steps).

## Quick Facts

| Property | Value |
|----------|-------|
| Model | agpt-20b-v2-512n-step-4000 |
| Original training length | 8K tokens |
| Extended with YaRN | 16K+ tokens |
| Weight conversion needed | **No** (pure config change) |
| Fine-tune steps required | ~400 (per YaRN paper) |
| Config file | `config.json` (already updated) |

## 1. Verify Installation

```bash
cd /lus/flare/projects/datascience/seonghapark
python3 yarn_validation_agpt20b.py
```

Expected output:
```
✓ All validations passed. Ready for continued pretraining with YaRN.
```

## 2. Training Recipe

### Option A: Using the launch script (recommended)

```bash
cd xpu_launcher/xpu_torchtitan
./train_agpt20b_yarn.sh 16384 400 8
```

Parameters:
- `16384` — Target sequence length (16K)
- `400` — Number of training steps
- `8` — Number of nodes

### Option B: Manual launch with environment variables

```bash
cd xpu_launcher/xpu_torchtitan

export CKPT=/path/to/step-4000-dcp-checkpoint
export MODEL_PATH=/lus/flare/projects/datascience/seonghapark/agpt-20b-v2-512n-step-4000-safetensors
export MODULE=agpt
export CONFIG=agpt_20b_yarn
export SEQ_LEN=16384
export TRAINING_STEPS=400
export DATASET_NAME=pg19_multinews
export TP=8  # Tensor parallelism

./run_train_torchtitan.sh multi
```

## 3. Monitor Training

During training, watch for:
- ✓ Loss should be stable (not spiking)
- ✓ Learning curves should be smooth
- ✓ Effective batch sizes and throughput stable

## 4. Config File Explained

The `config.json` now includes:

```json
"rope_scaling": {
  "type": "yarn",
  "factor": 2.0,
  "original_max_position_embeddings": 8192,
  "beta_fast": 1.0,
  "beta_slow": 32.0
}
```

| Parameter | Meaning |
|-----------|---------|
| `type` | Scaling method (yarn) |
| `factor` | 2.0x extension (8K → 16K) |
| `original_max_position_embeddings` | Length model was trained at |
| `beta_fast` | High-frequency scaling (1.0 = no scaling) |
| `beta_slow` | Low-frequency scaling (32.0 = extrapolate) |

## 5. Next Steps After Training

### Generate evaluation checkpoint (HF format)
```bash
python3 llm_evaluation/convert_dcp_to_safetensors.py \
  --input-path /path/to/step-400-yarn-checkpoint \
  --output-path agpt-20b-v2-512n-step-4000-yarn-16k-safetensors
```

### Evaluate with extended context
```bash
export MODEL_PATH=agpt-20b-v2-512n-step-4000-yarn-16k-safetensors
export ROPE_SCALING=yarn
./run_lm-evaluation-harness_vllm_longcontext_MultipleNodes.sh
```

## 6. Troubleshooting

| Issue | Solution |
|-------|----------|
| Loss spikes immediately | Check `original_max_position_embeddings` matches checkpoint training length |
| CUDA OOM | Reduce `SEQ_LEN` or increase `TP` (tensor parallelism) |
| Training diverges | Start with smaller factor (1.5x) or fewer steps |
| Config validation fails | Run `python3 yarn_validation_agpt20b.py` to diagnose |

## 7. Files Provided

1. **config.json** — Updated with YaRN rope_scaling
2. **YaRN_agpt20b_implementation.md** — Detailed implementation guide
3. **yarn_validation_agpt20b.py** — Validation and metrics script
4. **train_agpt20b_yarn.sh** — Training launch script template
5. **YaRN_QUICK_START.md** — This file

## 8. Key Resources

- **YaRN Paper**: [arxiv.org/abs/2309.00071](https://arxiv.org/abs/2309.00071)
- **Reference Repo**: [github.com/jquesnelle/yarn](https://github.com/jquesnelle/yarn)
- **TorchTitan RoPE**: `torchtitan/models/common/rope.py`
- **Existing Usage**: gpt_oss, deepseek_v3 configs

## 9. Example Training Run

```bash
# Single-node test (5 steps to verify setup)
SEQ_LEN=16384 TRAINING_STEPS=5 ./run_train_torchtitan.sh single

# Full training (400 steps on 8 nodes)
SEQ_LEN=16384 TRAINING_STEPS=400 ./run_train_torchtitan.sh multi
```

## 10. Expected Timeline

| Phase | Duration | Action |
|-------|----------|--------|
| Setup | 5-10 min | Verify config, run validation |
| Single-node test | 10-15 min | 5-step local run |
| Full training | 2-4 hours | 400 steps on 8x nodes |
| Conversion | 30-60 min | Convert to HF safetensors |
| Evaluation | 1-2 hours | Run evals on extended context |

## Questions?

Refer to:
1. `YaRN_agpt20b_implementation.md` for detailed technical info
2. `yarn_validation_agpt20b.py` output for config diagnostics
3. TorchTitan docs at `xpu_launcher/xpu_torchtitan/torchtitan_repo`

---

**Ready to train?** Run: `python3 yarn_validation_agpt20b.py` then `./train_agpt20b_yarn.sh`
