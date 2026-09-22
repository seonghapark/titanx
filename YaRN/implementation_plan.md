# Implement YaRN for training agpt models

## Context

`agpt` (AuroraGPT) is trained via a vendored TorchTitan fork
(`xpu_launcher/xpu_torchtitan/torchtitan_repo/torchtitan/models/agpt/`).
The user wants to implement YaRN (Yet another RoPE extensioN,
arxiv.org/abs/2309.00071, github.com/jquesnelle/yarn) to extend the
context length of agpt models via continued training/fine-tuning.

**Key finding: YaRN math is already implemented in this codebase** in
the shared RoPE module (`torchtitan/models/common/rope.py`), used
today by `gpt_oss` and `deepseek_v3` model configs. It is *not* wired
up anywhere for `agpt`. So this is primarily a configuration task
(new model flavors + a training recipe), not new kernel/math code —
matching the DeepSeek-V3-style NTK-by-parts YaRN from the paper
(dynamic interpolation ramp between `beta_fast`/`beta_slow`, exactly
`find_correction_dim`/`find_correction_range`/`linear_ramp_factor`
from the reference `github.com/jquesnelle/yarn` repo), plus the
`mscale` attention-temperature correction from the YaRN paper (§3.4).

Current agpt checkpoints (`agpt-20b-v2-512n-step-4000-safetensors`,
`agpt-2b-v2-256n-step-92859-safetensors`) are configured at
`max_position_embeddings=131072` with plain (unscaled) RoPE
(`rope_theta=500000`/`50000`) and were trained at `seq_len=8192`
(see production docs) — i.e. **the RoPE cache already reaches 128k,
but the model was never actually trained on sequences anywhere near
that length.** This plan treats "128k" as the *pretend* original
context (`original_seq_len`) that YaRN will further extend from, via
continued pretraining, matching the YaRN paper's proven recipe: start
from a checkpoint trained without scaling, apply YaRN, and fine-tune
briefly (~ 400 steps in the paper) at longer sequence length.

## Approach

Two parts: (1) expose YaRN as a first-class, named option in the agpt
model registry (mirroring the existing `gpt_oss`/`deepseek_v3`
pattern and the existing `agpt_2b_real`/`agpt_20b_real`
backend-switching helper in `config_registry.py`), and (2) a training
recipe/launch config for the actual context-extension fine-tune run.

### 1. `torchtitan/models/agpt/__init__.py` — add YaRN-scaled flavors

Add a `scaling`/`rope_factor`/`original_seq_len`/`beta_fast`/
`beta_slow` passthrough is *already present* on `_build_agpt_config`
(`scaling: Literal["none", "llama", "yarn"] = "none"`, `max_seq_len`)
— but `rope_factor`, `beta_fast`, `beta_slow`, `original_seq_len` are
**not** currently forwarded from `_build_agpt_config` into
`rope_cls(...)` (only `dim`, `max_seq_len`, `theta`, `scaling` are
passed — see lines 357-362). Extend `_build_agpt_config`'s signature
and body to accept and forward these four YaRN params (defaulting to
the `RoPE.Config` dataclass defaults so existing callers are
unaffected).

Then add new flavors alongside the existing `"2B"`/`"20B"` entries,
following the `gpt_oss` pattern exactly:

```python
"2B_yarn": _build_agpt_config(
    dim=2048, n_layers=12, n_heads=16, n_kv_heads=4,
    rope_theta=50000, vocab_size=256128, hidden_dim=11008,
    scaling="yarn",
    max_seq_len=262144,          # 2x extension target


'''
max_seq_len=262144 is the RoPE cache ceiling (pre-computed rotation matrices), not the actual training sequence length. The "2x extension target" comment is wrong/incomplete.

The actual extension happens at the training level:
- Original training: 8K (original_seq_len)
- Current run: 16K (what you just ran, apt_2b flavor)
- Next target: 32K (with agpt_2b_yarn, which would be 4x the original)
- Max possible: up to 262K (limited only by max_seq_len)

The 262144 is a generous upper bound borrowed from the gpt_oss pattern in the codebase -- it lets you extend arbitrarily later without recompiling the config. You control the actual training length separately with --training.seq_len.

So the comment should have been something like:
max_seq_len=262144,  # Upper bound for RoPE cache; actual training seq_len set via --training.seq_len

The "2x extension target" is where you'll train next (32K), but max_seq_len is an upper ceiling that gives you headroom. Does that clarify the logic?
'''


    original_seq_len=8192,       # the length it was actually trained at
    rope_factor=32.0,            # matches gpt_oss's factor for a similar ratio; tune per target length
    beta_fast=32.0, beta_slow=1.0,  # YaRN paper / jquesnelle defaults (beta_fast=32, beta_slow=1)
),
"20B_yarn": _build_agpt_config(
    dim=5120, n_layers=64, n_heads=40, n_kv_heads=8,
    rope_theta=500000, vocab_size=256128,
    hidden_dim=compute_ffn_hidden_dim(5120, multiple_of=1024),
    scaling="yarn",
    max_seq_len=262144,
    original_seq_len=8192,
    rope_factor=32.0,
    beta_fast=32.0, beta_slow=1.0,
),
```

Add matching lowercase aliases (`agpt_configs["2b_yarn"] = ...`) next
to the existing alias block.

Note on `rope_backend`: default `_build_agpt_config` uses
`ComplexRoPE` (`rope_backend="complex"`). `ComplexRoPE`'s YaRN branch
(rope.py lines 176-218) implements the NTK-by-parts interpolation but
**not** the `mscale` attention-temperature correction — only
`CosSinRoPE`'s YaRN branch (lines 279-315) applies `mscale = 0.1 *
log(rope_factor) + 1.0` to the cos/sin cache directly. Since agpt's
`Attention` (via `common/attention.py` `GQAttention`) does not read
`config.rope.mscale` itself (unlike `deepseek_v3/model.py:86` which
manually multiplies `softmax_scale` by `mscale²`), **use
`rope_backend="cos_sin"` for the new YaRN flavors** so the attention
temperature correction from the YaRN paper is actually applied,
matching the reference implementation's recommended `attn_factor`
scaling. Pass `rope_backend="cos_sin"` in the two new flavor calls
above.

### 2. `torchtitan/models/agpt/config_registry.py` — training recipe

Add `Trainer.Config` factories for the continued-pretraining run,
following the existing `agpt_2b`/`agpt_20b` pattern (which calls the
shared `agpt(flavor, ...)` helper):

```python
def agpt_2b_yarn(seq_len: int = 32768) -> Trainer.Config:
    """Continue-pretrain agpt-2b with YaRN-scaled RoPE for context extension.

    Load the existing agpt-2b checkpoint's weights only
    (--checkpoint.initial_load_path=<DCP step-92859 dir>,
    initial_load_model_only=True is already the CheckpointManager
    default) and fine-tune at a longer seq_len than the original
    8192 pretraining length. Per the YaRN paper, only a small number
    of additional steps (~400) at the target length are needed.
    """
    return agpt("2b_yarn", seq_len=seq_len, activation_checkpoint_mode="none")


def agpt_20b_yarn(seq_len: int = 32768) -> Trainer.Config:
    return agpt("20b_yarn", seq_len=seq_len)
```

`agpt(...)`'s existing seq_len/dtype/compile/checkpoint plumbing is
reused as-is — no changes needed there. The `Decoder.Config.
update_from_config` check in `common/decoder.py` (raises if
`training.seq_len > rope.max_seq_len`) will pass as long as
`seq_len` stays ≤ the new flavor's `max_seq_len=262144`.

### 3. Launch / run instructions (docs only, no new script needed)

`run_train_torchtitan.sh` already supports everything needed via env
vars — no launcher changes required. Document the invocation (e.g. in
a short note, not a new file unless the user wants one):

```bash
CKPT=/lus/flare/projects/AuroraGPT/foremans/runs/agpt-2b-v2/torchtitan-ezpz/outputs/checkpoints/agpt-2b-sophiag-olmo-mix-1124-n256-gbs6144/step-92859 \
MODEL_PATH=/lus/flare/projects/datascience/seonghapark/agpt-2b-v2-256n-step-92859-safetensors \
MODULE=agpt \
CONFIG=agpt_2b_yarn \
SEQ_LEN=32768 \
DATASET_NAME=pg19_multinews \
TRAINING_STEPS=400 \
./run_train_torchtitan.sh multi
```

`CKPT` provides the DCP weights (loaded model-only by default per
`CheckpointManager.Config.initial_load_model_only=True`); `CONFIG=
agpt_2b_yarn` selects the YaRN-scaled architecture the weights get
loaded into. Since YaRN's `mscale`/interpolation cache is purely a
function of config (not learned state), no weight conversion or
special checkpoint surgery is needed — this mirrors exactly how
`agpt_2b_real`/`agpt_20b_real` already reload the same checkpoint
into a differently-configured RoPE backend.

### 4. HF `config.json` for eval/serving after fine-tuning (optional, follow-up)

Not part of training, but flagged for completeness: once a
YaRN-extended checkpoint is converted to HF safetensors (via
`llm_evaluation/convert_dcp_to_safetensors.py`), its `config.json`
should carry a `rope_scaling` block so vLLM/HF serving applies the
same scaling — following the existing pattern already used for Gemma
in `assets/hf/gemma-7b-rope131072/config.json` (which the
`run_lm-evaluation-harness_vllm_longcontext_MultipleNodes.sh`
`ROPE_SCALING=yarn` path already knows how to generate/consume). This
plan does not implement that conversion step — call it out to the
user as a follow-up once training configs are approved, since it's
inference-side and out of scope for "implement YaRN for training."

## Files to modify

- `xpu_launcher/xpu_torchtitan/torchtitan_repo/torchtitan/models/agpt/__init__.py`
  — extend `_build_agpt_config` to forward YaRN params; add
  `"2B_yarn"` / `"20B_yarn"` flavors + lowercase aliases.
- `xpu_launcher/xpu_torchtitan/torchtitan_repo/torchtitan/models/agpt/config_registry.py`
  — add `agpt_2b_yarn()` / `agpt_20b_yarn()` factories.

No changes needed to `rope.py` (YaRN math already correct and
matches the paper/reference repo), `decoder.py` (seq_len sync
already generic), `attention.py`, `state_dict_adapter.py`, or any
launch scripts.

## Verification

1. **Unit-level sanity check (no cluster needed):** run a small
   Python snippet that imports `torchtitan.models.agpt.agpt_configs`,
   builds `agpt_configs["2B_yarn"]`, and confirms:
   - `layers[0].attention.rope.scaling == "yarn"`
   - `isinstance(layers[0].attention.rope, CosSinRoPE.Config)`
   - Instantiate `CosSinRoPE(config)` directly and check
     `cache.shape[0] == max_seq_len` and that it does not raise the
     `0 < low < high < d_half - 1` assertion (this is the main
     failure mode if `beta_fast`/`beta_slow`/`original_seq_len` are
     mismatched with `dim`/`theta`).
   - Compare `ComplexRoPE` vs `CosSinRoPE` cache at position 0..100 vs
     a plain (`scaling="none"`) RoPE at the same theta, to sanity
     check that low-frequency dims are extrapolated (unscaled) and
     high-frequency dims are interpolated (scaled by `rope_factor`),
     per the YaRN paper's NTK-by-parts design.
2. **debugmodel dry run:** `MODULE=agpt CONFIG=agpt_debugmodel`
   equivalent for the new flavor — add a
   `"debugmodel_yarn"` flavor (tiny dims) temporarily or just run
   `agpt_2b_yarn` with `TRAINING_STEPS=5` and `--training.steps 5
   --dry-run` (if supported) / a short local single-node run to
   confirm the trainer builds without shape errors and the loss is
   finite for a handful of steps at the extended `seq_len`.
3. **Actual continued-pretraining run:** launch `agpt_2b_yarn` on a
   real allocation per the command in section 3, watch loss for the
   first ~50-400 steps at the new seq_len — it should not spike/NaN
   (a spike indicates a `rope_factor`/`beta_fast`/`beta_slow`
   mismatch with the checkpoint's actual effective training length).
4. Confirm `Decoder.Config.update_from_config`'s seq_len assertion in
   `common/decoder.py` (~line 205) triggers correctly if someone sets
   `SEQ_LEN` beyond `max_seq_len=262144` — this is existing code, just
   confirm it still fires post-change.
