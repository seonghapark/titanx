# Implementation Summary — Hardcoded Values, Checkpoint Forwarding, Dataset-Derived Budgets

**Date:** 2026-10-08
**Scope:** `xpu_launcher/run_train.sh`, `xpu_launcher/xpu_torchtitan/{run_train_torchtitan.sh, run_train_agpt.sh, calculate_agpt_params.sh, titan_train.py}`
**Companion documents:** `HARDCODED_VALUES_REPORT.md` (findings), `AGPT_CLI_COMPATIBILITY_PLAN.md`, `AGPT_CONVENTION_ALIGNMENT_PLAN.md`

## Overview

This round closes every finding in `HARDCODED_VALUES_REPORT.md` (H1, H2, M1–M3), replaces two
hardcoded dataset constants with values derived from the dataset itself, and forwards the checkpoint
retention settings into the actual training command.

`torchtitan_repo` was left untouched, as previously agreed. All changes live in launcher code.

---

## 1. H1 — `AGPT_OPTIMIZER` / `AGPT_LR` now actually reach training

**Problem:** The calculator computed `AGPT_OPTIMIZER` and `AGPT_LR`, printed them in the summary, and
then dropped them. No optimizer or learning-rate flag was ever passed to torchtitan, so training
silently used whatever the model registry hardcoded (`default_adamw(lr=8e-4)`), not the requested
`2.28e-5` — a ~35× difference in learning rate.

**Why it was missed:** torchtitan has no flat `--optimizer.lr`. The optimizer is a *list* of parameter
groups (`optimizer.param_groups[]`), each with its own name and kwargs dict, so the real flags are
index-addressed. This was confirmed empirically by introspecting the generated tyro CLI:

```
--optimizer.param-groups.0.optimizer-name STR
--optimizer.param-groups.0.optimizer-kwargs.lr FLOAT
```

**Implementation** (`run_train_torchtitan.sh`):

```bash
OPTIMIZER_NAME="${OPTIMIZER_NAME:-${AGPT_OPTIMIZER:-}}"
LEARNING_RATE="${LEARNING_RATE:-${AGPT_LR:-}}"
OPTIMIZER_PARAM_GROUP="${OPTIMIZER_PARAM_GROUP:-0}"
```

Both are **unset by default**, so a plain `run_train_torchtitan.sh` keeps deferring to the model
registry; a flag is emitted only when a value is supplied. Group `0` is AGPT's catch-all `".*"` group.
`run_train_agpt.sh` now exports `OPTIMIZER_NAME` / `LEARNING_RATE` from the calculator output.

**Optimizer default changed:** `sophiag` → `AdamW`. `OptimizersContainer._resolve_optimizer_cls`
registers only `Adam` and `AdamW`; `sophiag` raises `NotImplementedError` at optimizer-build time, so
the previous default could never have run.

## 2. H2 — `MODULE` / `CONFIG` no longer hardcoded

Already addressed in the prior round and re-verified here:

```bash
export MODULE="${MODULE:-agpt}"
export CONFIG="${CONFIG:-agpt_${AGPT_MODEL}}"
```

Caller-supplied values win, so AGPT variants (`agpt_2b_yarn`, `*_flex_attn`, …) are reachable through
the wrapper. Confirmed by dry run: `MODULE=agpt CONFIG=agpt_2b_yarn` propagates to `--module` / `--config`.

## 3. M1 — Interpreter path resolved instead of hardcoded

`run_train_torchtitan.sh` and `run_train.sh` both hardcoded
`/lus/flare/projects/datascience/seonghapark/venv/bin/python`. Both now resolve in order:

1. `$VIRTUAL_ENV/bin/python` (an activated virtualenv)
2. `.venv/bin/python` beside the checkout
3. `$TORCHTITAN_VENV` / `$XPU_VENV` — the site venv (default: the former hardcoded path)
4. `python3` from `PATH` — last resort

**Ordering note:** an earlier revision put `PATH` lookup *above* the site venv. A dry run caught this
immediately — it selected `/usr/bin/python3`, which has no torch, and would have broken every launch.
The site venv is deliberately ranked above bare `python3`; override with `TRAIN_PYTHON_BIN`,
`PYTHON_BIN`, `TORCHTITAN_VENV`, or `XPU_VENV`.

## 4. M2 — Dead `${HOME}/models/...` default removed

`run_train_agpt.sh` defaulted `MODEL_PATH` to `${HOME}/models/agpt-${MODEL_SIZE}`, a path that does not
exist for any user; the run then failed with a confusing "not found" naming a path nobody chose. The
default is gone, replaced by an actionable error:

```
error: model assets path is required for agpt-2b
       pass --model-path PATH, or set MODEL=/path/to/agpt-2b-...
```

## 5. M3 — Dataset defaults bridged

The launcher's generic default (`pg19_multinews`, streaming) and the calculator's (`pg19` on local
parquet) serve different purposes, so neither was changed. They are now bridged by `AGPT_*` fallback so
calculator output flows through unchanged:

```bash
DATASET_NAME="${DATASET_NAME:-${AGPT_DATASET:-pg19_multinews}}"
DATASET_PATH="${DATASET_PATH:-${AGPT_DATASET_PATH:-}}"
```

---

## 6. `TRAIN_TOKENS` and `DATA_LIST` derived from the dataset

Previously `TRAIN_TOKENS=4673780159710` (4.67T) and `DATA_LIST=olmo-mix-1124` were constants describing
a corpus unrelated to the configured dataset. Both are now derived, and both remain overridable.

- **`DATA_LIST`** defaults to the dataset's own name (`pg19`).
- **`TRAIN_TOKENS`** is derived from dataset size, in priority order:
  1. Sum of `num_bytes` across splits in the HuggingFace dataset card (`README.md` / `dataset_infos.json`)
  2. On-disk size (`du -sb`) of the dataset directory
  3. Hard error naming the path, with the flags to fix it

  Tokens are estimated as `bytes / BYTES_PER_TOKEN` (default 4, `--bytes-per-token` to tune).

**Verified against the real PG19 assets:** the card reports `dataset_size: 11511573599`, yielding
**2,877,893,399 tokens** and 1,829 steps at GBS=192 / seq_len=8192. The dry-run output names its source
(`Token Budget Source: dataset card (num_bytes)`).

> **Behavioral note:** the derived budget is ~1,600× smaller than the old 4.67T constant, because that
> constant described olmo-mix-1124, not PG19. Step counts therefore drop sharply — this is the
> correction, not a regression. Pass `--train-tokens` to pin the old value.

## 7. Checkpoint interval / retention forwarded

`CKPT_INTERVAL` and `CKPT_KEEP_LATEST_K` were computed in the calculator and never used. They are now
emitted (`AGPT_CKPT_INTERVAL`, `AGPT_CKPT_KEEP_LATEST_K`, plus a `checkpoint` block in JSON output),
consumed by `run_train_torchtitan.sh`, and forwarded:

```bash
--checkpoint.interval 100 --checkpoint.keep_latest_k 0
```

`keep_latest_k = 0` means **keep every checkpoint**: torchtitan's `CheckpointManager` purges only under
`if self.keep_latest_k > 0`, so zero disables deletion entirely — exactly the requested semantics.
Validation allows zero while rejecting negatives, and new flags `--ckpt-interval` /
`--ckpt-keep-latest-k` expose both.

## 8. Import-path layout (no change required)

An earlier draft of this document claimed `torchtitan/hf_datasets/` and `torchtitan/experiments/` had
been deleted from the repo tree, and described a `_extend_torchtitan_path()` shim in `titan_train.py`
that backfilled them from the installed wheel. **That premise was wrong, and the shim has been
removed.** It is recorded here because the claim was previously published.

Re-verified on disk: `git status --porcelain` is clean, and both directories are present in
`torchtitan_repo/torchtitan/`. The two trees are:

| | `hf_datasets/`, `experiments/` | `models/agpt/` |
|---|---|---|
| site-packages `torchtitan` 0.3.0 | present | **missing** |
| `torchtitan_repo/torchtitan` | present | present |

`titan_train.py` inserts `TORCHTITAN_ROOT` at `sys.path[0]`, and the repo `torchtitan/` is a *regular*
package (it has `__init__.py`), so it fully shadows the installed wheel. Because the repo copy is
complete, that shadowing is exactly what is wanted — every import resolves to the repo, including the
`models/agpt/` package the wheel lacks. No fallback mechanism is needed.

**Verified resolution** (via `importlib.util.find_spec`, which resolves without executing):

```
REPO  torchtitan.hf_datasets.text_datasets
REPO  torchtitan.experiments
REPO  torchtitan.models.agpt
```

The shim was not merely redundant but actively harmful: the wheel's `hf_datasets/text_datasets.py`
registers only `c4`, `c4_test`, and `c4_validation`, whereas the repo copy also registers `pg19` and
`pg19_multinews`. Backfilling from the wheel could have shadowed the only copy that knows about PG19.

---

## Verification

All checks were `--dry-run` or static; no training job was launched.

- `bash -n` clean on all four shell scripts; `ast.parse` clean on `titan_train.py`.
- **End-to-end dry run** of the target CLI (`MODULE=agpt CONFIG=agpt_2b_yarn`, PG19 path, `--training.steps 400`,
  `--training.seq_len 16384`) through both `run_train_torchtitan.sh` and `run_train_agpt.sh`; all four new
  flags confirmed in the final `mpiexec` line.
- **Override precedence:** passing `--checkpoint.interval 7` and `...optimizer-kwargs.lr 9e-9` after `--`
  yields exactly **one** occurrence of each flag, carrying the user's value — `has_extra_arg` suppresses
  the default rather than emitting a duplicate.
- **Error paths:** unreadable dataset path, negative `keep_latest_k`, and missing model path each produce
  a specific, actionable message.
- **Import resolution** confirmed via `find_spec` (table in §8); all three modules resolve to the repo.
- **`DATASET_NAME=PG19`** confirmed safe: `HuggingFaceTextDataset.__init__` lowercases the name before
  `_validate_dataset`, and both the plain and interleaved dataloaders construct through it. The local
  parquet directory carries `train-`/`test-`/`validation-` prefixes and a `text` column, matching
  `_process_c4_text`.
- **No `ezpz` dependency**, verified by full-tree survey: zero imports anywhere, and the package is not
  installed in the venv. The `ezpz_agpt_*` registry functions and `EzpzScaledDotProductAttention` are
  locally defined in the vendored tree — a fork naming convention, not a package reference.

### Known limitations

- **`torch` cannot be imported on a login node** (`libpti_view.so.0: cannot open shared object file`).
  This is pre-existing and environmental — the launcher assumes a loaded Aurora framework module — so
  imports were verified via `importlib.util.find_spec`, which resolves modules without executing them.
  Import-time behavior under a real job environment is therefore verified structurally, not at runtime.
- **The vendored tree advertises two experiments that do not exist.**
  `torchtitan/experiments/__init__.py:17-18` lists `"ezpz.moe"` and `"ezpz.qwen3"` in
  `_supported_experiments`, but `torchtitan/experiments/ezpz/` is absent, so passing either as
  `--module` would raise at `config/manager.py:118-121`. Comments at `config_registry.py:188` and
  `:215` likewise point at `experiments/ezpz/{validator,trainer}.py`, neither of which is present.
  Reported rather than fixed: these are inside `torchtitan_repo`, which is off-limits. They do not
  affect the target CLI, which uses `MODULE=agpt`.
- **`typeguard` is absent from the venv — the one outstanding blocker.** It is a *declared, mandatory*
  dependency of tyro (`typeguard>=4.0.0`), which torchtitan uses to build its CLI. tyro imports it
  lazily from `_resolver.is_instance`, so it only fires for complex-typed defaults — but introspecting
  the AGPT config in this session did reach it and raised
  `ModuleNotFoundError: No module named 'typeguard'`. The venv sets
  `include-system-site-packages = true` and it is still not resolvable. Fix before launching:

  ```bash
  /lus/flare/projects/datascience/seonghapark/venv/bin/python -m pip install 'typeguard>=4.0.0'
  ```

  Not installed as part of this change: it mutates a shared venv, which is the user's call.
- **Token estimate is approximate** — `bytes / 4` is a heuristic, not a tokenizer count. For an exact
  budget, tokenize the corpus and pass `--train-tokens`.

## Files changed

| File | Change |
|---|---|
| `xpu_torchtitan/calculate_agpt_params.sh` | Derive `TRAIN_TOKENS`/`DATA_LIST`; `AdamW` default; emit checkpoint settings; `--bytes-per-token`, `--ckpt-interval`, `--ckpt-keep-latest-k` |
| `xpu_torchtitan/run_train_torchtitan.sh` | Forward optimizer/LR/checkpoint flags; resolve interpreter; `AGPT_*` dataset fallback; summary lines |
| `xpu_torchtitan/run_train_agpt.sh` | Export optimizer/LR/checkpoint to launcher; remove dead `MODEL_PATH` default |
| `run_train.sh` | Resolve interpreter instead of hardcoding |
