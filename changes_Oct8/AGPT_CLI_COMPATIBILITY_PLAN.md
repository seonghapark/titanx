# Plan: CLI compatibility for `run_train_agpt.sh` + calculator options

Date: 2026-10-08

## The CLI under test

```
MODEL=.../agpt-2b-v2-256n-step-92859-safetensors \
MODULE=agpt_yarn CONFIG=agpt_2b_yarn \
DATASET_NAME=PG19 DATASET_PATH=.../assets/hf/datasets/pg19/data \
RESOURCE_MONITOR=1 RESOURCE_INTERVAL=15 RESOURCE_OUTPUT_DIR="$PWD/resource_metrics_<ts>" \
LOSS_STD_TERMINATION_ENABLED=1 LOSS_STD_THRESHOLD=0.001 LOSS_STD_WINDOW=50 \
./<script> multi \
--resource-monitor --resource-interval 15 --resource-output-dir "$RESOURCE_OUTPUT_DIR" \
-- --training.steps 400 --training.max_duration_hours 4.0 --training.seq_len 16384
```

Both referenced paths exist on disk. `--training.max_duration_hours` is a real
field (`torchtitan/config/configs.py:92`, `TrainingConfig`, default 4.0).

## Verdict

| Script | Result |
|---|---|
| `run_train_torchtitan.sh` | **Works as-is.** Verified by dry-run: every env var and flag reaches the launched command, including `--dataloader.dataset PG19`, `--dataloader.dataset_path …`, the resource-monitor wrapper, `--training.enable_loss_std_termination`, and all three `--` args. No change needed. |
| `run_train_agpt.sh` | **Fails.** Exits with `error: --model is required`. Four distinct discrepancies below. |

## Discrepancies in `run_train_agpt.sh`

### D1 — `--model` is mandatory, CLI does not pass it (fatal)

The CLI identifies the model by the `MODEL=` *path*; the AGPT wrapper requires a
`--model {2b|20b|80b}` *size* to drive the calculator. Reproduced:
`error: --model is required`, then usage, exit 1. This is what `run.log`
already recorded.

**Fix:** when `--model` is absent but `MODEL`/`MODEL_PATH` is set, infer the
size from the path basename (`agpt-2b-…` → `2b`) and fall back to a clear error
naming both options if inference fails. Accept `MODEL` as the path, matching
`run_train_torchtitan.sh:158`.

### D2 — flags placed after the mode word are silently discarded (severe)

The parser's `single|multi)` branch consumes the mode, optionally one hostfile,
then collects `EXTRA_ARGS` **only if** the next token is exactly `--`;
otherwise it `break`s. The CLI puts `--resource-monitor --resource-interval 15
--resource-output-dir …` after `multi`, so `$1` is `--resource-monitor`, the
`--` check fails, and *everything* after the mode is dropped — including the
`-- --training.steps 400 --training.max_duration_hours 4.0 --training.seq_len
16384` that follows it.

Confirmed by dry-run: the launched command contained none of those, and ran
with the calculator's `--training.steps 23772075` instead of the requested
`400`. This is the most dangerous discrepancy: no error, no warning, a wildly
different job than the one requested.

**Fix:** continue option parsing after the mode instead of breaking. Accept the
monitoring flags in either position, and treat a later `--` as the extra-args
separator.

### D3 — `MODULE` / `CONFIG` are overridden, not honored

`run_train_agpt.sh:377-378` unconditionally does `export MODULE="agpt"` and
`export CONFIG="agpt_${AGPT_MODEL}"`. With `MODULE=agpt_yarn
CONFIG=agpt_2b_yarn` in the environment, the dry-run still launched `--module
agpt --config agpt_2b`. The caller's YaRN request is silently ignored.

**Fix:** respect caller-supplied values — `MODULE="${MODULE:-agpt}"`,
`CONFIG="${CONFIG:-agpt_${AGPT_MODEL}}"` — the `:-` convention used throughout
`run_train_torchtitan.sh`.

Note `agpt_2b_yarn` exists (`config_registry.py:404`) but **`agpt_yarn` is not
a valid module**: `_supported_models` is `{agpt, deepseek_v3, flux, gpt_oss,
llama3, qwen3, qwen3_5}`, and no `agpt_yarn` package exists to satisfy the
fully-qualified fallback in `manager.py:126-133`. The correct invocation is
`MODULE=agpt CONFIG=agpt_2b_yarn`. Honoring MODULE turns a silent override into
an explicit, debuggable failure.

### D4 — `DATASET_NAME` / `DATASET_PATH` never forwarded

The wrapper sets neither, so `run_train_torchtitan.sh` applies its own default
and the run used `--dataloader.dataset pg19_multinews` with no dataset_path,
ignoring `DATASET_NAME=PG19` and the explicit PG19 directory.

**Fix:** pass both through when set, defaulting to the calculator's values.

## Calculator changes (requested separately)

In `calculate_agpt_params.sh`:

1. Default `DATASET` to the on-disk PG19 data, and add `--dataset-path` plus a
   new `AGPT_DATASET_PATH` output. Default:
   `/lus/flare/projects/datascience/seonghapark/assets/hf/datasets/pg19/data`
   (env-overridable via `DATASET_PATH`).
2. Add `--optimizer`, overriding the hardcoded `sophiag`.
3. Add `--lr`, overriding the hardcoded `2.28e-5`.

All three keep their current values as defaults except the dataset, which the
user asked to repoint at PG19. Each becomes env-overridable
(`${VAR:-default}`) to match the house convention.

## Out of scope / blocked

`torchtitan_repo` currently has 827 uncommitted deletions
(`torchtitan/hf_datasets/`, `torchtitan/experiments/`), while
`models/agpt/config_registry.py:16` still imports
`torchtitan.hf_datasets.text_datasets` unguarded. Any real AGPT run therefore
fails at import. Confirmed with the user as an intentional in-progress 0.3.0
cleanup, so it is **not** touched here and all verification below is static
(`--dry-run` + launched-command inspection) rather than a live training run.

## Verification

Per change: `bash -n`, then the exact CLI above under `--dry-run` against both
scripts, asserting the launched command carries `--module`, `--config`,
`--dataloader.dataset`, `--dataloader.dataset_path`, the resource-monitor
wrapper, and all three `--` training args with the requested values.
