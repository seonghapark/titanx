# Report: hardcoded datasets, models, and output paths

Date: 2026-10-08
Scope: `run_train_agpt.sh`, `run_train_torchtitan.sh`, `calculate_agpt_params.sh`,
`titan_train.py`, `../run_train.sh`. `torchtitan_repo/` is upstream and excluded
except where a launcher default depends on it.

Severity: **High** = wrong/unportable results or silently ignored input ·
**Medium** = portability or correctness risk off this machine ·
**Low** = site constant, reasonable as-is.

## High

### H1 — Calculator advertises an optimizer, LR, and dataset that never take effect
`calculate_agpt_params.sh:31-33` hardcodes `OPTIMIZER="sophiag"`,
`LR="2.28e-5"`, `DATASET="blendcorpus"`, emitted as `AGPT_OPTIMIZER`,
`AGPT_LR`, `AGPT_DATASET`. `run_train_agpt.sh` reads **none** of them (verified:
zero references). Runs therefore use torchtitan's own optimizer/LR from the
config registry, and dataset `pg19_multinews` — not SophiaG at 2.28e-5 on
blendcorpus. The printed summary asserts a configuration that is not what runs.

`blendcorpus` additionally has no registered dataset entry on this path; it
lives under the now-deleted `experiments/ezpz/`.

*Addressed in part:* `--optimizer`, `--lr`, `--dataset-path` are now options
with env overrides, and the dataset default is repointed at the real PG19
directory. **Still unresolved:** the wrapper does not consume
`AGPT_OPTIMIZER` / `AGPT_LR`, so they remain display-only. Wiring them changes
what trains and needs an explicit decision.

### H2 — `MODULE` / `CONFIG` hardcoded in the AGPT wrapper
`run_train_agpt.sh:377-378` unconditionally exports `MODULE="agpt"` and
`CONFIG="agpt_${AGPT_MODEL}"`, overwriting any caller value. `MODULE=agpt_yarn
CONFIG=agpt_2b_yarn` was silently discarded in a dry-run. Blocks every
non-default AGPT variant (`_yarn`, `_flex_attn`, `_chunkedce`, `_real`, …) —
roughly 20 registry entries unreachable through this wrapper.

Fix tracked in `AGPT_CLI_COMPATIBILITY_PLAN.md` (D3).

## Medium

### M1 — Absolute interpreter path baked into two launchers
- `run_train_torchtitan.sh:180` — `TRAIN_PYTHON_BIN_DEFAULT="/lus/flare/projects/datascience/seonghapark/venv/bin/python"`
- `../run_train.sh:117` — `PYTHON_BIN="${PYTHON_BIN:-/lus/flare/…/seonghapark/venv/bin/python}"`

Both embed one user's home venv, so the scripts are not usable by another user
or on another filesystem without edits. `run_train_torchtitan.sh` partly
mitigates with a `${TORCHTITAN_ROOT}/.venv/bin/python` fallback and a
`TRAIN_PYTHON_BIN` override; `run_train.sh` offers only the `PYTHON_BIN`
override. Prefer resolving from `VIRTUAL_ENV`, then `.venv`, then `PATH`, with
the absolute path as last resort.

### M2 — Default model path assumes a home-directory layout
`run_train_agpt.sh:331` — `MODEL_PATH="${HOME}/models/agpt-${MODEL_SIZE}"`.
No such tree exists for the current user; every real invocation must pass
`--model-path`. The default is effectively dead and misleading in `--help`.

### M3 — Dataset defaults diverge between the two layers
`run_train_torchtitan.sh:173` defaults to `pg19_multinews` (streaming,
network-dependent), while the calculator claims `blendcorpus`. Neither matches
the on-disk PG19 the CLI points at. A run that loses network, or an offline
node, silently changes behavior. The calculator default is now the local PG19
path; `run_train_torchtitan.sh` is unchanged (it is the reference script).

## Low — site constants, acceptable

| Location | Value | Note |
|---|---|---|
| `calculate_agpt_params.sh:34` | `DEVICES_PER_NODE=12` | Aurora XPU per-node count; correct here, wrong on any other machine. Worth an env override if these scripts ever leave Aurora. |
| `calculate_agpt_params.sh:25` | `TRAIN_TOKENS=4673780159710` | 4.67T budget; already overridable via `--train-tokens` / env. |
| `calculate_agpt_params.sh:26` | `DATA_LIST=olmo-mix-1124` | Already overridable via `--data-list`. Feeds only the checkpoint-directory name. |
| `calculate_agpt_params.sh:35-36` | `CKPT_INTERVAL=100`, `CKPT_KEEP_LATEST_K=0` | Computed, never emitted or consumed — dead values. |
| `run_train_torchtitan.sh:177` | `CKPT_FOLDER=checkpoint` | Overridable. |

## Output paths — no remaining issues

Checked after this session's fixes:
- `run_train_agpt.sh` default `LOG_DIR` is now absolute, rooted at `SCRIPT_DIR`
  (was relative, and resolved against a different directory downstream because
  `run_train_torchtitan.sh:247` `cd`s to `TORCHTITAN_ROOT` before launching).
- `run_train_torchtitan.sh:175` defaults to `${TORCHTITAN_ROOT}/outputs/…` —
  absolute, timestamped.
- `RESOURCE_OUTPUT_DIR` defaults to `${LOG_DIR}/resource_metrics` — derived,
  absolute.
- Checkpoint dir is a relative *name* (`agpt-2b-sophiag-olmo-mix-1124-n8-gbs192`)
  resolved by torchtitan under its dump folder — intentional, not a path bug.

No output path is written outside `LOG_DIR` or the torchtitan dump folder.

## Suggested order

1. **H2** — small, unblocks ~20 configs, no behavior change when unset.
2. **H1 (remainder)** — decide whether `AGPT_OPTIMIZER`/`AGPT_LR` should be
   forwarded or dropped from the summary; either way stop asserting untruths.
3. **M2** — drop the dead default or point it somewhere real.
4. **M1** — interpreter resolution, if these scripts are ever shared.
5. **M3** — align dataset defaults once H1 is settled.
