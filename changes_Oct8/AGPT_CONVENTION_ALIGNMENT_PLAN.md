# Plan: Align `run_train_agpt.sh` with `run_train_torchtitan.sh` conventions

Target file: `xpu_torchtitan/run_train_agpt.sh`
Reference file: `xpu_torchtitan/run_train_torchtitan.sh` (not modified)

Status: **revised 2026-10-08** after re-verifying the original plan against the
current file. All five items in the previous revision were already implemented;
this revision replaces them with the divergences that actually remain.

## Already done (verified, no action)

The previous plan's items 1-5 are present in the current file and need no work:

| # | Item | Evidence |
|---|------|----------|
| 1 | No-args usage check | `run_train_agpt.sh:104` |
| 2 | Lowercase `error:` style, no `error()` helper | 9 inline sites; no `Error:` remains |
| 3 | Executable check + chmod hint on downstream script | `run_train_agpt.sh:333-337` |
| 4 | Bracketed 80-`=` summary block | `run_train_agpt.sh:394-431` |
| 5 | Unconditional print-then-exec | `run_train_agpt.sh:433-437` |

## Changes

### 1. Forward computed parallelism/batch to torchtitan (correctness bug)

The calculator computes `AGPT_TP/PP/CP/LBS/GAS`; the summary prints all five;
**none reach torchtitan.** Verified by dry-run: `--model 80b --tp 4` prints
`Tensor Parallelism = 4`, but the launched command contains no
`--parallelism.*` flag, so the job runs at `agpt_80b`'s registry value
`tensor_parallel_degree=2`. The user's override is silently discarded.

Add to `TRAINING_ARGS`, each guarded by a `has_extra_arg` check (item 2) so an
explicit user `--` argument still wins:

```
--parallelism.tensor_parallel_degree   $AGPT_TP
--parallelism.pipeline_parallel_degree $AGPT_PP
--parallelism.context_parallel_degree  $AGPT_CP
--training.local_batch_size            $AGPT_LBS
```

Flag names and sections verified against
`torchtitan_repo/torchtitan/config/configs.py` (`TrainingConfig` line 28,
`ParallelismConfig` line 97) and `torchtitan/trainer.py:95`.

`AGPT_GAS` is **not** forwarded: torchtitan exposes no gradient-accumulation
field (only `pipeline_parallel_microbatch_size`, a different concept), and the
calculator already folds GAS into `AGPT_GBS`. It stays a display-only value.

**Accepted behavior change** (confirmed with the user): `agpt_2b` resolves to
`agpt("2b", ...)` whose default `local_batch_size=1`, while the calculator says
`LBS=2`. After this change 2B runs at the advertised LBS=2 rather than 1, which
roughly doubles per-device batch and memory. This is the point of the fix — the
summary becomes true — but it is a real change to what trains.

### 2. Adopt the `has_extra_arg` guard convention

`run_train_torchtitan.sh:145-154` defines `has_extra_arg()` and gates every
defaulted flag with it. `run_train_agpt.sh` has no equivalent and appends
`--training.steps` / `--training.seq_len` unconditionally, so a user override
produces the flag twice.

Verified harmless today (argparse is last-wins, so the user's value still
takes effect) — this is duplicate-noise cleanup, not a bug fix. It becomes
load-bearing for item 1, where the new flags must not stomp user overrides.

Port the same function and gate `--training.steps`, `--training.seq_len`, and
the four new flags from item 1.

### 3. Fix `MODEL` clobbered before the summary prints

`run_train_agpt.sh:359` runs `export MODEL="$MODEL_PATH"` to hand the path to
the downstream script, overwriting the `2b|20b|80b` size held in `MODEL`. The
summary at line 400 then prints `Model = <path>` and `Model Path = <path>` —
the size is lost. Confirmed in a dry-run: both lines showed the same path.

Capture the size into a separate variable (e.g. `MODEL_SIZE="$MODEL"`) before
the export and print that, so `[MODEL]` reads:

```
  Model                  = 2b
  Model Path             = /path/to/model
```

### 4. Make the default `LOG_DIR` absolute

`run_train_agpt.sh:327` defaults to the relative `outputs/agpt_<model>_<ts>`
and `mkdir -p`s it relative to the **caller's** CWD.
`run_train_torchtitan.sh:247` then does `cd "$TORCHTITAN_ROOT"` before
launching, so the same relative string resolves against a different directory
and the created directory is not the one training writes to. Demonstrated in
`/tmp`: created `cwd/outputs/...`, passed a path resolving to
`titanroot/outputs/...`.

Root it at `SCRIPT_DIR` (matching torchtitan's absolute
`${TORCHTITAN_ROOT}/outputs/...` default) so the directory created is the
directory used. Explicit `--log-dir` is unaffected.

## Out of scope

- `calculate_agpt_params.sh` — not the file named by the user. Note it still
  uses the old capitalized `error()` style; aligning it is a separate change.
- `AGPT_OPTIMIZER`, `AGPT_LR`, `AGPT_DATASET`, `AGPT_DATA_LIST`, `AGPT_NGPUS`,
  `AGPT_GBS` are computed and never used. In particular the calculator reports
  `DATASET=blendcorpus` while runs actually use torchtitan's default
  `pg19_multinews` (`blendcorpus` lives under `experiments/ezpz/` and is not a
  registered dataset for this path). Left alone deliberately: wiring the
  dataset would change what data trains, which is well beyond a convention
  alignment and needs its own decision.
- No new flags, defaults, or exported env vars beyond those listed above.

## Verification

For each step: `--dry-run` for 2b/20b/80b in `single` and `multi`, diffing the
launched command against the pre-change baseline, plus `bash -n`. Expected
command-line delta is confined to the four new flags in item 1 and the removal
of duplicate `--training.*` flags from item 2; items 3 and 4 change only the
summary text and the default log path.
