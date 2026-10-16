#!/usr/bin/env bash
#PBS -A AuroraGPT
#PBS -q debug-scaling
#PBS -k doe
#PBS -j oe
#PBS -l filesystems=home:flare
#PBS -l select=4
#PBS -l walltime=01:00:00
#PBS -N agpt2b_post16384
#
# AGPT-2B context-extension run: 400 steps at seq_len 16384 on PG19 (local
# parquet), AdamW, 4 nodes x 12 XPU = 48 ranks, TP=2 x FSDP=24.
#
# Final model is exported to:
#   /lus/flare/projects/datascience/seonghapark/agpt-2b-v2-256n-post-16384
#
# No `set -u`: lmod's init scripts reference unbound variables
# (ZSH_EVAL_CONTEXT), which would abort the job during `module load`.
set -o pipefail

BASE=/lus/flare/projects/datascience/seonghapark
LAUNCH=${BASE}/xpu_launcher
MODEL_IN=${BASE}/agpt-2b-v2-256n-step-92859-safetensors
DATASET_DIR=${BASE}/assets/hf/datasets/pg19/data
FINAL_OUT=${BASE}/agpt-2b-v2-256n-post-16384

STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=${LAUNCH}/post16384_${STAMP}
mkdir -p "$LOGDIR"

# ---------------------------------------------------------------------------
# Run parameters. Single source of truth: the banner, the launch and the
# report all read these, so the report cannot drift from what actually ran.
# ---------------------------------------------------------------------------
STEPS=400
SEQ_LEN=16384
TP=2             # tensor parallel degree
PP=1
CP=1
LBS=1            # local batch size; the only value measured at seq_len 16384
PPN=12           # XPU devices per Aurora node
OPT_NAME=AdamW
LR=2.28e-5
CKPT_INTERVAL=100
RESOURCE_INTERVAL=15

echo "=================================================================="
echo "=== job ${PBS_JOBID:-interactive} on $(hostname) at $(date) ==="
echo "=================================================================="
echo "logs       : $LOGDIR"
echo "model in   : $MODEL_IN"
echo "dataset    : $DATASET_DIR"
echo "final out  : $FINAL_OUT"

# --- environment -----------------------------------------------------------
# The module is "frameworks" (plural); "framework/2026.1.0" does not exist.
module load frameworks/2026.1.0 || module load frameworks || {
  echo "FATAL: could not load the frameworks module" >&2; exit 1; }
export LD_LIBRARY_PATH=/opt/aurora/26.181.0/frameworks/aurora_frameworks-2026.1.0/lib:${LD_LIBRARY_PATH:-}

cd "$BASE" || exit 1
source venv/bin/activate || { echo "FATAL: venv activate failed" >&2; exit 1; }

export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
export no_proxy="localhost,127.0.0.1"

export EVAL_BASE="${EVAL_BASE:-${BASE}/evaluation}"
export DATA_PATH="${DATA_PATH:-${EVAL_BASE}/datasets/}"
export HF_HOME="${EVAL_BASE}/models/cache"
export HF_DATASETS_CACHE="${DATA_PATH}"
export HF_DATASETS_OFFLINE=0
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_XET=1

export TORCHINDUCTOR_CACHE_DIR=${EVAL_BASE}/.cache/inductor_${PBS_JOBID:-local}
export VLLM_CACHE_ROOT=${EVAL_BASE}/.cache/vllm/_${PBS_JOBID:-local}
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

export CCL_OP_SYNC=1
export CCL_PROCESS_LAUNCHER=pmix
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export PYTHONMULTIPROCESSINGMETHOD=spawn

# wandb off via the environment, not a CLI flag: tyro renders bool fields as
# flags, so "--metrics.enable_wandb false" would leave a stray positional.
export WANDB_MODE=offline

# --- mitigations (both established by prior jobs; see memory) --------------
#   per-rank Inductor cache  -> job 8931618: 18/18 clean vs 6/17 flakes shared
#   guarded os._exit teardown-> job 8929540: ~505s stall -> ~0s, exit 124 -> 0
# Injected via PYTHONPATH sitecustomize, which displaces nothing. Do NOT use a
# TRAIN_PYTHON_BIN wrapper: that killed every rank in Triton compile (8924922).
SHIM="$LAUNCH/inductor_iso"
[[ -f "$SHIM/sitecustomize.py" ]] || { echo "FATAL: missing $SHIM/sitecustomize.py" >&2; exit 1; }
[[ -f "$LAUNCH/osexit_exp/sitecustomize.py" ]] || { echo "FATAL: missing osexit shim" >&2; exit 1; }
export PYTHONPATH="${SHIM}${PYTHONPATH:+:$PYTHONPATH}"
export AGPT_INDUCTOR_ISOLATE=1
export AGPT_OSEXIT_EXPERIMENT=1
export AGPT_CORE_LIMIT_BYTES=$((1024*1024*1024))   # cap cores at 1GB, in-process
echo "iso shim md5   : $(md5sum "$SHIM/sitecustomize.py" | awk '{print $1}')"
echo "osexit shim md5: $(md5sum "$LAUNCH/osexit_exp/sitecustomize.py" | awk '{print $1}')"

# --- preflight: refuse to burn the allocation on a bad input ---------------
if [[ ! -f "$MODEL_IN/config.json" ]] || ! ls "$MODEL_IN"/*.safetensors >/dev/null 2>&1; then
  echo "FATAL: $MODEL_IN is not a usable HF model directory." >&2
  exit 2
fi
if ! ls "$DATASET_DIR"/*.parquet >/dev/null 2>&1; then
  echo "FATAL: no parquet shards in $DATASET_DIR" >&2
  exit 2
fi
python3 - <<'DEPS' || { echo "FATAL: dependency check failed" >&2; exit 2; }
import importlib.util as u, sys
missing = [m for m in ("tyro", "typeguard", "tensorboard", "torch") if u.find_spec(m) is None]
print("deps missing:", missing if missing else "none")
sys.exit(1 if missing else 0)
DEPS

# --- topology --------------------------------------------------------------
NODES_ALLOC=$(awk 'NF {print $1}' "${PBS_NODEFILE:-/dev/null}" 2>/dev/null | sort -u | wc -l)
[[ "$NODES_ALLOC" -lt 1 ]] && NODES_ALLOC=1
WORLD=$(( NODES_ALLOC * PPN ))
FSDP=$(( WORLD / (TP * PP * CP) ))

echo
echo "--- topology ---"
echo "  nodes            = $NODES_ALLOC"
echo "  ranks/node (ppn) = $PPN"
echo "  world size       = $WORLD"
echo "  TP               = $TP"
echo "  PP               = $PP"
echo "  CP               = $CP"
echo "  FSDP (dp_shard)  = $FSDP"
echo "  check: TP*PP*CP*FSDP = $(( TP*PP*CP*FSDP )) (must equal $WORLD)"

if [[ $(( TP*PP*CP*FSDP )) -ne $WORLD ]]; then
  echo "FATAL: parallelism product does not equal world size" >&2; exit 2
fi
# torchtitan asserts seq_len % (tp * cp * 2) == 0
DIVISOR=$(( TP * CP * 2 ))
if (( SEQ_LEN % DIVISOR != 0 )); then
  echo "FATAL: seq_len $SEQ_LEN not divisible by tp*cp*2 = $DIVISOR" >&2; exit 2
fi
echo "  seq_len $SEQ_LEN % (tp*cp*2 = $DIVISOR) == 0  OK"

# Snapshot the core directory so this run's cores are not confused with the
# 8924552-era leftovers (that mistake sized a failure wrongly once already).
CORE_DIR="${LAUNCH}/xpu_torchtitan/torchtitan_repo"
CORES_BEFORE=$(ls -1 "$CORE_DIR"/core.* 2>/dev/null | wc -l)
echo "  cores before     = $CORES_BEFORE"

# --- launch ----------------------------------------------------------------
cd "${LAUNCH}/xpu_torchtitan" || exit 1

RES_DIR="${LOGDIR}/resource_metrics"
CKPT_LOAD_DIR=""   # weights come from HF assets, not a DCP checkpoint

echo
echo "=== launching at $(date) ==="

# Budget: measured 2.65 s/step at seq_len 16384 (job 8917595) -> 400 steps is
# ~18 min, plus ~6 min model load + Inductor compile. 2700s leaves >25 min of
# headroom inside the 1h allocation for the export step and the report.
MODEL="$MODEL_IN" \
MODULE=agpt \
CONFIG=agpt_2b_yarn \
DATASET_NAME=PG19 \
DATASET_PATH="$DATASET_DIR" \
OPTIMIZER_NAME="$OPT_NAME" \
LEARNING_RATE="$LR" \
RESOURCE_MONITOR=1 \
RESOURCE_INTERVAL=$RESOURCE_INTERVAL \
RESOURCE_OUTPUT_DIR="$RES_DIR" \
LOSS_STD_TERMINATION_ENABLED=1 \
LOSS_STD_THRESHOLD=0.001 \
LOSS_STD_WINDOW=50 \
CKPT_INTERVAL=$CKPT_INTERVAL \
CKPT_KEEP_LATEST_K=0 \
# The resource flags MUST be passed on the command line, not only as env vars.
# run_train_agpt.sh:42-44 does RESOURCE_MONITOR="" unconditionally at startup,
# which clobbers an inherited value; only the --resource-* flags survive. Job
# 8935820 exported RESOURCE_MONITOR=1 and still logged "Resource Monitor = 0"
# and collected zero samples.
timeout 2700 ./run_train_agpt.sh --model 2b --tp $TP --pp $PP --cp $CP --lbs $LBS multi \
  --resource-monitor \
  --resource-interval $RESOURCE_INTERVAL \
  --resource-output-dir "$RES_DIR" \
  -- \
  --training.steps $STEPS \
  --training.seq_len $SEQ_LEN \
  --training.local_batch_size $LBS \
  --parallelism.tensor_parallel_degree $TP \
  --parallelism.pipeline_parallel_degree $PP \
  --parallelism.context_parallel_degree $CP \
  2>&1 | tee "$LOGDIR/run.log"
TRAIN_RC=${PIPESTATUS[0]}

echo
echo "=== training finished at $(date), rc=$TRAIN_RC ==="

# Strip ANSI once; every downstream grep reads the clean copy.
CLEAN="$LOGDIR/run.clean.log"
sed 's/\x1b\[[0-9;]*m//g' "$LOGDIR/run.log" > "$CLEAN"

# ===========================================================================
# REPORT. Written BEFORE the expensive export/filesystem walk: job 8931618 was
# killed at walltime mid-report because `du -sh` over Lustre overran the
# reserve, and the verdict never printed.
# ===========================================================================
echo
echo "=================================================================="
echo "=== 1. HYPERPARAMETERS AS LAUNCHED ==="
echo "=================================================================="
grep -E "^ +[A-Za-z].*=" "$CLEAN" | head -60

echo
echo "--- trainer-side banner (authoritative: read from the live config) ---"
grep -E "^\[titan\].*(\[MODEL\]|\[TRAINING\]|\[OPTIMIZER\]|\[LR SCHEDULER\]|\[PARALLELISM\]|\[DATA\]|\[ARCHITECTURE|\[MEMORY|\[CHECKPOINT\]|\[DISTRIBUTED\]|RoPE |Learning Rate|Optimizer|Sequence Length|Steps|Batch)" "$CLEAN" \
  | sed 's/^\[titan\][^-]*- root - INFO - //' | head -80

echo
echo "=================================================================="
echo "=== 2. PARALLELISM VERIFICATION (TP and FSDP) ==="
echo "=================================================================="
echo "--- requested ---"
echo "  nodes=$NODES_ALLOC  world=$WORLD  TP=$TP  PP=$PP  CP=$CP  FSDP(dp_shard)=$FSDP"
echo
echo "--- observed: ParallelDims line from the run ---"
grep -oE "pp=[0-9]+, dp_replicate=[0-9]+, dp_shard=[0-9]+[^\"]*" "$CLEAN" | sort -u | head -5
grep -E "Building [0-9]+-D device mesh|device mesh|world_size" "$CLEAN" | sed 's/^\[titan\][^-]*- root - INFO - //' | sort -u | head -10
echo
echo "--- observed: FSDP / TP application ---"
grep -cE "Applied FSDP to the model" "$CLEAN" | xargs echo "  'Applied FSDP' lines   :"
grep -cE "Applied HSDP to the model" "$CLEAN" | xargs echo "  'Applied HSDP' lines   :"
echo
echo "--- observed: distinct hosts that ran ranks (proves multi-node) ---"
grep -oE "SHUTDOWN rank=[0-9]+" "$CLEAN" | sort -u | wc -l | xargs echo "  distinct ranks seen    :"
if [[ -f "$LOGDIR/hosts.txt" ]]; then wc -l < "$LOGDIR/hosts.txt" | xargs echo "  hosts in hostfile      :"; fi
awk 'NF {print $1}' "${PBS_NODEFILE:-/dev/null}" 2>/dev/null | sort -u | sed 's/^/    /' | head -8

echo
echo "=================================================================="
echo "=== 3. TRAINING PROGRESS ==="
echo "=================================================================="
grep -cE "Training completed" "$CLEAN" | xargs echo "  'Training completed' ranks :"
echo "  first / last step lines:"
grep -oE "step: *[0-9]+ +loss: *[0-9.]+ +grad_norm: *[0-9.]+ +memory: *[0-9.]+GiB\([0-9.]+%\) +tps: *[0-9,]+ +tflops: *[0-9.]+ +mfu: *[0-9.]+%" "$CLEAN" \
  | sort -u -t: -k2 -n | awk 'NR<=3 || NR>=total-2' total="$(grep -coE 'step: *[0-9]+ +loss:' "$CLEAN")" | head -8
echo
echo "  --- last 5 distinct steps ---"
grep -oE "step: *[0-9]+ +loss: *[0-9.]+[^m]*mfu: *[0-9.]+%" "$CLEAN" | sort -u | tail -5
echo
grep -E "loss std|early termination|Early termination|converged" "$CLEAN" | tail -5

echo
echo "=================================================================="
echo "=== 4. RESOURCE CONSUMPTION (sampled every ${RESOURCE_INTERVAL}s) ==="
echo "=================================================================="
echo "  output dir: $RES_DIR"
if [[ -d "$RES_DIR" ]]; then
  ls -1 "$RES_DIR" | head -10
  echo "  files: $(ls -1 "$RES_DIR" | wc -l)"
  # Aggregate ALL hosts. `head -1` here would report one node out of four.
  python3 -I "${LAUNCH}/agg_resources.py" "$RES_DIR" 2>&1 | sed 's/^/  /'
else
  echo "  WARNING: resource metrics directory not created"
fi
echo
echo "  --- peak memory reported by the trainer ---"
grep -oE "memory: *[0-9.]+GiB\([0-9.]+%\)" "$CLEAN" | sort -u -t'(' -k2 -rn | head -3

echo
echo "=================================================================="
echo "=== 5. SHUTDOWN / INTEGRITY ==="
echo "=================================================================="
grep -cE "os\._exit\(0\) NOW" "$CLEAN" | xargs echo "  ranks taking guarded os._exit :"
grep -cE "isolated cache" "$CLEAN" | xargs echo "  ranks with isolated Inductor  :"
grep -cE "@jit functions should be defined" "$CLEAN" | xargs echo "  @jit compile flake hits       :"
CORES_AFTER=$(ls -1 "$CORE_DIR"/core.* 2>/dev/null | wc -l)
echo "  cores before / after          : $CORES_BEFORE / $CORES_AFTER  (new: $(( CORES_AFTER - CORES_BEFORE )))"

# ===========================================================================
# EXPORT the trained model. Everything above has already printed, so a
# walltime kill here costs the export, not the report.
# ===========================================================================
echo
echo "=================================================================="
echo "=== 6. EXPORT -> $FINAL_OUT ==="
echo "=================================================================="

# --- GUARD: refuse to export a model that trained from random init --------
# Job 8935565 silently did exactly that: checkpoint.py passed an EMPTY dict to
# dcp.load(), which plans from the keys already present, so 0 of 111 tensors
# were read and load_state_dict() was a no-op -- while still logging "Finished
# loading the checkpoint in 1.12 seconds". The only visible symptom was the
# step-1 loss: 13.01, i.e. ln(vocab=256128)=12.45, a from-scratch curve.
# A continued-pretraining run must start FAR below that, so gate on it.
FIRST_LOSS=$(grep -oE "step: *1 +loss: *[0-9.]+" "$CLEAN" | head -1 | grep -oE "[0-9.]+$")
echo "  step-1 loss : ${FIRST_LOSS:-<not found>}  (random init would be ~12.45 = ln 256128)"
if [[ -n "$FIRST_LOSS" ]] && awk "BEGIN{exit !($FIRST_LOSS > 11.0)}"; then
  echo "  *** ABORT: step-1 loss $FIRST_LOSS indicates the pretrained weights did NOT load."
  echo "  *** Refusing to export a randomly-initialized model to $FINAL_OUT."
  exit 3
fi

DUMP_DIR=$(grep -oE -- "--dump_folder [^ ]+" "$CLEAN" | head -1 | awk '{print $2}')
echo "  dump_folder : ${DUMP_DIR:-<not found>}"
CKPT_ROOT=""
if [[ -n "$DUMP_DIR" && -d "$DUMP_DIR" ]]; then
  CKPT_ROOT=$(find "$DUMP_DIR" -maxdepth 1 -type d -name 'agpt-2b-*' | head -1)
fi
echo "  ckpt root   : ${CKPT_ROOT:-<not found>}"

LAST_STEP_DIR=""
if [[ -n "$CKPT_ROOT" && -d "$CKPT_ROOT" ]]; then
  LAST_STEP_DIR=$(find "$CKPT_ROOT" -maxdepth 1 -type d -name 'step-*' \
    | sed 's/.*step-//' | sort -n | tail -1 | sed "s|^|${CKPT_ROOT}/step-|")
fi
echo "  last step   : ${LAST_STEP_DIR:-<none>}"

if [[ -n "$LAST_STEP_DIR" && -f "$LAST_STEP_DIR/.metadata" ]]; then
  echo "  shards      : $(ls -1 "$LAST_STEP_DIR"/*.distcp 2>/dev/null | wc -l)"
  mkdir -p "$(dirname "$FINAL_OUT")"
  # Conversion MUST use the same --config the checkpoint was trained under:
  # to_hf() derives the Q/K rotary permutation from the RoPE class, and a
  # mismatch loads clean while training on corrupted attention.
  timeout 900 python3 "$LAUNCH/dcp_to_hf.py" \
    --dcp-path "$LAST_STEP_DIR" \
    --output "$FINAL_OUT" \
    --assets-from "$MODEL_IN" \
    --force \
    -- --module agpt --config agpt_2b_yarn --training.seq_len $SEQ_LEN \
    2>&1 | tee "$LOGDIR/export.log"
  EXPORT_RC=${PIPESTATUS[0]}
  echo "  export rc   : $EXPORT_RC"
  if [[ $EXPORT_RC -eq 0 ]]; then
    echo "  --- $FINAL_OUT ---"
    ls -la "$FINAL_OUT" | head -15
  fi
else
  EXPORT_RC=1
  echo "  SKIPPED: no usable checkpoint directory found."
  [[ -n "$CKPT_ROOT" ]] && find "$CKPT_ROOT" -maxdepth 1 -type d | head -10
fi

echo
echo "=================================================================="
echo "=== VERDICT ==="
echo "=================================================================="
COMPLETED=$(grep -cE "Training completed" "$CLEAN")
JITFLAKE=$(grep -cE "@jit functions should be defined" "$CLEAN")
if [[ $JITFLAKE -gt 0 ]]; then
  echo "  NOT TESTED - Inductor @jit first-compile flake (known, ~40% without"
  echo "               isolation; isolation was ON, so this is unexpected)."
elif [[ $TRAIN_RC -ne 0 && $COMPLETED -lt 1 ]]; then
  echo "  FAILED - training rc=$TRAIN_RC with no 'Training completed'."
elif [[ $COMPLETED -ge 1 && ${EXPORT_RC:-1} -eq 0 ]]; then
  echo "  SUCCESS - $COMPLETED ranks completed; model exported to"
  echo "            $FINAL_OUT"
elif [[ $COMPLETED -ge 1 ]]; then
  echo "  PARTIAL - training completed ($COMPLETED ranks) but export failed."
else
  echo "  UNKNOWN - rc=$TRAIN_RC, completed=$COMPLETED. Read $CLEAN."
fi
echo "  train rc = $TRAIN_RC ; logs = $LOGDIR"
echo "=================================================================="

exit $TRAIN_RC
