#!/usr/bin/env bash
set -euo pipefail

# ==========
# User config
# ==========
PROJ="/project/rhino-ffm/reranker_ce"
VENV_DIR="${PROJ}/.venv"
REQ_FILE="${PROJ}/requirements.txt"
REQ_HASH_FILE="${VENV_DIR}/.requirements.sha256"
export HF_TOKEN="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"

cd "$PROJ"

# ==========
# Modules (SGE)
# ==========
module purge || true
module load cuda/12.8
module load python3/3.10.12
module load cmake/3.31.7

echo "[bootstrap] PROJ=$PROJ"
echo "[bootstrap] python3=$(which python3)"
python3 -V

# ==========
# Identify job + host
# ==========
JOB_TAG="${JOB_ID:-$$}"
HOST_TAG="$(hostname)"
echo "[bootstrap] JOB_ID=${JOB_ID:-<none>} HOST=$HOST_TAG JOB_TAG=$JOB_TAG"

# ==========
# Local tmp (avoid NFS .nfs* unlink issues)
# ==========
export TMPDIR="/tmp/${USER}/reranker_tmp_${JOB_TAG}"
mkdir -p "$TMPDIR"
chmod 700 "$TMPDIR" || true
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
echo "[bootstrap] TMPDIR=$TMPDIR"

# ==========
# HF token (DO NOT hardcode here)
# ==========
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[ERROR] HF_TOKEN is not set in the job environment."
  echo "SGE example: qsub -v HF_TOKEN=hf_... <job.sh>"
  exit 1
fi

# ==========
# HF caches (put on scratch)
# ==========
# Hugging Face supports relocating caches via HF_HOME / HF_*_CACHE env vars. [web:609][web:453]
export HF_HOME="/scratch/${USER}/hf"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "$HF_DATASETS_CACHE" "$HF_HUB_CACHE"
echo "[bootstrap] HF_HOME=$HF_HOME"

# Hub timeouts (optional but fine)
export HF_HUB_ETAG_TIMEOUT=300
export HF_HUB_DOWNLOAD_TIMEOUT=300

# Ensure logs flush immediately
export PYTHONUNBUFFERED=1
# Avoid tokenizer thread oversubscription when multiprocessing/forking
export TOKENIZERS_PARALLELISM=false

# ==========
# Create + activate venv
# ==========
if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  echo "[bootstrap] Creating venv at ${VENV_DIR}"
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
echo "[bootstrap] venv python=$(which python)"
python -V

# ==========
# Install requirements only when needed
# ==========
if [[ ! -f "$REQ_FILE" ]]; then
  echo "[ERROR] requirements.txt not found at: $REQ_FILE"
  exit 1
fi

NEW_HASH="$(sha256sum "$REQ_FILE" | awk '{print $1}')"
OLD_HASH=""
if [[ -f "$REQ_HASH_FILE" ]]; then
  OLD_HASH="$(cat "$REQ_HASH_FILE" || true)"
fi

if [[ "$NEW_HASH" != "$OLD_HASH" ]]; then
  echo "[bootstrap] Installing/updating Python deps (requirements changed)"
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install -r "$REQ_FILE"
  echo "$NEW_HASH" > "$REQ_HASH_FILE"
else
  echo "[bootstrap] requirements.txt unchanged; skipping pip install"
fi

# ==========
# Paths derived from config (no hardcoding run_name)
# ==========
RUN_DIR="$(python -c "import yaml; c=yaml.safe_load(open('configs/train.yaml')); print(f\"{c.get('project_dir','/project/rhino-ffm/reranker_ce')}/runs/{c.get('run_name','run')}\")")"
ART_DIR="${RUN_DIR}/artifacts"
PRETOK_SRC="${ART_DIR}/pretokenized"

echo "[bootstrap] RUN_DIR=$RUN_DIR"
echo "[bootstrap] PRETOK_SRC=$PRETOK_SRC"

# ==========
# Stage pretokenized dataset to scratch for fast training I/O
# ==========
JOB_SCRATCH="/scratch/${USER}/rhino-ffm/${JOB_TAG}"
PRETOK_DST="${JOB_SCRATCH}/pretokenized"
mkdir -p "$JOB_SCRATCH"

# If pretokenized doesn't exist yet, run the prepare step once (on this node).
# (prepare_artifacts.py downloads/processes/tokenizes and saves to PRETOK_SRC.)
if [[ ! -d "$PRETOK_SRC" ]]; then
  echo "[prepare] PRETOK_SRC missing; running prepare_artifacts.py to create it"
  python src/prepare_artifacts.py \
    --config configs/train.yaml \
    --num_proc 28 \
    --tok_batch_size 2048
fi

echo "[stage] rsync pretokenized -> scratch"
rsync -a --delete "$PRETOK_SRC/" "$PRETOK_DST/"

# ==========
# GPU monitoring (CSV)
# ==========
mkdir -p "$RUN_DIR/mon"
GPU_LOG="${RUN_DIR}/mon/gpu_${JOB_TAG}.csv"
echo "[mon] writing GPU stats to $GPU_LOG"
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv -l 5 > "$GPU_LOG" &
GPU_MON_PID=$!
trap 'kill $GPU_MON_PID 2>/dev/null || true' EXIT

# ==========
# Train (resume by default)
# ==========
echo "[train] starting training from scratch-staged dataset"
python src/train_reranker.py \
  --config configs/train.yaml \
  --pretokenized_dir "$PRETOK_DST" \
  --resume

echo "[done] job finished"
