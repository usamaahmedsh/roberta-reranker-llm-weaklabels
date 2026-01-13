#!/usr/bin/env bash
set -euo pipefail

# check_run_outputs.sh
#
# Usage:
#   bash scripts/check_run_outputs.sh deberta_v3_bm25_mined_01
#   bash scripts/check_run_outputs.sh /project/rhino-ffm/reranker_ce/runs/deberta_v3_bm25_mined_01
#
# Optional env:
#   PROJ=/project/rhino-ffm/reranker_ce

PROJ="${PROJ:-/project/rhino-ffm/reranker_ce}"

arg="${1:-}"
if [[ -z "$arg" ]]; then
  echo "Usage: $0 <run_name|run_dir>"
  exit 1
fi

# Resolve RUN_DIR
if [[ "$arg" == /* ]]; then
  RUN_DIR="$arg"
else
  RUN_DIR="${PROJ}/runs/${arg}"
fi

echo "== Run dir =="
echo "$RUN_DIR"
[[ -d "$RUN_DIR" ]] || { echo "[ERROR] RUN_DIR not found"; exit 1; }

echo
echo "== Top-level files =="
ls -lah "$RUN_DIR" | sed -n '1,200p'

echo
echo "== Final model =="
FINAL_DIR="$RUN_DIR/final_model"
if [[ -d "$FINAL_DIR" ]]; then
  ls -lah "$FINAL_DIR" | sed -n '1,200p'
  echo
  echo "-- config.json / model files (if present) --"
  for f in config.json tokenizer_config.json tokenizer.json vocab.json merges.txt special_tokens_map.json \
           pytorch_model.bin model.safetensors; do
    [[ -f "$FINAL_DIR/$f" ]] && echo "  OK  $FINAL_DIR/$f"
  done
else
  echo "[WARN] No final_model directory: $FINAL_DIR"
fi

echo
echo "== Checkpoints =="
CKPT_DIR="$RUN_DIR/checkpoints"
if [[ -d "$CKPT_DIR" ]]; then
  echo "-- checkpoint folders --"
  ls -1 "$CKPT_DIR" 2>/dev/null | sed -n '1,200p' || true

  echo
  echo "-- latest checkpoint (by step) --"
  latest="$(ls -1 "$CKPT_DIR" 2>/dev/null | grep -E '^checkpoint-[0-9]+$' | sort -V | tail -n 1 || true)"
  if [[ -n "$latest" ]]; then
    echo "$CKPT_DIR/$latest"
    ls -lah "$CKPT_DIR/$latest" | sed -n '1,200p'
  else
    echo "[INFO] No checkpoint-* directories found."
  fi
else
  echo "[WARN] No checkpoints directory: $CKPT_DIR"
fi

echo
echo "== TensorBoard logs =="
TB_DIR="$RUN_DIR/tb"
if [[ -d "$TB_DIR" ]]; then
  ls -lah "$TB_DIR" | sed -n '1,200p'
  echo
  echo "-- most recent event files --"
  find "$TB_DIR" -maxdepth 2 -type f -name "events.out.tfevents.*" -printf "%TY-%Tm-%Td %TH:%TM  %s  %p\n" 2>/dev/null \
    | sort | tail -n 20 || true
else
  echo "[WARN] No tb directory: $TB_DIR"
fi

echo
echo "== Training throughput log (metrics.jsonl) =="
METRICS="$RUN_DIR/metrics.jsonl"
if [[ -f "$METRICS" ]]; then
  echo "-- last 5 lines --"
  tail -n 5 "$METRICS"
  echo
  echo "-- last record pretty-printed (if valid json) --"
  tail -n 1 "$METRICS" | python -m json.tool || true
else
  echo "[WARN] No metrics.jsonl found: $METRICS"
fi

echo
echo "== Config snapshot used for the run =="
CFG="$RUN_DIR/config.yaml"
if [[ -f "$CFG" ]]; then
  sed -n '1,200p' "$CFG"
else
  echo "[WARN] No config.yaml found: $CFG"
fi

echo
echo "== Done =="
