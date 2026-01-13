#!/usr/bin/env python3
import os
import argparse
from typing import Dict, Any, Optional

import yaml
from datasets import load_dataset, load_from_disk
from sentence_transformers import CrossEncoder

# -----------------------------------------------------------------------------
# HOW TO RUN
#
# One-shot run (process -> tokenize):
#   python src/prepare_artifacts.py \
#     --config configs/train.yaml \
#     --num_proc 28 \
#     --tok_batch_size 256
#
# Resume (reuse cached processed triples):
#   python src/prepare_artifacts.py \
#     --config configs/train.yaml \
#     --skip_proc \
#     --num_proc 28 \
#     --tok_batch_size 256
#
# Notes:
# - --force deletes the pretokenized directory and re-tokenizes.
# - This produces tokenized columns:
#     sentence_0_input_ids, sentence_0_attention_mask, (maybe token_type_ids)
#     sentence_1_input_ids, sentence_1_attention_mask, (maybe token_type_ids)
#   where sentence_0 = (query, pos_text) and sentence_1 = (query, neg_text).
# -----------------------------------------------------------------------------


# -------------------------
# Offline/prefetch controls
# -------------------------
def set_offline_mode() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def cfg_get(cfg: Dict[str, Any], key: str, default=None):
    v = cfg.get(key, default)
    return default if v in (None, "", "null") else v


# -------------------------
# Dataset loading
# -------------------------
def load_triples_raw(cfg: Dict[str, Any], cache_dir: Optional[str] = None):
    return load_dataset(
        cfg["synthetic_dataset"],
        cfg.get("synthetic_config", "triples"),
        split=cfg.get("synthetic_split", "train"),
        cache_dir=cache_dir,
    )


def process_triples(cfg: Dict[str, Any], ds, num_proc: int):
    needed = [
        cfg.get("col_query", "query"),
        cfg.get("col_pos_text", "pos_text"),
        cfg.get("col_neg_text", "neg_text"),
    ]
    for c in needed:
        if c not in ds.column_names:
            raise ValueError(f"Missing column {c} in triples. Found: {ds.column_names}")

    def _map(ex):
        out = {
            "query": ex[cfg.get("col_query", "query")],
            "pos_text": ex[cfg.get("col_pos_text", "pos_text")],
            "neg_text": ex[cfg.get("col_neg_text", "neg_text")],
        }
        qidc = cfg.get("col_query_id", "query_id")
        if qidc in ex:
            out["query_id"] = ex[qidc]
        pidc = cfg.get("col_pos_doc_id", "pos_doc_id")
        if pidc in ex:
            out["pos_doc_id"] = ex[pidc]
        nidc = cfg.get("col_neg_doc_id", "neg_doc_id")
        if nidc in ex:
            out["neg_doc_id"] = ex[nidc]
        nkc = cfg.get("col_neg_kind", "neg_kind")
        if nkc in ex:
            out["neg_kind"] = ex[nkc]
        return out

    keep = [
        cfg.get("col_query_id", "query_id"),
        cfg.get("col_query", "query"),
        cfg.get("col_pos_doc_id", "pos_doc_id"),
        cfg.get("col_pos_text", "pos_text"),
        cfg.get("col_neg_doc_id", "neg_doc_id"),
        cfg.get("col_neg_text", "neg_text"),
        cfg.get("col_neg_kind", "neg_kind"),
    ]
    remove_cols = [c for c in ds.column_names if c not in keep]
    return ds.map(_map, remove_columns=remove_cols, num_proc=num_proc, desc="process triples")


def load_or_make_triples(cfg: Dict[str, Any], artifacts_dir: str, cache_dir: Optional[str], offline: bool, num_proc: int):
    raw_triples_dir = os.path.join(artifacts_dir, "triples_raw")
    proc_triples_dir = os.path.join(artifacts_dir, "triples_proc")

    if os.path.isdir(proc_triples_dir):
        triples = load_from_disk(proc_triples_dir)
        return triples

    if os.path.isdir(raw_triples_dir):
        raw = load_from_disk(raw_triples_dir)
    else:
        if offline:
            raise ValueError(f"Offline but missing {raw_triples_dir}. Run once online.")
        raw = load_triples_raw(cfg, cache_dir=cache_dir)
        ensure_dir(raw_triples_dir)
        raw.save_to_disk(raw_triples_dir)

    triples = process_triples(cfg, raw, num_proc=num_proc)
    ensure_dir(proc_triples_dir)
    triples.save_to_disk(proc_triples_dir)
    return triples


# -------------------------
# Pre-tokenize (query,pos) and (query,neg)
# -------------------------
def pretokenize_dataset(triples_ds, tokenizer, max_length: int, batch_size: int, num_proc: int):
    def _tok_batch(batch):
        queries = batch["query"]
        pos_texts = batch["pos_text"]
        neg_texts = batch["neg_text"]

        out: Dict[str, Any] = {}

        # sentence_0 := (query, pos_text)
        enc0 = tokenizer(queries, pos_texts, padding=True, truncation=True, max_length=max_length)
        for k, v in enc0.items():
            out[f"sentence_0_{k}"] = v

        # sentence_1 := (query, neg_text)
        enc1 = tokenizer(queries, neg_texts, padding=True, truncation=True, max_length=max_length)
        for k, v in enc1.items():
            out[f"sentence_1_{k}"] = v

        return out

    # Keep metadata if you want it later for debugging / grouping; remove only the huge texts.
    remove_cols = [c for c in triples_ds.column_names if c in ("pos_text", "neg_text", "query")]

    # Note: remove_columns is applied after the mapped function receives the batch. [web:13]
    return triples_ds.map(
        _tok_batch,
        batched=True,
        batch_size=batch_size,
        remove_columns=remove_cols,
        num_proc=num_proc,
        desc="pre-tokenize (q,pos) and (q,neg)",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache_dir", type=str, default=None)
    ap.add_argument("--artifacts_dir", type=str, default=None)

    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--num_proc", type=int, default=8)
    ap.add_argument("--tok_batch_size", type=int, default=256)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip_proc", action="store_true")
    ap.add_argument("--skip_tok", action="store_true")

    args = ap.parse_args()
    cfg = read_yaml(args.config)

    proj = cfg.get("project_dir", "/project/rhino-ffm/reranker_ce")
    run_name = cfg.get("run_name", "run")
    run_dir = os.path.join(proj, "runs", run_name)
    artifacts_dir = args.artifacts_dir or cfg_get(cfg, "artifacts_dir", None) or os.path.join(run_dir, "artifacts")

    if args.offline:
        set_offline_mode()
        print("[mode] OFFLINE=1")

    ensure_dir(run_dir)
    ensure_dir(artifacts_dir)

    proc_triples_dir = os.path.join(artifacts_dir, "triples_proc")
    if args.skip_proc and os.path.isdir(proc_triples_dir):
        triples = load_from_disk(proc_triples_dir)
    else:
        triples = load_or_make_triples(
            cfg,
            artifacts_dir=artifacts_dir,
            cache_dir=args.cache_dir,
            offline=args.offline,
            num_proc=max(1, args.num_proc),
        )

    if args.skip_tok:
        print("[prepare] skip_tok=1; done (processed triples created/loaded).")
        return

    max_length = int(cfg.get("max_length", 384))

    tokenizer_kwargs = dict(cfg.get("tokenizer_kwargs", {}) or {})
    tokenizer_kwargs.setdefault("truncation", True)
    tokenizer_kwargs.setdefault("padding", True)

    # CrossEncoder takes pairs of texts as input. [web:442]
    model = CrossEncoder(
        cfg["model_name"],
        num_labels=1,
        max_length=max_length,
        tokenizer_kwargs=tokenizer_kwargs,
    )

    tok_dir = os.path.join(artifacts_dir, "pretokenized")
    if os.path.isdir(tok_dir) and not args.force:
        print(f"[prepare] tok_dir exists: {tok_dir} (skipping)")
        return

    if os.path.isdir(tok_dir) and args.force:
        import shutil
        shutil.rmtree(tok_dir, ignore_errors=True)

    print("[prepare] pre-tokenizing...")
    tok_ds = pretokenize_dataset(
        triples,
        tokenizer=model.tokenizer,
        max_length=max_length,
        batch_size=int(args.tok_batch_size),
        num_proc=max(1, args.num_proc),
    )

    ensure_dir(tok_dir)
    tok_ds.save_to_disk(tok_dir)
    print(f"[prepare] saved tokenized dataset to {tok_dir}")


if __name__ == "__main__":
    main()
