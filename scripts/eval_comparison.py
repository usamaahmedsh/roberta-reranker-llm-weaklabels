#!/usr/bin/env python3
import os
import json
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import pytrec_eval
import ir_datasets

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_BASELINES_NON_MSMARCO = [
    "cross-encoder/stsb-roberta-base",
    "cross-encoder/stsb-roberta-large",
]



def load_msmarco_dev():
    # dev scoreddocs are BM25 top-1000 for re-ranking in MS MARCO. [web:137]
    dataset = ir_datasets.load("msmarco-passage/dev/small")

    queries = {str(q.query_id): q.text for q in dataset.queries_iter()}

    qrels = defaultdict(dict)
    for qr in dataset.qrels_iter():
        qrels[str(qr.query_id)][str(qr.doc_id)] = int(qr.relevance)

    # Candidate pool: scoreddocs_iter yields top-1000 docs per query in this setting. [web:137]
    candidates = defaultdict(list)
    for sd in dataset.scoreddocs_iter():
        candidates[str(sd.query_id)].append(str(sd.doc_id))

    # Docstore for the full msmarco passage collection
    docs = ir_datasets.load("msmarco-passage").docs_store()
    def get_doc_text(docid: str) -> str:
        return docs.get(docid).text

    return queries, dict(qrels), dict(candidates), get_doc_text


@torch.no_grad()
def score_pairs(model, tokenizer, pairs, batch_size=128, max_length=384, device="cuda"):
    model.eval()
    scores = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i+batch_size]
        q = [x[0] for x in batch]
        d = [x[1] for x in batch]
        enc = tokenizer(
            q, d,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        out = model(**enc)
        logits = out.logits
        # Common convention: if 2 labels, take "positive" logit; if 1, take scalar. [web:3]
        if logits.shape[-1] == 1:
            s = logits[:, 0]
        else:
            s = logits[:, 1]
        scores.append(s.detach().float().cpu().numpy())
    return np.concatenate(scores, axis=0)


def rerank(queries, candidates, get_doc_text, model, tokenizer,
           topk_rerank=1000, batch_size=128, max_length=384, device="cuda"):
    """
    Returns run dict: run[qid][docid] = score
    """
    run = {}
    for qid, docids in candidates.items():
        if qid not in queries:
            continue
        docids_k = docids[:topk_rerank]
        qtext = queries[qid]
        pairs = [(qtext, get_doc_text(did)) for did in docids_k]
        s = score_pairs(model, tokenizer, pairs, batch_size=batch_size, max_length=max_length, device=device)
        run[qid] = {did: float(score) for did, score in zip(docids_k, s)}
    return run


def truncate_run(run, k):
    out = {}
    for qid, doc2score in run.items():
        items = sorted(doc2score.items(), key=lambda x: x[1], reverse=True)[:k]
        out[qid] = dict(items)
    return out


def eval_metrics(qrels, run, ks=(10, 15, 20)):
    # pytrec_eval supports cut measures like ndcg_cut.K and recall.K. [web:330][web:327]
    measures = set()
    for k in ks:
        measures.add(f"ndcg_cut.{k}")
        measures.add(f"recall.{k}")

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, measures)
    perq = evaluator.evaluate(run)

    out = {}
    for k in ks:
        out[f"nDCG@{k}"] = float(np.mean([v.get(f"ndcg_cut_{k}", 0.0) for v in perq.values()]))
        out[f"Recall@{k}"] = float(np.mean([v.get(f"recall_{k}", 0.0) for v in perq.values()]))

    # MRR@K via truncation then recip_rank (RR). [web:327]
    for k in ks:
        evaluator_rr = pytrec_eval.RelevanceEvaluator(qrels, {"recip_rank"})
        perq_rr = evaluator_rr.evaluate(truncate_run(run, k))
        out[f"MRR@{k}"] = float(np.mean([v.get("recip_rank", 0.0) for v in perq_rr.values()]))

    return out


def plot_metric_bars(df, metric, out_png):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    tmp = df[["model", metric]].sort_values(metric, ascending=False)
    plt.figure(figsize=(10, max(3, 0.45 * len(tmp))))
    plt.barh(tmp["model"], tmp[metric])
    plt.gca().invert_yaxis()
    plt.xlabel(metric)
    plt.title(f"MS MARCO dev/judged rerank: {metric}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--my_model", required=True, help="Path/HF id of your trained model (e.g., .../final_model)")
    ap.add_argument("--baselines", nargs="*", default=DEFAULT_BASELINES_NON_MSMARCO,
                    help="HF ids for baseline cross-encoders (avoid msmarco-trained).")
    ap.add_argument("--topk_rerank", type=int, default=1000)
    ap.add_argument("--max_length", type=int, default=384)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default="msmarco_ce_eval")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("[data] loading MS MARCO dev/judged from ir_datasets")
    queries, qrels, candidates, get_doc_text = load_msmarco_dev()

    models = [("my_model", args.my_model)] + [(m.split("/")[-1], m) for m in args.baselines]
    rows = []

    for short, model_id in models:
        print(f"[model] loading {short}: {model_id}")
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        mdl = AutoModelForSequenceClassification.from_pretrained(model_id).to(args.device)

        print(f"[rerank] topk={args.topk_rerank} bs={args.batch_size} maxlen={args.max_length}")
        run = rerank(
            queries=queries,
            candidates=candidates,
            get_doc_text=get_doc_text,
            model=mdl,
            tokenizer=tok,
            topk_rerank=args.topk_rerank,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device,
        )

        metrics = eval_metrics(qrels, run, ks=(10, 15, 20))
        metrics["model"] = short
        metrics["model_id"] = model_id
        metrics["topk_rerank"] = args.topk_rerank
        rows.append(metrics)

        with open(os.path.join(args.out_dir, f"metrics_{short}.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    df = pd.DataFrame(rows).sort_values("MRR@10", ascending=False)
    df.to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)

    for metric in ["MRR@10","MRR@15","MRR@20","Recall@10","Recall@15","Recall@20","nDCG@10","nDCG@15","nDCG@20"]:
        plot_metric_bars(df, metric, os.path.join(args.out_dir, "plots", f"{metric}.png"))

    print(f"[done] wrote {os.path.join(args.out_dir, 'summary.csv')}")
    print(f"[done] plots in {os.path.join(args.out_dir, 'plots')}")


if __name__ == "__main__":
    main()
