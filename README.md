
# RoBERTa-base Cross-Encoder Reranker (LLM Weak Labels) v1

A RoBERTa-base **cross-encoder** reranker trained with weak/synthetic labels, intended for re-ranking a candidate set of passages/documents for a query. Cross-encoders score *(query, document)* pairs jointly and output a single relevance score. [web:54]

## Model
- **HF model repo:** `usamaahmedsh/roberta-base-reranker-llm-weaklabels-v1`
- **Architecture:** RoBERTa-base sequence classification head (1 score per pair)
- **Task:** passage/document reranking (query–text relevance scoring)

## Repository layout
Typical structure (adjust to your repo):
- `src/` Training + evaluation scripts.
- `configs/` YAML configs for runs.
- `runs/` Outputs (checkpoints, logs) *(usually gitignored)*.
- `requirements.txt` or `environment.yml`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

If you use GPU:


```bash
python -c "import torch; print(torch.cuda.is_available())"
Inference (scoring pairs)
python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_id = "usamaahmedsh/roberta-base-reranker-llm-weaklabels-v1"

tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
mdl = AutoModelForSequenceClassification.from_pretrained(model_id).eval()

pairs = [
    ("what is aurora kinase", "Aurora kinases are a family of serine/threonine kinases..."),
    ("what is aurora kinase", "Seattle is a city in Washington..."),
]

q = [p for p in pairs]
d = [p for p in pairs][1]

enc = tok(q, d, padding=True, truncation=True, max_length=384, return_tensors="pt")
with torch.no_grad():
    logits = mdl(**enc).logits
scores = logits.view(-1).tolist()

print(scores)  # higher = more relevant
```

## Evaluation (MS MARCO reranking)
Evaluation follows a common reranking setup: take an initial candidate set (e.g., BM25 top-1000), score each (query, passage) pair with the cross-encoder, sort by score, and compute ranking metrics like MRR@10 / nDCG@k / Recall@k. MS MARCO passage ranking is commonly reported with MRR@10. [web:39]

Example:
```bash
python eval_comparison.py \
  --my_model /path/to/final_model \
  --out_dir msmarco_ce_eval \
  --topk_rerank 1000 \
  --batch_size 128 \
  --max_length 384
```
## Training
Training is implemented with sentence-transformers cross-encoder utilities (trainer/args) and a ranking-style objective (pairwise/groupwise depending on the run configuration). The Sentence-Transformers reranker training approach is described in the Hugging Face reranker training guide. [web:54]

Example:

```bash
python train_cross_encoder.py --config configs/train.yaml
```
## Notes on artifacts
Do not commit large checkpoints or model weight files to GitHub. Push model weights to Hugging Face Hub and keep GitHub for code/configs; use .gitignore for runs/**/checkpoints/ and runs/**/final_model/.

## License
Add a license that matches your intended usage (e.g., MIT/Apache-2.0) and ensure it is compatible with any upstream model/data constraints.
