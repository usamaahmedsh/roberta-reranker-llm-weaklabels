#!/usr/bin/env python3
"""
Push a RoBERTa cross-encoder reranker to Hugging Face Hub.

Repo: usamaahmedsh/roberta-base-reranker-llm-weaklabels-v1

Usage:
  export HF_TOKEN=hf_...   # preferred
  python push_to_hf.py --local_model_dir /path/to/final_model --private 0
"""

import os
import argparse
from pathlib import Path

from huggingface_hub import HfApi
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_model_dir", required=True, help="Path to your saved model folder (final_model/)")
    ap.add_argument("--repo_id", default="usamaahmedsh/roberta-base-reranker-llm-weaklabels-v1")
    ap.add_argument("--private", type=int, default=0, help="1 = private repo, 0 = public repo")
    ap.add_argument("--commit_message", default="Initial upload")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError(
            "Missing HF token. Set HF_TOKEN (recommended) or HUGGINGFACE_TOKEN in the environment."
        )

    local_dir = Path(args.local_model_dir)
    if not local_dir.exists():
        raise FileNotFoundError(f"local_model_dir not found: {local_dir}")

    # Quick sanity-check: folder must be loadable by Transformers
    tok = AutoTokenizer.from_pretrained(str(local_dir), use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(str(local_dir))

    # Ensure the common files are written to disk (in case training script didn't save them all)
    tok.save_pretrained(str(local_dir))
    mdl.save_pretrained(str(local_dir), safe_serialization=True)

    api = HfApi()

    # Create repo if it doesn't exist
    api.create_repo(
        repo_id=args.repo_id,
        token=token,
        private=bool(args.private),
        exist_ok=True,
        repo_type="model",
    )

    # Upload entire folder (config, tokenizer files, safetensors/bin, etc.)
    api.upload_folder(
        repo_id=args.repo_id,
        folder_path=str(local_dir),
        path_in_repo="",
        commit_message=args.commit_message,
        token=token,
        repo_type="model",
    )

    print(f"[done] Uploaded to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
