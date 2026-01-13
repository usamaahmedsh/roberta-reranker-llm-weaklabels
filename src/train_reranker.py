#!/usr/bin/env python3
import os
import re
import time
import json
import argparse
import inspect
from typing import Dict, Any, Optional, List

import yaml
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder import CrossEncoderTrainer, CrossEncoderTrainingArguments
from transformers import TrainerCallback, EarlyStoppingCallback

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# -------------------------
# Offline controls (optional)
# -------------------------
def set_offline_mode() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"


# -------------------------
# Utils
# -------------------------
def cfg_get(cfg: Dict[str, Any], key: str, default=None):
    v = cfg.get(key, default)
    return default if v in (None, "", "null") else v


def read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def find_latest_checkpoint(checkpoints_dir: str) -> Optional[str]:
    if not os.path.isdir(checkpoints_dir):
        return None
    pat = re.compile(r"^checkpoint-(\d+)$")
    best = None
    best_step = -1
    for name in os.listdir(checkpoints_dir):
        m = pat.match(name)
        if not m:
            continue
        step = int(m.group(1))
        if step > best_step:
            best_step = step
            best = os.path.join(checkpoints_dir, name)
    return best


# -------------------------
# Debug helpers
# -------------------------
def _summarize_batch(batch: Dict[str, Any], max_keys: int = 60) -> Dict[str, Any]:
    keys = sorted(list(batch.keys()))
    sample = keys[:max_keys]
    ends_input_ids = [k for k in keys if k.endswith("_input_ids")]

    tensor_shapes = {}
    tensor_dtypes = {}
    tensor_devices = {}
    for k in sample:
        v = batch[k]
        if isinstance(v, torch.Tensor):
            tensor_shapes[k] = list(v.shape)
            tensor_dtypes[k] = str(v.dtype)
            tensor_devices[k] = str(v.device)

    return {
        "num_keys": len(keys),
        "sample_keys": sample,
        "num_endswith__input_ids": len(ends_input_ids),
        "endswith__input_ids_sample": ends_input_ids[:30],
        "tensor_shapes_sample": tensor_shapes,
        "tensor_dtypes_sample": tensor_dtypes,
        "tensor_devices_sample": tensor_devices,
    }


def debug_print_batch(tag: str, batch: Dict[str, Any]) -> None:
    info = _summarize_batch(batch)
    print(f"[debug:{tag}] num_keys={info['num_keys']} num*_input_ids={info['num_endswith__input_ids']}")
    print(f"[debug:{tag}] sample_keys={info['sample_keys']}")
    print(f"[debug:{tag}] endswith__input_ids(sample)={info['endswith__input_ids_sample']}")
    for k in info["endswith__input_ids_sample"][:2]:
        v = batch[k]
        if isinstance(v, torch.Tensor):
            print(f"[debug:{tag}] {k}: shape={tuple(v.shape)} dtype={v.dtype} device={v.device}")


def debug_print_trainer_provenance(trainer) -> None:
    print("[debug:trainer_class]", trainer.__class__)
    print("[debug:trainer_mro]", [c.__name__ for c in trainer.__class__.mro()])

    cf = trainer.collect_features
    print("[debug:collect_features_attr]", cf)
    try:
        print("[debug:collect_features_qualname]", cf.__func__.__qualname__ if hasattr(cf, "__func__") else str(cf))
    except Exception as e:
        print("[debug:collect_features_qualname] failed:", repr(e))

    try:
        print("[debug:collect_features_sourcefile]", inspect.getsourcefile(cf))
    except Exception as e:
        print("[debug:collect_features_sourcefile] failed:", repr(e))


def reconstruct_feature_dicts_from_inputs(inputs: Dict[str, Any]) -> List[Dict[str, torch.Tensor]]:
    prefixes = []
    for k in inputs.keys():
        if k.endswith("_input_ids") and k.startswith("sentence_"):
            prefixes.append(k[: -len("_input_ids")])
    prefixes = sorted(set(prefixes), key=lambda s: int(s.split("_")[1]) if s.split("_")[1].isdigit() else s)

    feats: List[Dict[str, torch.Tensor]] = []
    for p in prefixes:
        d: Dict[str, torch.Tensor] = {}
        for k, v in inputs.items():
            if not isinstance(v, torch.Tensor):
                continue
            if k.startswith(p + "_"):
                suffix = k[len(p) + 1 :]   # e.g. "input_ids", "attention_mask"
                if suffix in ("input_ids", "attention_mask", "token_type_ids", "inputs_embeds"):
                    d[suffix] = v
        feats.append(d)

    return feats


# -------------------------
# Throughput callback
# -------------------------
class ThroughputCallback(TrainerCallback):
    def __init__(self, metrics_path: str, max_length: int, train_batch_size: int, grad_accum: int):
        self.metrics_path = metrics_path
        self.max_length = max_length
        self.train_batch_size = train_batch_size
        self.grad_accum = grad_accum
        self._last_t = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._last_t = None
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return control
        t = time.time()
        if self._last_t is None:
            self._last_t = t
            return control
        dt = t - self._last_t
        self._last_t = t

        tokens_per_step_est = self.train_batch_size * self.max_length * self.grad_accum
        tokens_per_sec_est = (tokens_per_step_est / dt) if dt > 0 else None

        rec = {
            "time": now_ts(),
            "global_step": int(getattr(state, "global_step", -1)),
            "dt_sec": dt,
            "tokens_per_step_est": tokens_per_step_est,
            "tokens_per_sec_est": tokens_per_sec_est,
            "logs": logs,
        }
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return control


# -------------------------
# Fast collator (FIXED for sentence_* token columns)
# -------------------------
from torch.nn.utils.rnn import pad_sequence  # Pads variable-length tensors to equal length. [web:41]

class PreTokenizedCollator:
    # Label column must be one of these names for ST to auto-detect it. [web:5]
    valid_label_columns = ["label", "labels", "score", "scores"]

    def __init__(
        self,
        pad_token_id: int = 0,
        label_pad_token_id: int = -100,
        debug: bool = False,
    ):
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id
        self.debug = debug
        self._calls = 0

    def __call__(self, batch):
        self._calls += 1
        out: Dict[str, Any] = {}

        def _as_1d_tensor(x, *, dtype=None):
            t = x if isinstance(x, torch.Tensor) else torch.tensor(x)
            if dtype is not None:
                t = t.to(dtype)
            return t.view(-1)

        def _pad_1d(key, pad_value, *, dtype=None):
            seqs = [_as_1d_tensor(b[key], dtype=dtype) for b in batch]
            return pad_sequence(seqs, batch_first=True, padding_value=pad_value)  # [web:41]

        # ---- FIX 1: pad ALL token columns, including sentence_0_* / sentence_1_* ----
        for k in batch[0].keys():
            if k.endswith("_input_ids"):
                out[k] = _pad_1d(k, self.pad_token_id, dtype=torch.long)
            elif k.endswith("_attention_mask"):
                out[k] = _pad_1d(k, 0, dtype=torch.long)
            elif k.endswith("_token_type_ids"):
                out[k] = _pad_1d(k, 0, dtype=torch.long)

        # ---- FIX 2: normalize label into 'label' (ST examples/use often rely on this) ----
        if "label" in batch[0]:
            out["label"] = torch.stack(
                [b["label"] if isinstance(b["label"], torch.Tensor) else torch.tensor(b["label"]) for b in batch],
                dim=0,
            )
        elif "labels" in batch[0]:
            out["label"] = torch.stack(
                [b["labels"] if isinstance(b["labels"], torch.Tensor) else torch.tensor(b["labels"]) for b in batch],
                dim=0,
            )
        elif "score" in batch[0]:
            out["label"] = torch.stack(
                [b["score"] if isinstance(b["score"], torch.Tensor) else torch.tensor(b["score"]) for b in batch],
                dim=0,
            )
        elif "scores" in batch[0]:
            out["label"] = torch.stack(
                [b["scores"] if isinstance(b["scores"], torch.Tensor) else torch.tensor(b["scores"]) for b in batch],
                dim=0,
            )

        # Keep everything else (ids/metadata) as lists to avoid accidental shape assumptions
        for k in batch[0].keys():
            if k in out or k in ("labels", "scores"):  # 'labels/scores' normalized into 'label'
                continue
            v0 = batch[0][k]
            if isinstance(v0, torch.Tensor):
                try:
                    out[k] = torch.stack([b[k] for b in batch], dim=0)
                except RuntimeError:
                    out[k] = [b[k] for b in batch]
            else:
                out[k] = [b[k] for b in batch]

        if self.debug and self._calls <= 2:
            print("[collator] keys:", sorted(out.keys())[:80])
            for kk in sorted([x for x in out.keys() if x.endswith("_input_ids")])[:4]:
                vv = out[kk]
                if isinstance(vv, torch.Tensor):
                    print(f"[collator] {kk}: {tuple(vv.shape)} dtype={vv.dtype}")

        return out


# -------------------------
# Pairwise margin loss
# -------------------------
class PairwiseMarginLoss(torch.nn.Module):
    def __init__(self, ce: CrossEncoder, margin: float = 1.0, debug: bool = False):
        super().__init__()
        self.ce = ce
        self.margin = margin
        self.debug = debug
        self._calls = 0

    def forward(self, features, labels=None):
        self._calls += 1
        if self.debug and self._calls <= 2:
            print(
                f"[debug:loss] call={self._calls} features_type={type(features)} "
                f"len={len(features) if hasattr(features,'__len__') else 'NA'}"
            )

        if not isinstance(features, (list, tuple)) or len(features) == 0:
            raise TypeError(f"Expected non-empty list/tuple of features, got {type(features)}")

        if isinstance(features[0], torch.Tensor):
            raise TypeError("Trainer produced features as list[Tensor]; expected list[dict].")

        feats = [d for d in features if isinstance(d, dict)]
        if len(feats) < 2:
            raise TypeError(f"Expected >=2 feature dicts (pos+neg), got {len(feats)}")

        scores: List[torch.Tensor] = []
        for feat in feats:
            feat = {k: v.to(self.ce.model.device, non_blocking=True) for k, v in feat.items()}
            out = self.ce.model(**feat)
            scores.append(out.logits.squeeze(-1))

        logits = torch.stack(scores, dim=1)  # [B, num_pairs]
        s_pos = logits[:, 0]
        s_negs = logits[:, 1:]
        loss_mat = F.relu(self.margin - s_pos.unsqueeze(1) + s_negs)
        return loss_mat.mean()


# -------------------------
# Debug + fallback trainer
# -------------------------
class DebugFallbackCrossEncoderTrainer(CrossEncoderTrainer):
    def __init__(self, *args, debug: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._debug = debug
        self._compute_loss_calls = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        self._compute_loss_calls += 1
        if self._debug and self._compute_loss_calls <= 3 and isinstance(inputs, dict):
            print(f"[debug:trainer.compute_loss] call={self._compute_loss_calls}")
            debug_print_batch("inputs_to_compute_loss", inputs)

        features, labels = self.collect_features(inputs)

        need_fallback = (
            isinstance(features, (list, tuple))
            and len(features) > 0
            and not isinstance(features[0], dict)
        )
        if need_fallback:
            if self._debug and self._compute_loss_calls <= 3:
                print("[debug:trainer.compute_loss] FALLBACK: reconstructing list[dict] features from inputs keys")
            features = reconstruct_feature_dicts_from_inputs(inputs)

        loss = self.loss(features, labels)
        if return_outputs:
            return loss, {}
        return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--pretokenized_dir", type=str, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke_steps", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    cfg = read_yaml(args.config)

    proj = cfg.get("project_dir", "/project/rhino-ffm/reranker_ce")
    run_name = cfg.get("run_name", "run")
    run_dir = os.path.join(proj, "runs", run_name)

    if args.offline:
        set_offline_mode()
        print("[mode] OFFLINE=1")

    tb_dir = os.path.join(run_dir, "tb")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    final_dir = os.path.join(run_dir, "final_model")
    metrics_path = os.path.join(run_dir, "metrics.jsonl")

    ensure_dir(run_dir)
    ensure_dir(tb_dir)
    ensure_dir(ckpt_dir)
    ensure_dir(final_dir)

    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)

    # -------------------------
    # Load pretokenized dataset
    # -------------------------
    pretokenized_dir = (
        args.pretokenized_dir
        or cfg_get(cfg, "pretokenized_dir", None)
        or os.path.join(run_dir, "artifacts", "pretokenized")
    )
    if not os.path.isdir(pretokenized_dir):
        raise FileNotFoundError(
            f"pretokenized_dir not found: {pretokenized_dir}\n"
            f"Run: python src/prepare_artifacts.py --config {args.config} ..."
        )

    print(f"[data] loading pretokenized dataset: {pretokenized_dir}")
    tok_ds = load_from_disk(pretokenized_dir)

    # Keep all columns; they include sentence_0_input_ids etc.
    tok_ds.set_format(type="torch")

    if args.smoke:
        tok_ds = tok_ds.select(range(min(len(tok_ds), 2000)))

    # Split AFTER tokenization (fast)
    dev_ratio = float(cfg.get("dev_ratio", 0.01))
    seed = int(cfg.get("seed", 42))
    if dev_ratio > 0:
        d = tok_ds.train_test_split(test_size=dev_ratio, seed=seed)
        train_hf, dev_hf = d["train"], d["test"]
    else:
        train_hf, dev_hf = tok_ds, None

    effective_max_steps = args.max_steps
    if args.smoke and effective_max_steps is None:
        effective_max_steps = args.smoke_steps

    logging_steps = int(cfg.get("logging_steps", 50))
    eval_steps = int(cfg.get("eval_steps", 1000))
    save_steps = int(cfg.get("save_steps", 1000))
    if args.smoke:
        logging_steps = 1
        mid = max(1, int(effective_max_steps or 10) // 2)
        eval_steps = mid
        save_steps = mid

    # -------------------------
    # Model
    # -------------------------
    tokenizer_kwargs = dict(cfg.get("tokenizer_kwargs", {}) or {})
    tokenizer_kwargs.setdefault("truncation", True)
    tokenizer_kwargs.setdefault("padding", True)

    attn_impl = cfg_get(cfg, "attn_implementation", "sdpa")

    model = CrossEncoder(
        cfg["model_name"],
        num_labels=1,
        max_length=int(cfg.get("max_length", 384)),
        tokenizer_kwargs=tokenizer_kwargs,
        model_kwargs={"attn_implementation": attn_impl},
    )

    # -------------------------
    # Optimizer/scheduler + regularization + best model + early stopping
    # -------------------------
    optim = cfg_get(cfg, "optim", None)
    lr_scheduler_type = cfg_get(cfg, "lr_scheduler_type", None)

    weight_decay = float(cfg_get(cfg, "weight_decay", 0.01))
    adam_beta1 = float(cfg_get(cfg, "adam_beta1", 0.9))
    adam_beta2 = float(cfg_get(cfg, "adam_beta2", 0.999))
    adam_epsilon = float(cfg_get(cfg, "adam_epsilon", 1e-8))
    max_grad_norm = float(cfg_get(cfg, "max_grad_norm", 1.0))

    load_best_model_at_end = bool(cfg_get(cfg, "load_best_model_at_end", True))
    metric_for_best_model = str(cfg_get(cfg, "metric_for_best_model", "eval_loss"))
    greater_is_better = bool(cfg_get(cfg, "greater_is_better", False))

    early_stopping_patience = int(cfg_get(cfg, "early_stopping_patience", 0))
    early_stopping_threshold = float(cfg_get(cfg, "early_stopping_threshold", 0.0))

    num_workers = int(cfg.get("num_workers", 4))
    pin_memory = bool(cfg_get(cfg, "pin_memory", True))
    persistent_workers = bool(cfg_get(cfg, "persistent_workers", True))

    if early_stopping_patience > 0 and (dev_hf is None):
        raise ValueError("early_stopping_patience > 0 requires a dev split (set dev_ratio>0).")

    if early_stopping_patience > 0 and save_steps != eval_steps:
        print(f"[warn] early stopping enabled but save_steps({save_steps}) != eval_steps({eval_steps}); "
              f"stopping can be delayed until next save step.")

    st_args = CrossEncoderTrainingArguments(
        output_dir=ckpt_dir,
        num_train_epochs=float(cfg.get("num_epochs", 1)),
        max_steps=int(effective_max_steps) if effective_max_steps is not None else -1,
        learning_rate=float(cfg.get("lr", 2e-5)),

        per_device_train_batch_size=int(cfg.get("train_batch_size", 16)),
        per_device_eval_batch_size=int(cfg.get("eval_batch_size", 16)),
        gradient_accumulation_steps=int(cfg.get("grad_accum", 1)),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.05)),

        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=int(cfg.get("save_total_limit", 3)),
        eval_strategy="steps" if dev_hf is not None else "no",
        eval_steps=eval_steps,
        logging_dir=tb_dir,
        report_to=["tensorboard"],

        bf16=bool(cfg.get("bf16", False)),
        fp16=bool(cfg.get("fp16", False)),

        dataloader_num_workers=num_workers,
        dataloader_pin_memory=pin_memory,
        dataloader_persistent_workers=persistent_workers,
        remove_unused_columns=False,

        weight_decay=weight_decay,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_epsilon=adam_epsilon,
        max_grad_norm=max_grad_norm,

        load_best_model_at_end=bool(load_best_model_at_end and dev_hf is not None),
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,

        **({"optim": optim} if optim is not None else {}),
        **({"lr_scheduler_type": lr_scheduler_type} if lr_scheduler_type is not None else {}),
    )

    loss = PairwiseMarginLoss(model, margin=float(cfg.get("margin", 1.0)), debug=args.debug)
    data_collator = PreTokenizedCollator(debug=args.debug)

    trainer = DebugFallbackCrossEncoderTrainer(
        model=model,
        args=st_args,
        train_dataset=train_hf,
        eval_dataset=dev_hf,
        data_collator=data_collator,
        loss=loss,
        debug=args.debug,
    )

    trainer.add_callback(
        ThroughputCallback(
            metrics_path=metrics_path,
            max_length=int(cfg.get("max_length", 384)),
            train_batch_size=int(cfg.get("train_batch_size", 16)),
            grad_accum=int(cfg.get("grad_accum", 1)),
        )
    )

    if dev_hf is not None and early_stopping_patience > 0:
        trainer.add_callback(
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=early_stopping_threshold,
            )
        )

    if args.debug:
        debug_print_trainer_provenance(trainer)
        b0 = next(iter(trainer.get_train_dataloader()))
        print("[debug:dataloader] first batch from get_train_dataloader()")
        debug_print_batch("train_dataloader_batch0", b0)

    resume_from = None
    if args.resume:
        resume_from = find_latest_checkpoint(ckpt_dir)
        print(f"[resume] {resume_from}")

    trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_model(final_dir)
    print(f"[done] final model in {final_dir}")


if __name__ == "__main__":
    main()
