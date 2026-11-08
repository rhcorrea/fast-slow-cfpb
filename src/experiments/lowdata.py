"""Laya on the test set: zero-shot, or fine-tuned on n complaints per class, asked product
alone or product + issue type. One GPU job at a time on 16 GB, so run these one after another:

    python -m src.experiments.lowdata 0 0 0 multi     # zero-shot, both questions
    python -m src.experiments.lowdata 10 10 2         # N EPOCHS GRAD_ACCUM [multi]
    python -m src.experiments.lowdata 50 4 4
    python -m src.experiments.lowdata 100 3 4
    python -m src.experiments.lowdata 500 1 8         # the first pilot's settings
    python -m src.experiments.lowdata 50 4 4 multi

Each run appends a line to data/processed/laya_curve.jsonl, writes test_{name}.parquet and,
when it trains, a 1.6 GB checkpoint to models/{name}.
"""
import os

# Cap the Apple GPU allocator so an out-of-memory run fails instead of hanging the Mac. The low
# ratio must be set too, or PyTorch rejects the pair. Must happen before torch is imported.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.75")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6")

import json
import sys
import time

import torch
import laya.train as LT
from laya import load
from laya.train import TrainConfig, finetune

from ..models import system1
from .common import MULTI, P, RESULTS, issue_score, issue_type, product_scores, sample, test_frame, train_meta, with_text

# Full fp32 fine-tuning (421M params: ~6.4 GB of weights, grads and AdamW state) doesn't fit, so
# train the heads and the top K encoder layers only. finetune() loads through
# laya.train.load_checkpoint and only optimizes parameters that still require grad.
K = 8
# MPS keeps compiled work per input shape and collate_items pads each batch to its own longest
# row, so memory outside the tensors grows until the cap. Pad to multiples of 128 tokens
# (<= 4 shapes at 512) and release the allocator cache every 25 batches.
BUCKET = 128
_load, _collate, _calls = LT.load_checkpoint, LT.collate_items, [0]


def load_partly_frozen(model_dir):
    model, tok, cfg = _load(model_dir)
    for mod in (model.encoder.embeddings, *model.encoder.layers[:-K]):
        for p in mod.parameters():
            p.requires_grad_(False)
    return model, tok, cfg


def collate_bucketed(batch, pad_id):
    res = _collate(batch, pad_id)
    pad = -res["input_ids"].shape[1] % BUCKET
    if pad:
        res["input_ids"] = torch.nn.functional.pad(res["input_ids"], (0, pad), value=pad_id)
        for k in ("attention_mask", "position_ids", "option_ids"):
            if k in res:
                res[k] = torch.nn.functional.pad(res[k], (0, pad), value=0)
    _calls[0] += 1
    if _calls[0] % 25 == 0:
        torch.mps.empty_cache()
    return res


def main(n: int, epochs: int, grad_accum: int, multi: bool) -> dict:
    qs = MULTI if multi else system1.QUESTIONS
    name = f"laya-{'zs' if n == 0 else f'ft-n{n}'}{'-multi' if multi else ''}"
    tr, shares = train_meta()
    rec = {"model": name, "n": n, "epochs": epochs, "grad_accum": grad_accum, "multi": multi}
    model_dir = system1.MODEL
    if n:
        LT.load_checkpoint, LT.collate_items = load_partly_frozen, collate_bucketed
        s = with_text(sample(tr, n, 0))
        jsonl = P / f"{name}_train.jsonl"
        with jsonl.open("w") as f:
            for text, y, iss in zip(s["narrative"], s["canonical_product"], s["issue"]):
                expected = {"product": y}
                if multi and issue_type(iss):
                    expected["issue_type"] = issue_type(iss)
                f.write(json.dumps({"state": text, "questions": qs, "expected": expected}) + "\n")
        model_dir = str(system1.MODELS_DIR / name)
        t0 = time.perf_counter()
        cfg = TrainConfig(epochs=epochs, micro_batch=8, grad_accum=grad_accum, loss="soft-ce",
                          gradient_checkpointing=True, log_every=50)
        finetune(str(jsonl), system1.MODEL, model_dir, cfg, device="mps")
        rec["train_min"] = round((time.perf_counter() - t0) / 60, 1)

    agent = load(model_dir)
    te = test_frame()
    t0 = time.perf_counter()
    ans = [agent.system_one(text, qs)["answers"] for text in te["narrative"]]
    rec["s_per_complaint"] = round((time.perf_counter() - t0) / len(te), 3)
    prod = [a["product"] for a in ans]
    rec.update(product_scores(te["canonical_product"], [a["probabilities"] for a in prod], shares))
    te["pred"] = [a["choice"] for a in prod]
    te["confidence"] = [a["answer_confidence"] for a in prod]  # Laya's 'confidence' is entropy, not calibrated
    if multi:
        te["issue_pred"] = [a["issue_type"]["choice"] for a in ans]
        rec["issue_type"] = issue_score(te["issue_type"], te["issue_pred"])
    te.drop(columns=["narrative"]).to_parquet(P / f"test_{name}.parquet", index=False)
    with RESULTS.open("a") as out:
        print(json.dumps(rec), file=out)
    return rec


if __name__ == "__main__":
    a = sys.argv[1:]
    print(main(int(a[0]), int(a[1]), int(a[2]), len(a) > 3 and a[3] == "multi"))
