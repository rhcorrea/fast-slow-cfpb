"""Zero-shot classification with a quantized general small LM (MLX, Apple Silicon).

The prompt numbers the canonical labels 0-9 and asks for the number only. Instead of
generating, one forward pass reads the next-token logits for the ten digits and turns
them into per-label probabilities -- the same choice/confidence/probabilities shape
Laya returns (system1.classify), so the router and eval can use it either side.

Runs are resumable: each answer is appended to data/processed/{split}_{model}.jsonl.

    python -m src.models.slm qwen3.5-4b-4bit 100   # pilot: 10 per class from val
    python -m src.models.slm qwen3.5-4b-4bit       # full val + test
"""
import json
import sys
import time
from typing import NamedTuple

import mlx.core as mx
import pandas as pd
from mlx_lm import load
from mlx_lm.models.cache import ArraysCache, KVCache, make_prompt_cache

from ..data.taxonomy import LABELS
from .system1 import CRITERIA, MODELS_DIR, PROCESSED_DIR

assert len(LABELS) <= 10, "labels are answered as single digits"

INSTRUCTIONS = (
    "You classify CFPB consumer complaints by the financial product they concern. Categories:\n"
    + "\n".join(f"{i}. {label}: {CRITERIA[label]}" for i, label in enumerate(LABELS))
    + "\n\nReply with the category number only."
)
# ponytail: crude ~2k-token cap so a few very long narratives don't dominate run time;
# switch to token-based truncation if accuracy on long complaints suffers.
MAX_CHARS = 8000
# Every call has a different input length, so MLX's buffer cache keeps growing unless
# capped; on a 16 GB Mac that once hung the machine. Cap it, and stop the run rather
# than the computer if MLX memory still climbs past MEMORY_LIMIT.
mx.set_cache_limit(512 * 2**20)
MEMORY_LIMIT = 8 * 2**30


SLOT = "\x00"  # stands in for the narrative while splitting the chat template


class SLM(NamedTuple):
    model: object
    tokenizer: object
    digits: list[int]
    prefix: list  # model cache after the shared system prompt
    tail: str     # template text after the narrative


def prompt(tokenizer, narrative: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": INSTRUCTIONS}, {"role": "user", "content": narrative}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def fork(cache: list) -> list:
    """Copy of a prompt cache that the next forward pass can extend without touching the original.

    Qwen3.5 mixes attention layers (KVCache, written in place, so slice to new arrays)
    with linear-attention layers (ArraysCache, whose entries are replaced, never written
    into, so a new list is enough). Linear-attention state can't be trimmed back, hence
    a copy per complaint rather than trim_prompt_cache.
    """
    out = []
    for c in cache:
        if isinstance(c, KVCache):
            k, v, n = c.state
            out.append(KVCache.from_state((k[..., :n, :], v[..., :n, :], n)))
        else:
            assert isinstance(c, ArraysCache), f"unhandled cache type {type(c).__name__}"
            out.append(ArraysCache.from_state((list(c.cache), c.left_padding, c.lengths)))
    return out


def load_model(name: str) -> SLM:
    """Model, tokenizer, digit ids and the system prompt already run, for a folder under models/."""
    model, tokenizer = load(str(MODELS_DIR / name))
    digits = [tokenizer.encode(str(i), add_special_tokens=False) for i in range(len(LABELS))]
    assert all(len(d) == 1 for d in digits), "each digit must be a single token"
    # The instructions are identical on every call: run them once and reuse the cache.
    head, tail = prompt(tokenizer, SLOT).split(SLOT)
    prefix = make_prompt_cache(model)
    model(mx.array([tokenizer.encode(head, add_special_tokens=False)]), cache=prefix)
    mx.eval([a for c in prefix for a in (c.state[:2] if isinstance(c, KVCache) else c.cache)])
    return SLM(model, tokenizer, [d[0] for d in digits], prefix, tail)


def classify(slm: SLM, narrative: str) -> dict:
    model, tokenizer, digits, prefix, tail = slm
    # .strip() because the chat template strips message content; skipping it changes
    # the tokens for narratives cut at MAX_CHARS mid-whitespace.
    ids = tokenizer.encode(narrative[:MAX_CHARS].strip() + tail, add_special_tokens=False)
    probs = mx.softmax(model(mx.array([ids]), cache=fork(prefix))[0, -1, digits].astype(mx.float32)).tolist()
    best = max(range(len(LABELS)), key=probs.__getitem__)
    return {"choice": LABELS[best], "confidence": probs[best], "probabilities": dict(zip(LABELS, probs))}


def run_split(split: str, model_name: str, n: int | None = None) -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / f"{split}.parquet", columns=["complaint_id", "narrative", "canonical_product"])
    if n is not None:
        per_class = max(1, n // len(LABELS))
        df = pd.concat([g.sample(min(len(g), per_class), random_state=42) for _, g in df.groupby("canonical_product")],
                       ignore_index=True)

    tag = f"{split}_{model_name}{'_pilot' if n else ''}"
    log = PROCESSED_DIR / f"{tag}.jsonl"
    done = {}
    if log.exists():
        for line in log.read_text().splitlines():
            row = json.loads(line)
            done[row["complaint_id"]] = row

    slm = load_model(model_name)
    todo = df[~df["complaint_id"].isin(done)]
    start = time.perf_counter()
    with log.open("a") as f:
        for cid, text in zip(todo["complaint_id"], todo["narrative"]):
            t0 = time.perf_counter()
            row = {"complaint_id": cid, **classify(slm, text)}
            mem = mx.get_active_memory() + mx.get_cache_memory()
            row |= {"secs": round(time.perf_counter() - t0, 3), "mlx_gb": round(mem / 2**30, 2)}
            f.write(json.dumps(row) + "\n")
            f.flush()
            done[cid] = row
            if mem > MEMORY_LIMIT:
                sys.exit(f"stopping: MLX memory at {mem / 2**30:.1f} GB (limit {MEMORY_LIMIT / 2**30:.0f} GB); rerun to resume")
    secs = (time.perf_counter() - start) / max(len(todo), 1)

    df["slm_pred"] = df["complaint_id"].map(lambda c: done[c]["choice"])
    df["slm_confidence"] = df["complaint_id"].map(lambda c: done[c]["confidence"])
    df["slm_probabilities"] = df["complaint_id"].map(lambda c: done[c]["probabilities"])
    df.drop(columns="narrative").to_parquet(PROCESSED_DIR / f"{tag}.parquet", index=False)

    acc = (df["slm_pred"] == df["canonical_product"]).mean()
    print(f"{tag}: n={len(df)} accuracy={acc:.3f} {secs:.2f}s/complaint "
          f"peak_mem={mx.get_peak_memory() / 1e9:.1f}GB", file=sys.stderr)
    return df


if __name__ == "__main__":
    model_name = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else None
    for split_name in (("val",) if n else ("val", "test")):
        run_split(split_name, model_name, n=n)
