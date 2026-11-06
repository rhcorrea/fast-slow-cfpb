"""Checks that slm.classify's cached-prefix path reads the same prompt as the full template.

Needs the model under models/ (skipped otherwise). Run: uv run python -m tests.test_slm
"""
import sys

from src.models import slm

MODEL = "qwen3.5-4b-4bit"
TRICKY = ["plain text", "  leading and trailing spaces  ", "\nnewline first", "ends mid-word an",
          "x" * (slm.MAX_CHARS - 1) + " tail past the cut", "unicode: café — ½ ✓"]


def test_prefix_tokens(s):
    head, _ = slm.prompt(s.tokenizer, slm.SLOT).split(slm.SLOT)
    head_ids = s.tokenizer.encode(head, add_special_tokens=False)
    for t in TRICKY:
        full = s.tokenizer.encode(slm.prompt(s.tokenizer, t[:slm.MAX_CHARS]), add_special_tokens=False)
        cached = head_ids + s.tokenizer.encode(t[:slm.MAX_CHARS].strip() + s.tail, add_special_tokens=False)
        assert cached == full, repr(t[:30])


def test_prefix_untouched(s):
    # fork() must leave the shared cache as it was: same answer before and after other calls.
    a = slm.classify(s, TRICKY[0])
    slm.classify(s, TRICKY[4])
    assert slm.classify(s, TRICKY[0]) == a


if __name__ == "__main__":
    if not (slm.MODELS_DIR / MODEL).exists():
        sys.exit(f"skipped: models/{MODEL} not found")
    s = slm.load_model(MODEL)
    test_prefix_tokens(s)
    test_prefix_untouched(s)
    print("ok")
