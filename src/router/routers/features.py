"""Handcrafted prompt features — shared by `HandcraftedFeatureRouter` and
`GradientBoostingRouter`. Deliberately simple, surface-level signals
(length, punctuation, code-ish markers); this is the "engineered features"
baseline router, distinct from the TF-IDF router's bag-of-words signal."""

from __future__ import annotations

import numpy as np

FEATURE_NAMES: tuple[str, ...] = (
    "length_chars",
    "length_words",
    "avg_word_length",
    "num_digits",
    "num_question_marks",
    "num_newlines",
    "uppercase_ratio",
    "has_code_fence",
    "num_code_keywords",
)

_CODE_KEYWORDS = ("def ", "class ", "function", "import ", "algorithm", "recursion")


def extract_handcrafted_features(prompt: str) -> dict[str, float]:
    words = prompt.split()
    length_chars = len(prompt)
    length_words = len(words)
    avg_word_length = (sum(len(w) for w in words) / length_words) if words else 0.0
    uppercase_ratio = (sum(c.isupper() for c in prompt) / length_chars) if length_chars else 0.0
    prompt_lower = prompt.lower()

    return {
        "length_chars": float(length_chars),
        "length_words": float(length_words),
        "avg_word_length": avg_word_length,
        "num_digits": float(sum(c.isdigit() for c in prompt)),
        "num_question_marks": float(prompt.count("?")),
        "num_newlines": float(prompt.count("\n")),
        "uppercase_ratio": uppercase_ratio,
        "has_code_fence": 1.0 if "```" in prompt else 0.0,
        "num_code_keywords": float(sum(kw in prompt_lower for kw in _CODE_KEYWORDS)),
    }


def featurize_batch(prompts: list[str]) -> np.ndarray:
    rows = [extract_handcrafted_features(p) for p in prompts]
    return np.array([[row[name] for name in FEATURE_NAMES] for row in rows], dtype=float)
