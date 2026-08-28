"""Synthetic, hand-written prompts used ONLY to give the demo app's router
something to fit at startup. Not real benchmark data, not real model
responses — see `DEMO_DISCLAIMER` in `demo_router.py`."""

from __future__ import annotations

DEMO_HARD_EXAMPLES = [
    "Prove that this recursive algorithm terminates for all valid inputs.",
    "Derive a closed-form solution for this linear recurrence relation.",
    "Debug this race condition in a multithreaded job scheduler.",
    "Design an algorithm with better than quadratic time complexity for this problem.",
    "Write a rigorous proof by induction for the following claim.",
    "Refactor this recursive parser so it can't overflow the call stack.",
]
DEMO_EASY_EXAMPLES = [
    "What is the capital of Japan?",
    "Say good morning in Spanish.",
    "What color do you get by mixing blue and yellow?",
    "Convert 5 kilometers to miles.",
    "What day comes after Wednesday?",
    "Name two primary colors.",
]


def demo_training_data() -> tuple[list[str], list[int]]:
    prompts: list[str] = []
    labels: list[int] = []
    for i in range(4):
        for text in DEMO_HARD_EXAMPLES:
            prompts.append(f"{text} (sample {i})")
            labels.append(1)
        for text in DEMO_EASY_EXAMPLES:
            prompts.append(f"{text} (sample {i})")
            labels.append(0)
    return prompts, labels
