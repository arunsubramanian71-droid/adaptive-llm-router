#!/usr/bin/env python
"""Builds the ~200-item real pilot benchmark from verified-license public
sources into `data/benchmarks/pilot_v1.jsonl` + a manifest.

Sources (see docs/decisions/ADR-0003-pilot-benchmark-sources.md for the
full license verification trail):

  - GSM8K            (openai/grade-school-math, MIT)       -> exact_match, mathematical_reasoning
  - HumanEval        (openai/human-eval, MIT)               -> code_generation, coding
  - BIG-Bench-Hard   (suzgunmirac/BIG-Bench-Hard, MIT)      -> exact_match, logical_reasoning
  - TruthfulQA MC1   (sylinrl/TruthfulQA, Apache-2.0)       -> exact_match, factual_knowledge
  - IFEval           (google-research, Apache-2.0)          -> constraint_checking, structured_extraction_constraint_following

Every source's license permits redistribution, so the sampled items are
copied into the repo directly (not just referenced) — see the manifest's
`sources` block for the exact license/authoritative-source citation used.

Deterministic: same raw_cache contents + same --seed always produce the
same output JSONL byte-for-byte. No network access, no model calls — this
script only reads already-downloaded files from data/benchmarks/raw_cache/
(see fetch_pilot_raw_sources.py to (re)download them).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_CACHE_DIR = REPO_ROOT / "data" / "benchmarks" / "raw_cache"
OUTPUT_JSONL = REPO_ROOT / "data" / "benchmarks" / "pilot_v1.jsonl"
OUTPUT_MANIFEST = REPO_ROOT / "data" / "benchmarks" / "pilot_v1.manifest.json"

PILOT_VERSION = "pilot_v1"
DEFAULT_SEED = 1337

CATEGORY_TARGETS: dict[str, int] = {
    "mathematical_reasoning": 40,
    "coding": 40,
    "logical_reasoning": 40,
    "factual_knowledge": 40,
    "structured_extraction_constraint_following": 40,
}

SOURCE_INFO = {
    "gsm8k": {
        "dataset": "GSM8K",
        "repo": "openai/grade-school-math",
        "license": "MIT",
        "license_source_url": "https://raw.githubusercontent.com/openai/grade-school-math/master/LICENSE",
        "split": "test",
    },
    "humaneval": {
        "dataset": "HumanEval",
        "repo": "openai/human-eval",
        "license": "MIT",
        "license_source_url": "https://raw.githubusercontent.com/openai/human-eval/master/LICENSE",
        "split": "test (all 164 problems)",
    },
    "big_bench_hard": {
        "dataset": "BIG-Bench-Hard (BBH)",
        "repo": "suzgunmirac/BIG-Bench-Hard",
        "license": "MIT",
        "license_source_url": "https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/LICENSE",
        "split": "bbh/{logical_deduction_five_objects,date_understanding,object_counting,"
        "causal_judgement,tracking_shuffled_objects_three_objects}.json",
    },
    "truthfulqa": {
        "dataset": "TruthfulQA (MC1)",
        "repo": "sylinrl/TruthfulQA",
        "license": "Apache-2.0",
        "license_source_url": "https://raw.githubusercontent.com/sylinrl/TruthfulQA/master/LICENSE",
        "split": "data/v0/mc_task.json",
    },
    "ifeval": {
        "dataset": "IFEval",
        "repo": "google-research/google-research (instruction_following_eval)",
        "license": "Apache-2.0",
        "license_source_url": "https://raw.githubusercontent.com/google-research/google-research/master/LICENSE",
        "split": "instruction_following_eval/data/input_data.jsonl",
    },
}

BBH_TASKS = [
    "logical_deduction_five_objects",
    "date_understanding",
    "object_counting",
    "causal_judgement",
    "tracking_shuffled_objects_three_objects",
]


# ---------------------------------------------------------------------------
# Per-source loaders. Each returns (candidates, exclusions).
# ---------------------------------------------------------------------------

_GSM8K_FINAL_ANSWER_RE = re.compile(r"^-?[\d,]*\.?\d+$")


def load_gsm8k(raw_dir: Path) -> tuple[list[dict], list[dict]]:
    candidates, exclusions = [], []
    path = raw_dir / "gsm8k_test.jsonl"
    with path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            row = json.loads(line)
            question, answer_text = row["question"], row["answer"]
            if "####" not in answer_text:
                exclusions.append(
                    {"source": "gsm8k", "original_id": str(idx), "reason": "no '#### <answer>' marker in answer field"}
                )
                continue
            final_raw = answer_text.split("####")[-1].strip()
            if not _GSM8K_FINAL_ANSWER_RE.match(final_raw):
                exclusions.append(
                    {
                        "source": "gsm8k",
                        "original_id": str(idx),
                        "reason": f"final answer {final_raw!r} is not a clean parseable number",
                    }
                )
                continue
            final = final_raw.replace(",", "")
            candidates.append(
                {
                    "prompt_id": f"gsm8k-test-{idx}",
                    "task_type": "exact_match",
                    "prompt": f"{question}\n\nAnswer with only the final number, with no other text, units, or punctuation.",
                    "reference_answer": final,
                    "metadata": {
                        "source": "gsm8k",
                        "source_repo": "openai/grade-school-math",
                        "source_split": "test",
                        "source_license": "MIT",
                        "original_id": idx,
                        "task_category": "mathematical_reasoning",
                        "original_question": question,
                        "original_answer_with_reasoning": answer_text,
                        "prompt_augmentation": "appended an explicit answer-format instruction for exact_match compatibility",
                    },
                }
            )
    return candidates, exclusions


def load_humaneval(raw_dir: Path) -> tuple[list[dict], list[dict]]:
    candidates, exclusions = [], []
    path = raw_dir / "humaneval.jsonl"
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            task_id, entry_point, prompt_code = row["task_id"], row["entry_point"], row["prompt"]
            candidates.append(
                {
                    "prompt_id": f"humaneval-{task_id.replace('/', '-')}",
                    "task_type": "code_generation",
                    "prompt": (
                        "Complete the following Python function. Respond with the full function "
                        "implementation (including the signature) inside a single ```python code "
                        f"fence and nothing else.\n\n{prompt_code}"
                    ),
                    "reference_answer": {
                        "entry_point": entry_point,
                        "test_cases": [],
                        "humaneval_test_code": row["test"],
                        "humaneval_canonical_solution": row["canonical_solution"],
                    },
                    "metadata": {
                        "source": "humaneval",
                        "source_repo": "openai/human-eval",
                        "source_split": "test",
                        "source_license": "MIT",
                        "original_id": task_id,
                        "task_category": "coding",
                        "prompt_augmentation": "wrapped the raw HumanEval completion-style prompt in an instruct-style request",
                        "evaluator_note": (
                            "Scored only by the default HeuristicMockCodeEvalEvaluator (checks for "
                            "`def <entry_point>(` and a `return` statement — a non-executing surface "
                            "heuristic, NOT functional correctness). humaneval_test_code / "
                            "humaneval_canonical_solution are preserved for a future real, sandboxed "
                            "executor; test_cases is intentionally empty because HumanEval ships "
                            "assert-based test code, not call/expected pairs, so "
                            "UnsandboxedSubprocessCodeEvalEvaluator cannot consume this item as-is."
                        ),
                    },
                }
            )
    return candidates, exclusions


def load_bbh(raw_dir: Path, task_names: list[str]) -> tuple[list[dict], list[dict]]:
    candidates, exclusions = [], []
    for task_name in task_names:
        path = raw_dir / f"bbh_{task_name}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        for idx, ex in enumerate(data["examples"]):
            input_text, target = ex["input"], ex["target"].strip()
            if not target:
                exclusions.append(
                    {"source": "big_bench_hard", "original_id": f"{task_name}-{idx}", "reason": "empty target"}
                )
                continue
            # BBH tasks format their "Options:" section two different ways:
            # lettered, e.g. "(A) ..." (logical_deduction, date_understanding,
            # tracking_shuffled_objects) with a "(X)"-shaped target, or a plain
            # dash list, e.g. "- Yes\n- No" (causal_judgement) with the raw
            # option text as target. Using the wrong instruction for the wrong
            # style asks the model to answer in a format that can never match
            # the reference (e.g. "answer with a letter" when there is none).
            has_lettered_options = re.search(r"\n\([A-Z]\) ", input_text) is not None
            has_dash_options = "\nOptions:\n- " in input_text
            if has_lettered_options:
                instruction = "\n\nAnswer with only the letter of the correct option in parentheses, e.g. (A)."
            elif has_dash_options:
                instruction = "\n\nAnswer with only the exact text of the correct option from the list above, with no other text."
            else:
                instruction = "\n\nAnswer with only the final answer, with no other text."
            candidates.append(
                {
                    "prompt_id": f"bbh-{task_name}-{idx}",
                    "task_type": "exact_match",
                    "prompt": input_text + instruction,
                    "reference_answer": target,
                    "metadata": {
                        "source": "big_bench_hard",
                        "source_repo": "suzgunmirac/BIG-Bench-Hard",
                        "source_split": task_name,
                        "source_license": "MIT",
                        "original_id": idx,
                        "bbh_task": task_name,
                        "task_category": "logical_reasoning",
                        "original_input": input_text,
                        "prompt_augmentation": "appended an explicit answer-format instruction for exact_match compatibility",
                    },
                }
            )
    return candidates, exclusions


def load_truthfulqa_mc1(raw_dir: Path) -> tuple[list[dict], list[dict]]:
    candidates, exclusions = [], []
    data = json.loads((raw_dir / "truthfulqa_mc.json").read_text(encoding="utf-8"))
    for idx, item in enumerate(data):
        question, targets = item["question"], item["mc1_targets"]
        correct = [text for text, val in targets.items() if val == 1]
        if len(correct) != 1:
            exclusions.append(
                {
                    "source": "truthfulqa",
                    "original_id": str(idx),
                    "reason": f"mc1_targets has {len(correct)} answers marked correct, expected exactly 1",
                }
            )
            continue
        options = sorted(targets.keys(), key=str.casefold)
        if len(options) > 26:
            exclusions.append(
                {"source": "truthfulqa", "original_id": str(idx), "reason": f"{len(options)} options exceeds 26-letter limit"}
            )
            continue
        letters = [chr(ord("A") + i) for i in range(len(options))]
        option_lines = "\n".join(f"({letter}) {text}" for letter, text in zip(letters, options, strict=True))
        correct_letter = letters[options.index(correct[0])]
        candidates.append(
            {
                "prompt_id": f"truthfulqa-mc1-{idx}",
                "task_type": "exact_match",
                "prompt": (
                    f"{question}\nOptions:\n{option_lines}\n\n"
                    "Answer with only the letter of the correct option in parentheses, e.g. (A)."
                ),
                "reference_answer": f"({correct_letter})",
                "metadata": {
                    "source": "truthfulqa",
                    "source_repo": "sylinrl/TruthfulQA",
                    "source_split": "data/v0/mc_task.json",
                    "source_license": "Apache-2.0",
                    "original_id": idx,
                    "task_category": "factual_knowledge",
                    "original_question": question,
                    "mc1_options_in_letter_order": options,
                    "prompt_augmentation": (
                        "converted the free-text MC1 target dict into a lettered multiple-choice prompt; "
                        "options ordered case-insensitive-alphabetically by answer text for determinism "
                        "(not the original unordered dict order)"
                    ),
                },
            }
        )
    return candidates, exclusions


SAFE_IFEVAL_INSTRUCTIONS = {
    "keywords:existence",
    "keywords:forbidden_words",
    "startend:end_checker",
    "startend:quotation",
    "detectable_format:title",
    "change_case:english_capital",
    "change_case:english_lowercase",
    "punctuation:no_comma",
    "length_constraints:number_words",
}


def _ifeval_instruction_to_constraints(instruction_id: str, kwargs: dict) -> list[dict]:
    """Mapping verified against the authoritative instruction_id ->
    kwargs-field-name registry in google-research/google-research's
    instruction_following_eval/{instructions.py,instructions_registry.py}."""
    if instruction_id == "keywords:existence":
        return [{"type": "contains", "value": kw} for kw in kwargs["keywords"]]
    if instruction_id == "keywords:forbidden_words":
        return [{"type": "not_contains", "value": w} for w in kwargs["forbidden_words"]]
    if instruction_id == "startend:end_checker":
        return [{"type": "regex", "pattern": f"(?s){re.escape(kwargs['end_phrase'])}\\s*$"}]
    if instruction_id == "startend:quotation":
        return [{"type": "regex", "pattern": r'(?s)^".*"$'}]
    if instruction_id == "detectable_format:title":
        return [{"type": "regex", "pattern": r"(?s)<<.+>>"}]
    if instruction_id == "change_case:english_capital":
        return [{"type": "regex", "pattern": r"(?s)^[^a-z]*$"}]
    if instruction_id == "change_case:english_lowercase":
        return [{"type": "regex", "pattern": r"(?s)^[^A-Z]*$"}]
    if instruction_id == "punctuation:no_comma":
        return [{"type": "not_contains", "value": ","}]
    if instruction_id == "length_constraints:number_words":
        relation, num_words = kwargs["relation"], kwargs["num_words"]
        if relation == "at least":
            return [{"type": "min_words", "value": num_words}]
        if relation == "less than":
            # our max_words is "<=", IFEval's "less than" is strict "<" -> shift by one
            return [{"type": "max_words", "value": num_words - 1}]
        raise ValueError(f"unexpected relation {relation!r}")
    raise ValueError(f"unmapped instruction_id {instruction_id!r}")


def load_ifeval(raw_dir: Path) -> tuple[list[dict], list[dict]]:
    candidates, exclusions = [], []
    path = raw_dir / "ifeval_input.jsonl"
    with path.open(encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            key, instr_ids = item["key"], item["instruction_id_list"]
            unsupported = sorted(set(instr_ids) - SAFE_IFEVAL_INSTRUCTIONS)
            if unsupported:
                exclusions.append(
                    {
                        "source": "ifeval",
                        "original_id": str(key),
                        "reason": f"uses instruction type(s) not supported by our constraint_checking evaluator: {unsupported}",
                    }
                )
                continue
            try:
                constraints = [
                    c
                    for instr_id, kw in zip(instr_ids, item["kwargs"], strict=True)
                    for c in _ifeval_instruction_to_constraints(instr_id, kw)
                ]
            except (KeyError, ValueError) as exc:
                exclusions.append(
                    {"source": "ifeval", "original_id": str(key), "reason": f"failed to translate to constraint schema: {exc}"}
                )
                continue
            candidates.append(
                {
                    "prompt_id": f"ifeval-{key}",
                    "task_type": "constraint_checking",
                    "prompt": item["prompt"],
                    "reference_answer": constraints,
                    "metadata": {
                        "source": "ifeval",
                        "source_repo": "google-research/google-research (instruction_following_eval)",
                        "source_split": "input_data.jsonl",
                        "source_license": "Apache-2.0",
                        "original_id": key,
                        "task_category": "structured_extraction_constraint_following",
                        "original_instruction_id_list": instr_ids,
                        "original_kwargs": item["kwargs"],
                        "prompt_augmentation": "none — original IFEval prompt used verbatim",
                    },
                }
            )
    return candidates, exclusions


# ---------------------------------------------------------------------------
# Dedup + stratified sampling + assembly
# ---------------------------------------------------------------------------


def normalize_for_dedup(prompt: str) -> str:
    return " ".join(prompt.split()).strip().lower()


def build_pilot(raw_dir: Path, seed: int) -> tuple[list[dict], list[dict], dict[str, int]]:
    by_category: dict[str, list[dict]] = {
        "mathematical_reasoning": [],
        "coding": [],
        "logical_reasoning": [],
        "factual_knowledge": [],
        "structured_extraction_constraint_following": [],
    }
    exclusions: list[dict] = []

    loaders = [
        load_gsm8k(raw_dir),
        load_humaneval(raw_dir),
        load_bbh(raw_dir, BBH_TASKS),
        load_truthfulqa_mc1(raw_dir),
        load_ifeval(raw_dir),
    ]
    for cands, excl in loaders:
        exclusions.extend(excl)
        for c in cands:
            by_category[c["metadata"]["task_category"]].append(c)

    # Global de-duplication across the full eligible pool, in a fixed,
    # deterministic order (category order above, then original order within
    # each category) -- BEFORE sampling, so sampling always draws from an
    # already-deduplicated pool.
    seen: dict[str, str] = {}
    for category, category_candidates in by_category.items():
        deduped = []
        for c in category_candidates:
            norm = normalize_for_dedup(c["prompt"])
            if norm in seen:
                exclusions.append(
                    {
                        "source": c["metadata"]["source"],
                        "original_id": str(c["metadata"]["original_id"]),
                        "reason": f"duplicate prompt text (identical, after whitespace/case normalization, to {seen[norm]!r})",
                    }
                )
                continue
            seen[norm] = c["prompt_id"]
            deduped.append(c)
        by_category[category] = deduped

    selected: list[dict] = []
    counts: dict[str, int] = {}
    for category, target in CATEGORY_TARGETS.items():
        pool = by_category[category]
        rng = random.Random(seed)  # fresh Random per category: order-of-calls-independent
        pool_sorted = sorted(pool, key=lambda c: c["prompt_id"])  # stable input order before shuffling
        rng.shuffle(pool_sorted)
        chosen = pool_sorted[:target]
        if len(chosen) < target:
            exclusions.append(
                {
                    "source": category,
                    "original_id": "N/A",
                    "reason": f"only {len(chosen)} eligible de-duplicated items available for target {target}",
                }
            )
        selected.extend(chosen)
        counts[category] = len(chosen)

    selected.sort(key=lambda c: c["prompt_id"])
    return selected, exclusions, counts


def to_dataset_item_json(candidate: dict) -> dict[str, Any]:
    return {
        "prompt_id": candidate["prompt_id"],
        "task_type": candidate["task_type"],
        "prompt": candidate["prompt"],
        "system_prompt": None,
        "reference_answer": candidate["reference_answer"],
        "metadata": candidate["metadata"],
    }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_CACHE_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=OUTPUT_JSONL)
    parser.add_argument("--manifest-out", type=Path, default=OUTPUT_MANIFEST)
    args = parser.parse_args()

    if not args.raw_dir.exists():
        print(f"raw source directory not found: {args.raw_dir}", file=sys.stderr)
        print("run scripts/fetch_pilot_raw_sources.py first", file=sys.stderr)
        return 1

    selected, exclusions, counts = build_pilot(args.raw_dir, args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for candidate in selected:
            f.write(json.dumps(to_dataset_item_json(candidate), ensure_ascii=False))
            f.write("\n")

    raw_file_hashes = {
        p.name: sha256_file(p) for p in sorted(args.raw_dir.iterdir()) if p.is_file() and p.suffix != ".gz"
    }

    manifest = {
        "benchmark_version": PILOT_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "random_seed": args.seed,
        "selection_procedure": (
            "Per source: load all items, apply a source-specific eligibility filter (documented per "
            "source below), then a single global de-duplication pass across all eligible items (by "
            "whitespace/case-normalized prompt text, fixed category+original-item order, first "
            "occurrence wins). Then, independently per task category, deterministically shuffle the "
            "de-duplicated eligible pool with random.Random(seed) and take the first "
            f"{next(iter(CATEGORY_TARGETS.values()))} items. Final file is sorted by prompt_id. This is "
            "purely a pilot pool for model-pair screening / quality-separation / stochasticity / "
            "cost-quality-headroom analysis -- it is not, and must not be treated as, the held-out test "
            "set; no model, delta, k, or router selection may be tuned against it."
        ),
        "category_targets": CATEGORY_TARGETS,
        "category_counts_selected": counts,
        "total_selected": len(selected),
        "sources": SOURCE_INFO,
        "raw_source_files_sha256": raw_file_hashes,
        "exclusions": exclusions,
        "n_exclusions": len(exclusions),
    }
    args.manifest_out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"wrote {len(selected)} items -> {args.out}")
    print(f"wrote manifest -> {args.manifest_out}")
    print(f"category counts: {counts}")
    print(f"exclusions recorded: {len(exclusions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
