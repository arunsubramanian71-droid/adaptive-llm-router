#!/usr/bin/env python
"""Local validation for data/benchmarks/pilot_v1.jsonl — no network, no
model calls. Run after build_pilot_benchmark.py (or any time, to re-check
the committed file).

Checks:
  1. schema validation      -- loads via router.dataset.loader.load_dataset
  2. duplicate check         -- no two items share normalized prompt text
  3. missing-reference check -- every item has a reference_answer matching its task_type
  4. category-balance check  -- count per metadata.task_category
  5. evaluator-compatibility -- every task_type present has a registered evaluator
                                 (or is explicitly judge_scored)
  6. deterministic sampling  -- re-running build_pilot_benchmark.py's
                                 build_pilot() twice with the same seed
                                 yields byte-identical prompt_id lists
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from router.dataset.loader import DatasetError, load_dataset
from router.dataset.schemas import TaskType
from router.evaluation.evaluators import default_evaluator_registry

PILOT_PATH = REPO_ROOT / "data" / "benchmarks" / "pilot_v1.jsonl"
MANIFEST_PATH = REPO_ROOT / "data" / "benchmarks" / "pilot_v1.manifest.json"
RAW_CACHE_DIR = REPO_ROOT / "data" / "benchmarks" / "raw_cache"


def normalize(prompt: str) -> str:
    return " ".join(prompt.split()).strip().lower()


def main() -> int:
    ok = True

    print("=== 1. schema validation ===")
    try:
        items = load_dataset(PILOT_PATH)
        print(f"OK: {len(items)} items loaded and validated against DatasetItem schema")
    except DatasetError as exc:
        print(f"FAIL: {exc}")
        return 1

    print("\n=== 2. duplicate check ===")
    seen: dict[str, str] = {}
    dupes = []
    for item in items:
        norm = normalize(item.prompt)
        if norm in seen:
            dupes.append((item.prompt_id, seen[norm]))
        else:
            seen[norm] = item.prompt_id
    if dupes:
        print(f"FAIL: {len(dupes)} duplicate prompt(s): {dupes}")
        ok = False
    else:
        print(f"OK: no duplicate prompts among {len(items)} items")

    print("\n=== 3. missing-reference check ===")
    missing = [item.prompt_id for item in items if item.reference_answer is None]
    if missing:
        print(f"FAIL: {len(missing)} items with no reference_answer: {missing}")
        ok = False
    else:
        print(f"OK: all {len(items)} items have a non-null reference_answer")

    print("\n=== 4. category-balance check ===")
    by_category = Counter(item.metadata.get("task_category", "UNKNOWN") for item in items)
    for category, count in sorted(by_category.items()):
        print(f"  {category}: {count}")
    if len(set(by_category.values())) > 1:
        print(f"NOTE: categories are not perfectly equal-sized: {dict(by_category)}")
    else:
        print("OK: all categories equally sized")

    print("\n=== 5. evaluator-compatibility check ===")
    registry = default_evaluator_registry()
    by_task_type = Counter(item.task_type for item in items)
    for task_type, count in sorted(by_task_type.items(), key=lambda kv: kv[0].value):
        if task_type == TaskType.JUDGE_SCORED:
            print(f"  {task_type.value}: {count} items -- explicitly judge-required, no objective evaluator expected")
            continue
        if task_type in registry:
            print(f"  {task_type.value}: {count} items -- evaluator registered ({type(registry[task_type]).__name__})")
        else:
            print(f"FAIL: {task_type.value}: {count} items -- NO evaluator registered and not judge_scored")
            ok = False

    print("\n=== 6. deterministic sampling check ===")
    import build_pilot_benchmark as build_mod

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    seed = manifest["random_seed"]
    if not RAW_CACHE_DIR.exists():
        print("SKIP: raw_cache/ not present locally (only needed to re-verify sampling determinism)")
    else:
        selected_a, _, _ = build_mod.build_pilot(RAW_CACHE_DIR, seed)
        selected_b, _, _ = build_mod.build_pilot(RAW_CACHE_DIR, seed)
        ids_a = [c["prompt_id"] for c in selected_a]
        ids_b = [c["prompt_id"] for c in selected_b]
        file_ids = [item.prompt_id for item in items]
        if ids_a == ids_b == sorted(file_ids):
            print(f"OK: two independent build_pilot() runs with seed={seed} produced identical, sorted prompt_id lists")
        else:
            print("FAIL: sampling is not deterministic, or committed file doesn't match a fresh build")
            print(f"  run A == run B: {ids_a == ids_b}")
            print(f"  run A == committed file: {ids_a == sorted(file_ids)}")
            ok = False

    print("\n=== RESULT ===")
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
