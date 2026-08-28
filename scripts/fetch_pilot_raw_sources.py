#!/usr/bin/env python
"""Downloads the raw upstream files the pilot benchmark is built from.

Every URL here is the file we actually fetched to build
`data/benchmarks/pilot_v1.jsonl` — re-running this script re-downloads the
same files into `data/benchmarks/raw_cache/` (gitignored; not committed,
since it's full upstream dumps, not our sampled subset). Compare the
printed sha256 hashes against `data/benchmarks/pilot_v1.manifest.json`'s
`raw_source_files` block to confirm you got byte-identical source data
before re-running `build_pilot_benchmark.py`.

This script makes network requests to public GitHub raw-content URLs. It
does not call any LLM API and is unrelated to model inference.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

RAW_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "benchmarks" / "raw_cache"

# (output filename, source URL, source repo, license per that repo's own LICENSE file)
SOURCES = [
    (
        "gsm8k_test.jsonl",
        "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
        "openai/grade-school-math",
        "MIT",
    ),
    (
        "gsm8k_license.txt",
        "https://raw.githubusercontent.com/openai/grade-school-math/master/LICENSE",
        "openai/grade-school-math",
        "MIT",
    ),
    (
        "humaneval.jsonl.gz",
        "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz",
        "openai/human-eval",
        "MIT",
    ),
    (
        "humaneval_license.txt",
        "https://raw.githubusercontent.com/openai/human-eval/master/LICENSE",
        "openai/human-eval",
        "MIT",
    ),
    (
        "truthfulqa_mc.json",
        "https://raw.githubusercontent.com/sylinrl/TruthfulQA/master/data/v0/mc_task.json",
        "sylinrl/TruthfulQA",
        "Apache-2.0",
    ),
    (
        "truthfulqa_license.txt",
        "https://raw.githubusercontent.com/sylinrl/TruthfulQA/master/LICENSE",
        "sylinrl/TruthfulQA",
        "Apache-2.0",
    ),
    (
        "ifeval_input.jsonl",
        "https://raw.githubusercontent.com/google-research/google-research/master/instruction_following_eval/data/input_data.jsonl",
        "google-research/google-research (instruction_following_eval)",
        "Apache-2.0",
    ),
    (
        "google_research_license.txt",
        "https://raw.githubusercontent.com/google-research/google-research/master/LICENSE",
        "google-research/google-research",
        "Apache-2.0",
    ),
    (
        "bbh_license.txt",
        "https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/LICENSE",
        "suzgunmirac/BIG-Bench-Hard",
        "MIT",
    ),
]

BBH_TASKS = [
    "logical_deduction_five_objects",
    "date_understanding",
    "object_counting",
    "causal_judgement",
    "tracking_shuffled_objects_three_objects",
]
for _task in BBH_TASKS:
    SOURCES.append(
        (
            f"bbh_{_task}.json",
            f"https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/bbh/{_task}.json",
            "suzgunmirac/BIG-Bench-Hard",
            "MIT",
        )
    )


def fetch(filename: str, url: str) -> None:
    RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_CACHE_DIR / filename
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = resp.read()
    dest.write_bytes(data)
    print(f"{hashlib.sha256(data).hexdigest()}  {filename}  <- {url}")


def main() -> None:
    for filename, url, _repo, _license in SOURCES:
        fetch(filename, url)


if __name__ == "__main__":
    main()
