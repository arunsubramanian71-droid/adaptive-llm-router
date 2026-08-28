"""Code-generation evaluator.

Two very different implementations live here — do not confuse them:

`HeuristicMockCodeEvalEvaluator` / `FixtureCodeEvalEvaluator` — the
**default**. Neither executes a single line of model-generated code. They
score from surface features (does a `def <entry_point>` exist, is there a
`return`, ...) or from a caller-supplied fixture. This is what
`evaluators.default_evaluator_registry()` wires up for CODE_GENERATION, and
what the rest of the pipeline (tests, demo, aggregation) exercises.

`UnsandboxedSubprocessCodeEvalEvaluator` — a real executor, kept for later
hardening, NOT wired into the default registry and refuses to construct
without `confirm_unsandboxed_execution=True`. It runs generated code in a
plain OS subprocess with only a wall-clock timeout and a trimmed
environment as protection. That is **not** a sandbox: filesystem access,
network access, and further process creation are all unrestricted. Before
this is ever pointed at real model output, it needs real OS-level
isolation (container/gVisor or a Windows Job Object with memory/CPU/
process-count limits), denied network access, and output-size capping —
see the class docstring for the full list. Until then, treat it as
inspection-only code, not something to run.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import cast

from router.dataset.schemas import DatasetItem, TaskType
from router.evaluation.evaluators.base import Evaluator
from router.evaluation.schemas import EvalResult

DEFAULT_TIMEOUT_SECONDS = 5.0
_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
_RESULT_MARKER = "RESULT_JSON:"


def extract_code(response_text: str) -> str:
    match = _CODE_FENCE_RE.search(response_text)
    return match.group(1) if match else response_text


# ---------------------------------------------------------------------------
# Default (non-executing) evaluators
# ---------------------------------------------------------------------------


class HeuristicMockCodeEvalEvaluator(Evaluator):
    """Surface-level, non-execution heuristic: NOT a correctness check.

    `reference_answer` may include an optional `"entry_point"` (the
    expected function name). Score is a weighted combination of "does a
    `def <entry_point>(` exist" and "does the body contain a `return`" —
    good enough to give the rest of the pipeline (aggregation, routing,
    metrics) a non-constant, deterministic signal to run against without
    ever executing model output. Do not read its score as "the code is
    correct."
    """

    name = "mock_code_eval_heuristic"
    task_type = TaskType.CODE_GENERATION

    def evaluate(self, item: DatasetItem, record_id: str, response_text: str | None) -> EvalResult:
        if not response_text:
            return self._empty_response_result(item, record_id)

        # dataset/loader.py validates reference_answer is a dict for this task_type.
        spec = cast(dict, item.reference_answer)
        entry_point = spec.get("entry_point")
        code = extract_code(response_text)

        if entry_point:
            has_def = re.search(rf"\bdef\s+{re.escape(entry_point)}\s*\(", code) is not None
        else:
            has_def = re.search(r"\bdef\s+\w+\s*\(", code) is not None
        has_return = "return" in code

        score = 0.6 * has_def + 0.4 * has_return

        return EvalResult(
            record_id=record_id,
            prompt_id=item.prompt_id,
            evaluator_name=self.name,
            score=score,
            passed=score == 1.0,
            details={"has_def": has_def, "has_return": has_return, "entry_point": entry_point},
        )


class FixtureCodeEvalEvaluator(Evaluator):
    """Returns a pre-scripted score per record_id — for tests that need an
    exact, controlled score rather than the heuristic's approximation."""

    name = "mock_code_eval_fixture"
    task_type = TaskType.CODE_GENERATION

    def __init__(self, fixture: dict[str, float], default_score: float = 0.0) -> None:
        self._fixture = fixture
        self._default_score = default_score

    def evaluate(self, item: DatasetItem, record_id: str, response_text: str | None) -> EvalResult:
        if not response_text:
            return self._empty_response_result(item, record_id)
        score = self._fixture.get(record_id, self._default_score)
        return EvalResult(
            record_id=record_id,
            prompt_id=item.prompt_id,
            evaluator_name=self.name,
            score=score,
            passed=score == 1.0,
            details={"source": "fixture"},
        )


# ---------------------------------------------------------------------------
# Real executor — opt-in only, not sandboxed, see module docstring
# ---------------------------------------------------------------------------

_HARNESS_TEMPLATE = """\
import json

results = []

{setup_code}

{response_code}

_test_calls = {test_calls_literal!r}

for _call, _expected in _test_calls:
    try:
        _actual = eval(_call)
        results.append(_actual == _expected)
    except Exception:
        results.append(False)

print({marker!r} + json.dumps(results))
"""


def _minimal_child_env() -> dict[str, str]:
    """Strips the child down to just enough environment for the
    interpreter to start — notably NOT inheriting anything like an API key
    that happens to be set in this shell. This is a small mitigation, not
    a substitute for real sandboxing (the child can still read/write any
    path or open any socket the host account can)."""
    keep = ("SystemRoot", "SystemDrive", "PATH", "TEMP", "TMP", "PATHEXT")
    return {k: os.environ[k] for k in keep if k in os.environ}


class UnsandboxedSubprocessCodeEvalEvaluator(Evaluator):
    """Executes model-generated code in a plain OS subprocess.

    NOT SAFE FOR UNTRUSTED INPUT. The only real control is a wall-clock
    timeout; there is no memory/CPU limit, no filesystem confinement
    (`cwd` is advisory only), no network restriction, and no restriction on
    the child spawning further processes. Before pointing this at real
    model output, add: container/gVisor-level OS isolation (or a Windows
    Job Object with memory/CPU/process-count limits), denied network
    access, and output-size capping. Refuses to construct unless
    `confirm_unsandboxed_execution=True` is passed explicitly, so it can
    never be reached accidentally through the default evaluator registry.
    """

    name = "code_eval_unsandboxed"
    task_type = TaskType.CODE_GENERATION

    def __init__(self, confirm_unsandboxed_execution: bool = False) -> None:
        if not confirm_unsandboxed_execution:
            raise ValueError(
                "UnsandboxedSubprocessCodeEvalEvaluator executes model-generated code with no "
                "real sandboxing (see class docstring). Pass confirm_unsandboxed_execution=True "
                "only if you have deliberately accepted that risk for this use case."
            )

    def evaluate(self, item: DatasetItem, record_id: str, response_text: str | None) -> EvalResult:
        if not response_text:
            return self._empty_response_result(item, record_id)

        # dataset/loader.py validates reference_answer is a dict for this task_type.
        spec = cast(dict, item.reference_answer)
        test_cases = spec.get("test_cases", [])
        if not test_cases:
            return EvalResult(
                record_id=record_id,
                prompt_id=item.prompt_id,
                evaluator_name=self.name,
                score=0.0,
                passed=False,
                error="reference_answer.test_cases is empty",
            )

        timeout = float(spec.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS))
        code = extract_code(response_text)
        test_calls_literal = [(tc["call"], tc["expected"]) for tc in test_cases]

        script = _HARNESS_TEMPLATE.format(
            setup_code=spec.get("setup_code", ""),
            response_code=code,
            test_calls_literal=test_calls_literal,
            marker=_RESULT_MARKER,
        )

        with tempfile.TemporaryDirectory(prefix="router_code_eval_") as tmpdir:
            script_path = Path(tmpdir) / "harness.py"
            script_path.write_text(script, encoding="utf-8")
            try:
                proc = subprocess.run(
                    [sys.executable, "-I", str(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=tmpdir,
                    env=_minimal_child_env(),
                    check=False,  # non-zero exit is a normal "harness crashed" case, handled below
                )
            except subprocess.TimeoutExpired:
                return EvalResult(
                    record_id=record_id,
                    prompt_id=item.prompt_id,
                    evaluator_name=self.name,
                    score=0.0,
                    passed=False,
                    error=f"execution exceeded {timeout}s timeout",
                )

        results_line = next(
            (line for line in proc.stdout.splitlines() if line.startswith(_RESULT_MARKER)), None
        )
        if results_line is None:
            return EvalResult(
                record_id=record_id,
                prompt_id=item.prompt_id,
                evaluator_name=self.name,
                score=0.0,
                passed=False,
                error="harness did not report results (syntax error or crash)",
                details={"stderr": proc.stderr[-2000:], "returncode": proc.returncode},
            )

        per_case_results: list[bool] = json.loads(results_line[len(_RESULT_MARKER) :])
        n_passed = sum(per_case_results)
        n_total = len(per_case_results) or 1
        score = n_passed / n_total

        return EvalResult(
            record_id=record_id,
            prompt_id=item.prompt_id,
            evaluator_name=self.name,
            score=score,
            passed=score == 1.0,
            details={"per_case_results": per_case_results, "n_test_cases": n_total},
        )
