"""Mock judges — for exercising the judge pipeline without a real model call.

Neither of these is a quality judge. `HeuristicMockJudgeClient` scores
deterministically from surface features (length, overlap with a reference
string) purely so pipeline plumbing (aggregation, routing, metrics) has
something non-constant to run against in tests. `FixtureJudgeClient` looks
up a score from a caller-supplied fixture mapping so a test can pin exact
verdicts. Neither produces a real quality judgment — do not use their
output as an experimental result.
"""

from __future__ import annotations

from router.evaluation.judge.base import JudgeClient, JudgeVerdict


class FixtureJudgeClient(JudgeClient):
    """Returns a pre-scripted score per record_id, from a fixture dict."""

    name = "fixture_judge"

    def __init__(self, fixture: dict[str, float], default_score: float = 0.5) -> None:
        self._fixture = fixture
        self._default_score = default_score

    def judge(
        self,
        prompt_id: str,
        record_id: str,
        prompt: str,
        response_text: str | None,
        rubric: str | None = None,
        reference: str | None = None,
    ) -> JudgeVerdict:
        if not response_text:
            return self._empty_response_verdict(prompt_id, record_id)
        score = self._fixture.get(record_id, self._default_score)
        return JudgeVerdict(
            record_id=record_id,
            prompt_id=prompt_id,
            judge_name=self.name,
            score=score,
            rationale=f"fixture score for {record_id}",
        )


class HeuristicMockJudgeClient(JudgeClient):
    """Deterministic, non-LLM stand-in: scores word-overlap with `reference`
    when one is given, else a length-based heuristic. Used to exercise the
    judge pipeline end-to-end in tests without a real model."""

    name = "heuristic_mock_judge"

    def __init__(self, target_length_words: int = 40) -> None:
        self._target_length_words = target_length_words

    def judge(
        self,
        prompt_id: str,
        record_id: str,
        prompt: str,
        response_text: str | None,
        rubric: str | None = None,
        reference: str | None = None,
    ) -> JudgeVerdict:
        if not response_text:
            return self._empty_response_verdict(prompt_id, record_id)

        if reference:
            response_words = set(response_text.lower().split())
            reference_words = set(reference.lower().split())
            overlap = len(response_words & reference_words) / max(1, len(reference_words))
            score = min(1.0, overlap)
            rationale = f"word overlap with reference: {overlap:.2f}"
        else:
            n_words = len(response_text.split())
            score = max(0.0, 1.0 - abs(n_words - self._target_length_words) / self._target_length_words)
            rationale = f"length heuristic: {n_words} words vs target {self._target_length_words}"

        return JudgeVerdict(
            record_id=record_id,
            prompt_id=prompt_id,
            judge_name=self.name,
            score=score,
            rationale=rationale,
        )
