from __future__ import annotations

from router.dataset.schemas import DatasetItem, TaskType
from router.evaluation.judge import FixtureJudgeClient, HeuristicMockJudgeClient, run_judge_pipeline


def test_fixture_judge_returns_scripted_score():
    judge = FixtureJudgeClient(fixture={"r1": 0.9}, default_score=0.5)
    verdict = judge.judge(prompt_id="p1", record_id="r1", prompt="x", response_text="hello")
    assert verdict.score == 0.9

    verdict_default = judge.judge(prompt_id="p1", record_id="r2", prompt="x", response_text="hello")
    assert verdict_default.score == 0.5


def test_fixture_judge_empty_response():
    judge = FixtureJudgeClient(fixture={})
    verdict = judge.judge(prompt_id="p1", record_id="r1", prompt="x", response_text=None)
    assert verdict.score == 0.0
    assert verdict.error is not None


def test_heuristic_mock_judge_reference_overlap():
    judge = HeuristicMockJudgeClient()
    verdict = judge.judge(
        prompt_id="p1", record_id="r1", prompt="x", response_text="the cat sat on the mat", reference="cat mat"
    )
    assert 0.0 < verdict.score <= 1.0


def test_heuristic_mock_judge_length_fallback():
    judge = HeuristicMockJudgeClient(target_length_words=5)
    verdict = judge.judge(prompt_id="p1", record_id="r1", prompt="x", response_text="one two three four five")
    assert verdict.score == 1.0


def test_run_judge_pipeline_skips_unknown_prompt_ids(make_response_record):
    items = {"p1": DatasetItem(prompt_id="p1", task_type=TaskType.JUDGE_SCORED, prompt="x")}
    records = [
        make_response_record("r1", "p1", response_text="hi there"),
        make_response_record("r2", "unknown-prompt", response_text="hi there"),
    ]
    judge = HeuristicMockJudgeClient()
    verdicts = run_judge_pipeline(items, records, judge)
    assert len(verdicts) == 1
    assert verdicts[0].record_id == "r1"
