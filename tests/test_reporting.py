from __future__ import annotations

from pathlib import Path

from router.analysis.frontier import cost_quality_frontier, pareto_frontier
from router.analysis.thresholds import sweep_thresholds
from router.experiment.reporting import (
    plot_cost_quality_frontier,
    plot_reliability_diagram,
    rows_from_models,
    rows_to_markdown_table,
    threshold_results_table,
    write_csv_table,
)


def test_rows_from_models_and_markdown_table():
    results = sweep_thresholds([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2], thresholds=[0.5])
    rows = rows_from_models(results)
    assert rows[0]["tau"] == 0.5
    table = rows_to_markdown_table(rows)
    assert "tau" in table
    assert "0.5" in table


def test_rows_to_markdown_table_empty():
    assert rows_to_markdown_table([]) == "(no rows)"


def test_write_csv_table(tmp_path: Path):
    results = sweep_thresholds([1, 0], [0.9, 0.1], thresholds=[0.3, 0.7])
    path = threshold_results_table(results, tmp_path / "thresholds.csv")
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "tau" in content.splitlines()[0]
    assert len(content.splitlines()) == 3  # header + 2 rows


def test_write_csv_table_empty_rows(tmp_path: Path):
    path = write_csv_table([], tmp_path / "empty.csv")
    assert path.exists()
    assert path.read_text(encoding="utf-8") == ""


def test_plot_cost_quality_frontier_writes_png(tmp_path: Path):
    results = sweep_thresholds(
        [1, 0, 1, 0],
        [0.9, 0.1, 0.6, 0.4],
        thresholds=[0.0, 0.3, 0.5, 0.7, 1.0],
        cost_if_strong=[0.1, 0.1, 0.1, 0.1],
        cost_if_cheap=[0.01, 0.01, 0.01, 0.01],
        quality_if_strong=[1.0, 1.0, 1.0, 1.0],
        quality_if_cheap=[0.5, 0.5, 0.5, 0.5],
    )
    all_points = cost_quality_frontier(results)
    pareto_points = pareto_frontier(all_points)

    output_path = tmp_path / "frontier.png"
    result_path = plot_cost_quality_frontier(all_points, pareto_points, output_path)
    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_reliability_diagram_writes_png(tmp_path: Path):
    y_true = [1, 1, 1, 0, 0, 0, 1, 0]
    probs = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1, 0.6, 0.4]
    output_path = tmp_path / "reliability.png"
    result_path = plot_reliability_diagram(y_true, probs, output_path, n_bins=5)
    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_cost_quality_frontier_handles_empty_pareto(tmp_path: Path):
    output_path = tmp_path / "empty_frontier.png"
    plot_cost_quality_frontier([], [], output_path)
    assert output_path.exists()
