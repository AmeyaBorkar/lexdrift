"""Tests for lexdrift.nlp.velocity — compute_semantic_kinematics()."""

import pytest

np = pytest.importorskip("numpy")

from lexdrift.nlp.velocity import SemanticKinematics, compute_semantic_kinematics


def _make_history(drift_values: list[float], start_year: int = 2020) -> list[dict]:
    """Build a drift_history list with quarterly dates."""
    history = []
    for i, val in enumerate(drift_values):
        quarter = i % 4 + 1
        year = start_year + i // 4
        month = {1: "01", 2: "04", 3: "07", 4: "10"}[quarter]
        history.append({
            "filing_date": f"{year}-{month}-15",
            "cosine_distance": val,
        })
    return history


class TestComputeSemanticKinematics:
    def test_basic_result_structure(self):
        history = _make_history([0.05, 0.06, 0.08, 0.10, 0.13])
        result = compute_semantic_kinematics(history)
        assert isinstance(result, SemanticKinematics)
        assert result.periods_analyzed == 5
        assert len(result.velocity) == 4  # n-1 for forward differences
        assert len(result.acceleration) == 3  # n-2
        assert len(result.jerk) == 2  # n-3

    def test_accelerating_drift_detected(self):
        # Quadratically increasing drift -> positive acceleration
        history = _make_history([0.01, 0.02, 0.05, 0.10, 0.20])
        result = compute_semantic_kinematics(history)
        assert result.latest_velocity > 0
        assert result.latest_acceleration > 0
        assert result.phase in ("accelerating", "volatile", "regime_change")

    def test_stable_drift_detected(self):
        # Very small, nearly constant drift values
        history = _make_history([0.10, 0.10, 0.10, 0.10, 0.10])
        result = compute_semantic_kinematics(history)
        assert result.phase == "stable"

    def test_minimum_two_points_required(self):
        with pytest.raises(ValueError, match="At least 2"):
            compute_semantic_kinematics([{"filing_date": "2024-01-15", "cosine_distance": 0.1}])

    def test_momentum_computed(self):
        history = _make_history([0.05, 0.06, 0.08, 0.10, 0.13])
        result = compute_semantic_kinematics(history)
        assert len(result.momentum) == len(result.velocity)
        assert result.latest_momentum != 0.0

    def test_filing_dates_preserved(self):
        history = _make_history([0.05, 0.06, 0.08])
        result = compute_semantic_kinematics(history)
        assert len(result.filing_dates) == 3
        assert result.filing_dates[0] == "2020-01-15"
