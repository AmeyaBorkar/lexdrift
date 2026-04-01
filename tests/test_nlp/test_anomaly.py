"""Tests for lexdrift.nlp.anomaly — detect_anomaly() and detect_trends()."""

import pytest

from lexdrift.nlp.anomaly import AnomalyResult, compute_z_score, detect_anomaly, detect_trends


class TestComputeZScore:
    def test_basic_z_score(self):
        # value=3, mean=1, stddev=1 => z = 2.0
        assert compute_z_score(3.0, 1.0, 1.0) == 2.0

    def test_zero_stddev_returns_none(self):
        assert compute_z_score(5.0, 2.0, 0.0) is None

    def test_very_small_stddev_returns_none(self):
        assert compute_z_score(5.0, 2.0, 0.0005) is None


class TestDetectAnomaly:
    def test_spike_flagged_as_anomalous(self):
        # Stable history around 0.10, then a sudden jump to 0.50
        history = [0.10, 0.11, 0.09, 0.10, 0.12, 0.10]
        result = detect_anomaly(0.50, history)
        assert isinstance(result, AnomalyResult)
        assert result.is_anomalous is True
        assert result.anomaly_level in ("high", "extreme")

    def test_normal_drift_not_anomalous(self):
        history = [0.10, 0.11, 0.09, 0.10, 0.12, 0.10]
        result = detect_anomaly(0.11, history)
        assert result.is_anomalous is False
        assert result.anomaly_level == "normal"

    def test_insufficient_history_fallback(self):
        # With only 2 data points, falls back to absolute threshold check
        result = detect_anomaly(0.35, [0.05, 0.06])
        assert result.is_anomalous is True
        assert result.anomaly_level == "high"

    def test_sector_history_elevates_anomaly(self):
        company_hist = [0.10, 0.10, 0.10, 0.11, 0.10]
        # Sector peers also have low drift; the spike is unusual for the sector
        sector_hist = [0.09, 0.08, 0.10, 0.09, 0.10, 0.08, 0.11]
        result = detect_anomaly(0.40, company_hist, sector_history=sector_hist)
        assert result.is_anomalous is True

    def test_result_contains_statistics(self):
        history = [0.10, 0.12, 0.11, 0.09, 0.10]
        result = detect_anomaly(0.11, history)
        assert result.company_mean is not None
        assert result.company_stddev is not None


class TestDetectTrends:
    def test_drift_acceleration_detected(self):
        # Monotonically increasing drift
        history = [0.05, 0.08, 0.12, 0.18, 0.25]
        result = detect_trends(history, min_periods=4)
        assert result["has_trend"] is True
        signal_types = {s["type"] for s in result["signals"]}
        assert "drift_acceleration" in signal_types

    def test_no_trend_in_stable_history(self):
        history = [0.10, 0.10, 0.10, 0.10, 0.10]
        result = detect_trends(history, min_periods=4)
        # Stable history has no acceleration or spike
        accel_signals = [s for s in result["signals"] if s["type"] == "drift_acceleration"]
        assert len(accel_signals) == 0

    def test_insufficient_history(self):
        result = detect_trends([0.1, 0.2], min_periods=4)
        assert result["has_trend"] is False
        assert result["reason"] == "insufficient_history"
