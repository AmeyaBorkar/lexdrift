"""Semantic Velocity & Acceleration -- Kinematic Analysis of Filing Drift.

Novel contribution: Existing drift metrics capture *how much* a filing changed
between two consecutive periods.  This module computes the *derivatives* of the
drift time series, answering a fundamentally different question: **is the rate
of change itself accelerating?**

Physical analogy
----------------
If drift is *position* (where a filing's language currently sits in semantic
space), then:

- **Velocity** (1st derivative) = how fast the language is changing per unit
  time.  Positive velocity means the company is actively revising disclosures.
- **Acceleration** (2nd derivative) = is the revision rate speeding up or
  slowing down?  Positive acceleration in Risk Factors is a leading indicator
  of emerging material events.
- **Momentum** (exponential moving average of velocity) = captures sustained
  directional change, smoothing out quarter-to-quarter noise.
- **Jerk** (3rd derivative) = sudden discontinuities in acceleration.  Detects
  *regime changes* -- a company that was stable for years suddenly begins
  revising rapidly.

Phase classification
--------------------
The kinematic state vector is mapped to one of six disclosure phases:

- ``stable`` -- low velocity, low acceleration.
- ``drifting`` -- moderate velocity, near-zero acceleration.
- ``accelerating`` -- positive acceleration (changes speeding up).
- ``decelerating`` -- negative acceleration (changes slowing down).
- ``volatile`` -- high jerk, oscillating velocity.
- ``regime_change`` -- extreme jerk combined with high acceleration.

This classification provides an at-a-glance summary for analysts monitoring
thousands of companies simultaneously.

References
----------
- Cohen, L., Malloy, C., & Nguyen, Q. (2020). Lazy prices. *The Journal of
  Finance*, 75(3), 1371-1415.  [Documents that changes in 10-K language
  predict future returns -- our velocity/acceleration framework extends
  their insight to change dynamics.]
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default EMA half-life in periods (quarters).  A half-life of 3 means that
# observations from 3 quarters ago carry half the weight of the most recent.
_DEFAULT_EMA_HALFLIFE: int = 3

# Phase classification thresholds (calibrated on 10-K drift distributions).
_VELOCITY_LOW: float = 0.01       # drift units per quarter
_VELOCITY_HIGH: float = 0.05
_ACCEL_THRESHOLD: float = 0.01
_JERK_THRESHOLD: float = 0.005
_REGIME_JERK: float = 0.015       # extreme jerk threshold


# ---------------------------------------------------------------------------
# Helper: date parsing
# ---------------------------------------------------------------------------

def _parse_date(date_val) -> datetime:
    """Parse a filing date from various input formats.

    Accepts ``datetime`` objects, ISO-format strings, and common date strings.
    """
    if isinstance(date_val, datetime):
        return date_val
    if isinstance(date_val, str):
        # Try ISO format first, then common alternatives
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%m/%d/%Y", "%Y%m%d"):
            try:
                return datetime.strptime(date_val, fmt)
            except ValueError:
                continue
        raise ValueError(f"Cannot parse date: {date_val!r}")
    raise TypeError(f"Expected str or datetime, got {type(date_val).__name__}")


# ---------------------------------------------------------------------------
# Finite-difference derivatives
# ---------------------------------------------------------------------------

def _compute_derivatives(
    times: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute first, second, and third derivatives via central finite differences.

    Parameters
    ----------
    times : np.ndarray
        Monotonically increasing time values (in fractional years).
    values : np.ndarray
        Drift scores at each time point.

    Returns
    -------
    velocity : np.ndarray
        First derivative (d(drift)/dt) at each interior point.
        Has length ``len(values) - 1`` (forward differences).
    acceleration : np.ndarray
        Second derivative.  Length ``len(values) - 2``.
    jerk : np.ndarray
        Third derivative.  Length ``len(values) - 3``.

    Notes
    -----
    We use forward differences for velocity and central differences for
    higher derivatives where possible, falling back to forward/backward
    at boundaries.  All arrays are aligned so that index 0 corresponds to
    the *earliest* computable point.
    """
    n = len(values)

    # Velocity: forward differences dv/dt
    dt = np.diff(times)
    dt = np.where(dt == 0, 1e-6, dt)  # guard against duplicate dates
    dv = np.diff(values)
    velocity = dv / dt  # length n-1

    if n < 3:
        return velocity, np.array([]), np.array([])

    # Acceleration: differences of velocity
    dt_accel = (dt[:-1] + dt[1:]) / 2.0  # midpoint spacing
    dt_accel = np.where(dt_accel == 0, 1e-6, dt_accel)
    acceleration = np.diff(velocity) / dt_accel  # length n-2

    if n < 4:
        return velocity, acceleration, np.array([])

    # Jerk: differences of acceleration
    dt_jerk = (dt_accel[:-1] + dt_accel[1:]) / 2.0
    dt_jerk = np.where(dt_jerk == 0, 1e-6, dt_jerk)
    jerk = np.diff(acceleration) / dt_jerk  # length n-3

    return velocity, acceleration, jerk


def _exponential_moving_average(
    values: np.ndarray,
    halflife: int = _DEFAULT_EMA_HALFLIFE,
) -> np.ndarray:
    """Compute exponential moving average with the given half-life.

    Parameters
    ----------
    values : np.ndarray
        Time series of values (e.g. velocity).
    halflife : int
        Number of periods for the weight to decay by half.

    Returns
    -------
    np.ndarray
        EMA series of the same length as *values*.
    """
    if len(values) == 0:
        return np.array([])

    alpha = 1.0 - math.exp(-math.log(2.0) / max(halflife, 1))
    ema = np.empty_like(values, dtype=np.float64)
    ema[0] = values[0]
    for i in range(1, len(values)):
        ema[i] = alpha * values[i] + (1.0 - alpha) * ema[i - 1]
    return ema


# ---------------------------------------------------------------------------
# Phase classification
# ---------------------------------------------------------------------------

def _classify_phase(
    latest_velocity: float,
    latest_acceleration: float,
    latest_jerk: float,
    velocity_std: float,
) -> str:
    """Classify the company's current disclosure phase.

    Parameters
    ----------
    latest_velocity : float
        Most recent velocity value.
    latest_acceleration : float
        Most recent acceleration value.
    latest_jerk : float
        Most recent jerk value.
    velocity_std : float
        Standard deviation of the full velocity series (used to
        calibrate "volatile" detection).

    Returns
    -------
    str
        One of: stable, drifting, accelerating, decelerating,
        volatile, regime_change.
    """
    abs_vel = abs(latest_velocity)
    abs_accel = abs(latest_acceleration)
    abs_jerk = abs(latest_jerk)

    # Regime change: extreme jerk + high acceleration = sudden structural shift
    if abs_jerk > _REGIME_JERK and abs_accel > _ACCEL_THRESHOLD:
        return "regime_change"

    # Volatile: high jerk or high velocity variance
    if abs_jerk > _JERK_THRESHOLD or velocity_std > _VELOCITY_HIGH:
        return "volatile"

    # Accelerating / decelerating
    if abs_accel > _ACCEL_THRESHOLD:
        return "accelerating" if latest_acceleration > 0 else "decelerating"

    # Drifting: moderate velocity, stable acceleration
    if abs_vel > _VELOCITY_LOW:
        return "drifting"

    return "stable"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SemanticKinematics:
    """Full kinematic analysis of a company's filing drift history.

    Attributes
    ----------
    velocity : list[float]
        First derivative of drift w.r.t. time at each interval.
    acceleration : list[float]
        Second derivative of drift w.r.t. time.
    jerk : list[float]
        Third derivative of drift w.r.t. time.
    momentum : list[float]
        Exponential moving average of velocity (smoothed trend).
    latest_velocity : float
        Most recent velocity value.
    latest_acceleration : float
        Most recent acceleration value.
    latest_jerk : float
        Most recent jerk value.
    latest_momentum : float
        Most recent momentum (EMA of velocity) value.
    phase : str
        Current phase classification: stable, drifting, accelerating,
        decelerating, volatile, or regime_change.
    velocity_mean : float
        Mean velocity across all periods.
    velocity_std : float
        Standard deviation of velocity (measures consistency of change).
    max_velocity : float
        Peak velocity observed in the history.
    periods_analyzed : int
        Number of filing periods in the input.
    filing_dates : list[str]
        ISO-format filing dates from the input series.
    drift_values : list[float]
        Raw drift scores from the input series.
    """

    velocity: list[float]
    acceleration: list[float]
    jerk: list[float]
    momentum: list[float]
    latest_velocity: float
    latest_acceleration: float
    latest_jerk: float
    latest_momentum: float
    phase: str
    velocity_mean: float
    velocity_std: float
    max_velocity: float
    periods_analyzed: int
    filing_dates: list[str] = field(default_factory=list)
    drift_values: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_semantic_kinematics(
    drift_history: list[dict],
    ema_halflife: int = _DEFAULT_EMA_HALFLIFE,
) -> SemanticKinematics:
    """Compute full kinematic analysis of a company's drift trajectory.

    Treats the drift time series as a "position" signal and computes its
    velocity (1st derivative), acceleration (2nd derivative), jerk (3rd
    derivative), and momentum (EMA of velocity).

    Parameters
    ----------
    drift_history : list[dict]
        Chronologically ordered list of dicts, each containing:

        - ``filing_date`` : str or datetime -- when the filing was published.
        - ``cosine_distance`` : float -- semantic drift score for that period.

        Minimum two entries are required for velocity; four for jerk.
    ema_halflife : int
        Half-life (in periods) for the exponential moving average used to
        compute momentum.

    Returns
    -------
    SemanticKinematics
        Full kinematic analysis with derivative series, summary statistics,
        and phase classification.

    Raises
    ------
    ValueError
        If fewer than two data points are provided.

    Algorithm
    ---------
    1. Parse dates and convert to fractional years for uniform time units.
    2. Compute velocity via forward finite differences: v_i = delta_drift / delta_t.
    3. Compute acceleration via second-order differences of velocity.
    4. Compute jerk via third-order differences of acceleration.
    5. Compute momentum as EMA(velocity) with configurable half-life.
    6. Classify the current phase from the latest kinematic state.

    The use of fractional years as the time unit means velocity is measured
    in "drift units per year", making it comparable across companies with
    different filing frequencies (quarterly vs. annual).
    """
    if len(drift_history) < 2:
        raise ValueError(
            f"At least 2 data points required for kinematic analysis, "
            f"got {len(drift_history)}"
        )

    # Sort by date (defensive -- caller should provide sorted, but we ensure it)
    sorted_history = sorted(drift_history, key=lambda d: _parse_date(d["filing_date"]))

    dates = [_parse_date(d["filing_date"]) for d in sorted_history]
    drift_values = np.array(
        [float(d["cosine_distance"]) for d in sorted_history],
        dtype=np.float64,
    )

    # Convert dates to fractional years relative to first date
    base = dates[0]
    times = np.array(
        [(d - base).total_seconds() / (365.25 * 86400) for d in dates],
        dtype=np.float64,
    )

    # Compute derivatives
    velocity, acceleration, jerk = _compute_derivatives(times, drift_values)

    # Momentum (EMA of velocity)
    momentum = _exponential_moving_average(velocity, halflife=ema_halflife)

    # Summary statistics
    vel_mean = float(np.mean(velocity)) if len(velocity) > 0 else 0.0
    vel_std = float(np.std(velocity)) if len(velocity) > 0 else 0.0
    vel_max = float(np.max(np.abs(velocity))) if len(velocity) > 0 else 0.0

    # Latest values (most recent computable point for each derivative)
    latest_vel = float(velocity[-1]) if len(velocity) > 0 else 0.0
    latest_accel = float(acceleration[-1]) if len(acceleration) > 0 else 0.0
    latest_jerk_val = float(jerk[-1]) if len(jerk) > 0 else 0.0
    latest_mom = float(momentum[-1]) if len(momentum) > 0 else 0.0

    # Phase classification
    phase = _classify_phase(latest_vel, latest_accel, latest_jerk_val, vel_std)

    filing_date_strs = [d.strftime("%Y-%m-%d") for d in dates]

    logger.debug(
        "Kinematic analysis (%d periods): vel=%.4f accel=%.4f jerk=%.4f "
        "momentum=%.4f phase=%s",
        len(drift_history), latest_vel, latest_accel, latest_jerk_val,
        latest_mom, phase,
    )

    return SemanticKinematics(
        velocity=[round(float(v), 6) for v in velocity],
        acceleration=[round(float(a), 6) for a in acceleration],
        jerk=[round(float(j), 6) for j in jerk],
        momentum=[round(float(m), 6) for m in momentum],
        latest_velocity=round(latest_vel, 6),
        latest_acceleration=round(latest_accel, 6),
        latest_jerk=round(latest_jerk_val, 6),
        latest_momentum=round(latest_mom, 6),
        phase=phase,
        velocity_mean=round(vel_mean, 6),
        velocity_std=round(vel_std, 6),
        max_velocity=round(vel_max, 6),
        periods_analyzed=len(sorted_history),
        filing_dates=filing_date_strs,
        drift_values=[round(float(d), 6) for d in drift_values],
    )
