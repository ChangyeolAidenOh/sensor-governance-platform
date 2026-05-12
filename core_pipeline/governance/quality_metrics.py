"""
Data Quality Metrics for Sensor Health Scoring

Three dimensions of sensor data quality:
1. Completeness  - missing rate per sensor per window
2. Stability     - distributional stability (PSI) between windows
3. Consistency   - cross-sensor correlation stability
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy import stats
from typing import Optional


@dataclass
class SensorHealthScore:
    """Health score for a single sensor in a given window."""
    sensor: str
    completeness: float   # 0-100
    stability: float      # 0-100
    consistency: float    # 0-100
    composite: float      # 0-100 (weighted)
    details: dict         # raw metric values


# ---------------------------------------------------------------------------
# 1. Completeness
# ---------------------------------------------------------------------------

def compute_completeness(series: pd.Series) -> float:
    """Compute completeness score (0-100) for a sensor window.

    100 = no missing values, 0 = all missing.
    """
    if len(series) == 0:
        return 0.0
    missing_rate = series.isna().mean()
    return round((1.0 - missing_rate) * 100, 2)


# ---------------------------------------------------------------------------
# 2. Stability (PSI - Population Stability Index)
# ---------------------------------------------------------------------------

def compute_psi(reference: np.ndarray,
                current: np.ndarray,
                n_bins: int = 10,
                eps: float = 1e-4) -> float:
    """Compute Population Stability Index between two distributions.

    PSI < 0.1  : no significant change
    PSI 0.1-0.2: moderate change
    PSI > 0.2  : significant change

    Args:
        reference: baseline distribution (previous window)
        current: current distribution
        n_bins: number of bins for discretization
        eps: smoothing constant to avoid log(0)

    Returns:
        PSI value (lower is more stable)
    """
    ref_clean = reference[~np.isnan(reference)]
    cur_clean = current[~np.isnan(current)]

    if len(ref_clean) < 2 or len(cur_clean) < 2:
        return 1.0  # insufficient data = max instability

    # Create bins from reference distribution
    breakpoints = np.percentile(ref_clean, np.linspace(0, 100, n_bins + 1))
    breakpoints = np.unique(breakpoints)

    if len(breakpoints) < 2:
        return 0.0  # constant signal

    ref_counts = np.histogram(ref_clean, bins=breakpoints)[0]
    cur_counts = np.histogram(cur_clean, bins=breakpoints)[0]

    # Normalize to proportions
    ref_sum = ref_counts.sum()
    cur_sum = cur_counts.sum()

    if ref_sum == 0 or cur_sum == 0:
        return 1.0  # insufficient data = max instability

    ref_pct = ref_counts / ref_sum + eps
    cur_pct = cur_counts / cur_sum + eps

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return round(float(psi), 6)


def psi_to_score(psi: float) -> float:
    """Convert PSI value to 0-100 stability score.

    PSI=0 -> 100, PSI>=0.25 -> 0, linear between.
    """
    if psi <= 0:
        return 100.0
    if psi >= 0.25:
        return 0.0
    return round((1.0 - psi / 0.25) * 100, 2)


# ---------------------------------------------------------------------------
# 3. Consistency (cross-sensor correlation stability)
# ---------------------------------------------------------------------------

def compute_correlation_deviation(current_corr: pd.DataFrame,
                                  reference_corr: pd.DataFrame,
                                  sensor: str) -> float:
    """Compute how much a sensor's correlations with other sensors
    have changed relative to the reference period.

    Returns:
        Mean absolute deviation of correlation coefficients (0-1 scale)
    """
    if sensor not in current_corr.columns or sensor not in reference_corr.columns:
        return 0.5  # unknown = moderate risk

    ref_row = reference_corr[sensor].drop(sensor, errors="ignore")
    cur_row = current_corr[sensor].drop(sensor, errors="ignore")

    # Align indices
    common = ref_row.index.intersection(cur_row.index)
    if len(common) == 0:
        return 0.5

    deviation = (ref_row[common] - cur_row[common]).abs().mean()
    return round(float(deviation), 4)


def correlation_deviation_to_score(deviation: float,
                                   max_deviation: float = 0.3) -> float:
    """Convert correlation deviation to 0-100 consistency score.

    deviation=0 -> 100, deviation>=max_deviation -> 0.
    """
    if deviation <= 0:
        return 100.0
    if deviation >= max_deviation:
        return 0.0
    return round((1.0 - deviation / max_deviation) * 100, 2)


# ---------------------------------------------------------------------------
# Composite Score
# ---------------------------------------------------------------------------

def compute_sensor_health(sensor: str,
                          current_window: pd.Series,
                          reference_window: pd.Series,
                          current_corr: pd.DataFrame,
                          reference_corr: pd.DataFrame,
                          weights: tuple = (0.3, 0.4, 0.3)) -> SensorHealthScore:
    """Compute composite health score for one sensor.

    Args:
        sensor: sensor column name
        current_window: current window values
        reference_window: reference (baseline) window values
        current_corr: correlation matrix of current window
        reference_corr: correlation matrix of reference window
        weights: (completeness, stability, consistency) weights

    Returns:
        SensorHealthScore dataclass
    """
    w_comp, w_stab, w_cons = weights

    # Completeness
    completeness = compute_completeness(current_window)

    # Stability (PSI)
    psi_val = compute_psi(reference_window.values, current_window.values)
    stability = psi_to_score(psi_val)

    # Consistency
    corr_dev = compute_correlation_deviation(
        current_corr, reference_corr, sensor
    )
    consistency = correlation_deviation_to_score(corr_dev)

    # Composite
    composite = round(
        w_comp * completeness + w_stab * stability + w_cons * consistency, 2
    )

    return SensorHealthScore(
        sensor=sensor,
        completeness=completeness,
        stability=stability,
        consistency=consistency,
        composite=composite,
        details={
            "psi": psi_val,
            "correlation_deviation": corr_dev,
            "missing_rate": round(1 - completeness / 100, 4),
        },
    )


def compute_all_sensor_health(df_current: pd.DataFrame,
                              df_reference: pd.DataFrame,
                              sensor_cols: list,
                              weights: tuple = (0.3, 0.4, 0.3)
                              ) -> list[SensorHealthScore]:
    """Compute health scores for all sensors.

    Args:
        df_current: current window data
        df_reference: reference (baseline) window data
        sensor_cols: list of sensor column names
        weights: metric weights

    Returns:
        list of SensorHealthScore for each sensor
    """
    current_corr = df_current[sensor_cols].corr()
    reference_corr = df_reference[sensor_cols].corr()

    scores = []
    for sensor in sensor_cols:
        score = compute_sensor_health(
            sensor=sensor,
            current_window=df_current[sensor],
            reference_window=df_reference[sensor],
            current_corr=current_corr,
            reference_corr=reference_corr,
            weights=weights,
        )
        scores.append(score)

    return scores


def health_scores_to_dataframe(scores: list[SensorHealthScore]) -> pd.DataFrame:
    """Convert health scores to a summary DataFrame."""
    records = []
    for s in scores:
        records.append({
            "sensor": s.sensor,
            "completeness": s.completeness,
            "stability": s.stability,
            "consistency": s.consistency,
            "composite": s.composite,
            "psi": s.details["psi"],
            "corr_deviation": s.details["correlation_deviation"],
            "missing_rate": s.details["missing_rate"],
        })
    return pd.DataFrame(records).sort_values("composite")


# ---------------------------------------------------------------------------
# Windowed Health Monitoring
# ---------------------------------------------------------------------------

def monitor_engine_health(engine_df: pd.DataFrame,
                          sensor_cols: list,
                          window_size: int = 20,
                          reference_size: int = 30,
                          weights: tuple = (0.3, 0.4, 0.3)
                          ) -> pd.DataFrame:
    """Compute rolling health scores across an engine's lifecycle.

    Uses the first reference_size cycles as baseline, then slides
    a window of window_size across remaining cycles.

    Args:
        engine_df: single engine data sorted by cycle
        sensor_cols: sensor columns
        window_size: size of evaluation window
        reference_size: size of reference (baseline) window
        weights: metric weights

    Returns:
        DataFrame with health scores per window
    """
    engine_df = engine_df.sort_values("cycle").reset_index(drop=True)

    if len(engine_df) < reference_size + window_size:
        return pd.DataFrame()

    reference = engine_df.iloc[:reference_size]
    ref_corr = reference[sensor_cols].corr()

    results = []
    for start in range(reference_size, len(engine_df) - window_size + 1, window_size):
        window = engine_df.iloc[start:start + window_size]
        win_corr = window[sensor_cols].corr()

        scores = []
        for sensor in sensor_cols:
            score = compute_sensor_health(
                sensor=sensor,
                current_window=window[sensor],
                reference_window=reference[sensor],
                current_corr=win_corr,
                reference_corr=ref_corr,
                weights=weights,
            )
            scores.append(score)

        avg_composite = np.mean([s.composite for s in scores])
        min_composite = min(s.composite for s in scores)
        flagged = [s.sensor for s in scores if s.composite < 70]

        results.append({
            "window_start": int(window["cycle"].iloc[0]),
            "window_end": int(window["cycle"].iloc[-1]),
            "avg_health": round(avg_composite, 2),
            "min_health": round(min_composite, 2),
            "n_flagged": len(flagged),
            "flagged_sensors": flagged,
        })

    return pd.DataFrame(results)
