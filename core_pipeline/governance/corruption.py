"""
Corruption Injection for Data Quality Governance Experiments

Systematically injects realistic sensor data quality issues into C-MAPSS
data to quantify the impact of data quality on RUL prediction.

5 corruption types × 3 severity levels = 15 experimental scenarios.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class CorruptionType(Enum):
    SENSOR_DRIFT = "sensor_drift"
    RANDOM_MISSING = "random_missing"
    STUCK_AT_FAULT = "stuck_at_fault"
    GAUSSIAN_NOISE = "gaussian_noise"
    CONCEPT_DRIFT = "concept_drift"


class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class CorruptionConfig:
    """Configuration for a single corruption scenario."""
    corruption_type: CorruptionType
    severity: Severity
    target_sensors: list
    params: dict


@dataclass
class CorruptionResult:
    """Result of applying corruption to a dataset."""
    config: CorruptionConfig
    original_df: pd.DataFrame
    corrupted_df: pd.DataFrame
    corruption_mask: pd.DataFrame  # boolean mask of corrupted cells
    summary: dict


# ---------------------------------------------------------------------------
# Severity parameter mappings
# ---------------------------------------------------------------------------

SEVERITY_PARAMS = {
    CorruptionType.SENSOR_DRIFT: {
        Severity.LOW:    {"drift_rate": 0.001},     # 0.1% per cycle
        Severity.MEDIUM: {"drift_rate": 0.005},     # 0.5% per cycle
        Severity.HIGH:   {"drift_rate": 0.02},      # 2% per cycle
    },
    CorruptionType.RANDOM_MISSING: {
        Severity.LOW:    {"missing_rate": 0.05},    # 5%
        Severity.MEDIUM: {"missing_rate": 0.10},    # 10%
        Severity.HIGH:   {"missing_rate": 0.20},    # 20%
    },
    CorruptionType.STUCK_AT_FAULT: {
        Severity.LOW:    {"stuck_pct": 0.05},       # last 5% of cycles stuck
        Severity.MEDIUM: {"stuck_pct": 0.15},       # last 15%
        Severity.HIGH:   {"stuck_pct": 0.30},       # last 30%
    },
    CorruptionType.GAUSSIAN_NOISE: {
        Severity.LOW:    {"snr_db": 20},            # mild noise
        Severity.MEDIUM: {"snr_db": 10},            # moderate noise
        Severity.HIGH:   {"snr_db": 5},             # heavy noise
    },
    CorruptionType.CONCEPT_DRIFT: {
        Severity.LOW:    {"corr_shift": 0.1},       # small relationship change
        Severity.MEDIUM: {"corr_shift": 0.3},       # moderate
        Severity.HIGH:   {"corr_shift": 0.5},       # large
    },
}


# ---------------------------------------------------------------------------
# Corruption Functions
# ---------------------------------------------------------------------------

def inject_sensor_drift(df: pd.DataFrame,
                        sensors: list,
                        drift_rate: float,
                        rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inject linear drift into sensor readings.

    Simulates sensor calibration drift: readings gradually shift
    from true values over time.

    Args:
        df: input DataFrame (will not be modified)
        sensors: sensor columns to corrupt
        drift_rate: drift magnitude per cycle (fraction of sensor std)
        rng: random number generator

    Returns:
        (corrupted_df, mask_df)
    """
    corrupted = df.copy()
    mask = pd.DataFrame(False, index=df.index, columns=df.columns)

    for engine_id in corrupted["engine_id"].unique():
        engine_mask = corrupted["engine_id"] == engine_id
        n_cycles = engine_mask.sum()

        for sensor in sensors:
            sensor_std = corrupted.loc[engine_mask, sensor].std()
            if sensor_std == 0:
                continue

            # Linear drift: direction random per engine-sensor pair
            direction = rng.choice([-1, 1])
            drift = direction * drift_rate * sensor_std * np.arange(n_cycles)
            corrupted.loc[engine_mask, sensor] += drift
            mask.loc[engine_mask, sensor] = True

    return corrupted, mask


def inject_random_missing(df: pd.DataFrame,
                          sensors: list,
                          missing_rate: float,
                          rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inject Missing Completely At Random (MCAR) into sensor readings.

    Simulates communication failures and sensor intermittent faults.
    """
    corrupted = df.copy()
    mask = pd.DataFrame(False, index=df.index, columns=df.columns)

    for sensor in sensors:
        missing_mask = rng.random(len(corrupted)) < missing_rate
        corrupted.loc[missing_mask, sensor] = np.nan
        mask.loc[missing_mask, sensor] = True

    return corrupted, mask


def inject_stuck_at_fault(df: pd.DataFrame,
                          sensors: list,
                          stuck_pct: float,
                          rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inject stuck-at-fault: sensor value freezes at last valid reading.

    Most common real sensor failure mode. Value stops changing
    while the underlying process continues to degrade.
    """
    corrupted = df.copy()
    mask = pd.DataFrame(False, index=df.index, columns=df.columns)

    for engine_id in corrupted["engine_id"].unique():
        engine_mask = corrupted["engine_id"] == engine_id
        n_cycles = engine_mask.sum()
        stuck_start = int(n_cycles * (1 - stuck_pct))

        engine_idx = corrupted.index[engine_mask]

        for sensor in sensors:
            # Freeze at the value just before stuck_start
            freeze_value = corrupted.loc[engine_idx[stuck_start - 1], sensor]
            stuck_idx = engine_idx[stuck_start:]
            corrupted.loc[stuck_idx, sensor] = freeze_value
            mask.loc[stuck_idx, sensor] = True

    return corrupted, mask


def inject_gaussian_noise(df: pd.DataFrame,
                          sensors: list,
                          snr_db: float,
                          rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inject Gaussian noise at a specified SNR level.

    Simulates electromagnetic interference and grounding issues.
    """
    corrupted = df.copy()
    mask = pd.DataFrame(False, index=df.index, columns=df.columns)

    for sensor in sensors:
        signal = corrupted[sensor].values
        signal_power = np.mean(signal ** 2)

        if signal_power == 0:
            continue

        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = rng.normal(0, np.sqrt(noise_power), len(signal))
        corrupted[sensor] = signal + noise
        mask[sensor] = True

    return corrupted, mask


def inject_concept_drift(df: pd.DataFrame,
                         sensors: list,
                         corr_shift: float,
                         rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inject concept drift by altering cross-sensor relationships.

    Simulates equipment replacement or major maintenance that
    changes the physical relationships between sensors.
    Applied to second half of each engine's lifecycle.
    """
    corrupted = df.copy()
    mask = pd.DataFrame(False, index=df.index, columns=df.columns)

    if len(sensors) < 2:
        return corrupted, mask

    for engine_id in corrupted["engine_id"].unique():
        engine_mask = corrupted["engine_id"] == engine_id
        n_cycles = engine_mask.sum()
        drift_start = n_cycles // 2

        engine_idx = corrupted.index[engine_mask]
        drift_idx = engine_idx[drift_start:]

        # Mix sensor pairs to break correlations
        for i in range(0, len(sensors) - 1, 2):
            s1, s2 = sensors[i], sensors[i + 1]
            vals1 = corrupted.loc[drift_idx, s1].values.copy()
            vals2 = corrupted.loc[drift_idx, s2].values.copy()

            # Blend: new_s1 = (1-shift)*s1 + shift*s2
            corrupted.loc[drift_idx, s1] = (1 - corr_shift) * vals1 + corr_shift * vals2
            corrupted.loc[drift_idx, s2] = (1 - corr_shift) * vals2 + corr_shift * vals1
            mask.loc[drift_idx, s1] = True
            mask.loc[drift_idx, s2] = True

    return corrupted, mask


# ---------------------------------------------------------------------------
# Corruption Dispatcher
# ---------------------------------------------------------------------------

CORRUPTION_FUNCTIONS = {
    CorruptionType.SENSOR_DRIFT: inject_sensor_drift,
    CorruptionType.RANDOM_MISSING: inject_random_missing,
    CorruptionType.STUCK_AT_FAULT: inject_stuck_at_fault,
    CorruptionType.GAUSSIAN_NOISE: inject_gaussian_noise,
    CorruptionType.CONCEPT_DRIFT: inject_concept_drift,
}


def apply_corruption(df: pd.DataFrame,
                     corruption_type: CorruptionType,
                     severity: Severity,
                     target_sensors: Optional[list] = None,
                     seed: int = 42) -> CorruptionResult:
    """Apply a single corruption scenario to the dataset.

    Args:
        df: input DataFrame
        corruption_type: type of corruption
        severity: severity level
        target_sensors: sensors to corrupt (default: DEGRADATION_SENSORS)
        seed: random seed for reproducibility

    Returns:
        CorruptionResult with original, corrupted data, and mask
    """
    from core_pipeline.data.preprocess import DEGRADATION_SENSORS

    if target_sensors is None:
        target_sensors = DEGRADATION_SENSORS

    rng = np.random.default_rng(seed)
    params = SEVERITY_PARAMS[corruption_type][severity]

    func = CORRUPTION_FUNCTIONS[corruption_type]
    corrupted, corruption_mask = func(df, target_sensors, **params, rng=rng)

    n_corrupted = corruption_mask.sum().sum()
    n_total = len(df) * len(target_sensors)

    config = CorruptionConfig(
        corruption_type=corruption_type,
        severity=severity,
        target_sensors=target_sensors,
        params=params,
    )

    return CorruptionResult(
        config=config,
        original_df=df,
        corrupted_df=corrupted,
        corruption_mask=corruption_mask,
        summary={
            "type": corruption_type.value,
            "severity": severity.value,
            "n_sensors_targeted": len(target_sensors),
            "n_cells_corrupted": int(n_corrupted),
            "corruption_rate": round(n_corrupted / n_total, 4) if n_total > 0 else 0,
            "params": params,
        },
    )


def run_all_scenarios(df: pd.DataFrame,
                      target_sensors: Optional[list] = None,
                      seed: int = 42) -> list[CorruptionResult]:
    """Run all 15 corruption scenarios (5 types × 3 severities).

    Returns:
        list of 15 CorruptionResult objects
    """
    results = []
    for ctype in CorruptionType:
        for severity in Severity:
            result = apply_corruption(
                df, ctype, severity,
                target_sensors=target_sensors,
                seed=seed,
            )
            print(f"  {ctype.value:20s} | {severity.value:6s} | "
                  f"corrupted {result.summary['corruption_rate']:.1%} of cells")
            results.append(result)

    return results


def scenarios_summary_table(results: list[CorruptionResult]) -> pd.DataFrame:
    """Create a summary table of all corruption scenarios."""
    rows = []
    for r in results:
        rows.append({
            "type": r.summary["type"],
            "severity": r.summary["severity"],
            "n_cells_corrupted": r.summary["n_cells_corrupted"],
            "corruption_rate": r.summary["corruption_rate"],
            **r.summary["params"],
        })
    return pd.DataFrame(rows)
