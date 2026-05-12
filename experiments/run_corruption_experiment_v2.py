"""
Corruption Impact Experiment v2 — Type-Aware + Severity-Aware Remediation
v1 문제: 모든 corruption에 forward fill 단일 전략 → recovery 0% 또는 음수
v2 변경: corruption type별 올바른 remediation 전략 분기
v2 revision: Change the objects of corruption for all the sensors of 14.
--This is not natural case in the real world. Hence, it would be focused on the case for detection of 1 to 3 malfunction sensors.

전략 매핑:
  random_missing  → PASSTHROUGH (XGBoost native NaN handling이 더 우수)
  sensor_drift    → SENSOR_DROP (flagged sensor를 feature set에서 제거 후 재학습)
  stuck_at_fault  → SENSOR_DROP (동일)
  gaussian_noise  → SMOOTHING (이동평균 필터 적용)
  concept_drift   → RETRAIN_ALERT (기존 모델 사용 금지 — Detection Accuracy로 측정)

v1: forward fill all → recovery 0%
v2: type-aware remediation → sensor_drop 42.76 폭발 (14개 전부 drop)
v3: N_CORRUPT_SENSORS=4 → sensor_drop low/med 역효과 (-70%~-332%)
v3+: severity-aware → 센서별 PSI 체크, 심한 센서만 drop, 경미한 센서 유지
Next: Add Variance-based detection step in "remediate_smart_sensor_drop" function
==> Variance detection was ineffective, then successive trial: change the strategy of stuck_at_fault from smart_sensor_drop to passthrough at REMEDIATION_MAP

핵심 원칙: "개입 비용 > corruption 피해"이면 개입하지 않는다.
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core_pipeline.data.preprocess import (
    preprocess_subset, DEGRADATION_SENSORS, SENSOR_COLS,
)
from core_pipeline.governance.corruption import (
    CorruptionType, Severity, apply_corruption,
)
from core_pipeline.governance.quality_metrics import (
    compute_psi, compute_correlation_deviation,
)
from core_pipeline.rul.xgboost_rul import train_xgboost_rul, evaluate_rul


# Number of sensors to corrupt per scenario
N_CORRUPT_SENSORS = 4

# PSI threshold for smart sensor drop
# PSI > this → sensor is severely corrupted → drop
# PSI <= this → corruption is mild → keep sensor
PSI_DROP_THRESHOLD = 0.2


# ---------------------------------------------------------------------------
# Type-Aware Remediation Functions
# ---------------------------------------------------------------------------

def remediate_passthrough(corrupted_df: pd.DataFrame,
                          target_sensors: list,
                          **kwargs) -> pd.DataFrame:
    """PASSTHROUGH: do nothing. Let XGBoost handle NaN natively."""
    return corrupted_df.copy()


# Variance ratio threshold for stuck-at detection
# var(corrupted) / var(reference) < this → sensor likely stuck → drop
VARIANCE_RATIO_THRESHOLD = 0.5


def remediate_smart_sensor_drop(corrupted_df: pd.DataFrame,
                                 target_sensors: list,
                                 reference_df: pd.DataFrame = None,
                                 psi_threshold: float = PSI_DROP_THRESHOLD,
                                 var_ratio_threshold: float = VARIANCE_RATIO_THRESHOLD,
                                 **kwargs) -> tuple[pd.DataFrame, list]:
    """SMART SENSOR DROP: drop sensors failing PSI or variance check.

    Per-sensor dual check:
      PSI > threshold         → distributional shift (drift) → drop
      variance ratio < threshold → signal frozen (stuck-at)  → drop
      Neither                 → mild corruption → keep

    Returns:
        (remediated_df, list of (sensor, reason) tuples for dropped sensors)
    """
    df = corrupted_df.copy()
    dropped = []

    for sensor in target_sensors:
        if sensor not in df.columns or reference_df is None:
            continue

        should_drop = False
        reason = ""

        # Check 1: PSI (distributional shift — catches drift, noise)
        psi = compute_psi(
            reference_df[sensor].dropna().values,
            df[sensor].dropna().values,
        )
        if psi > psi_threshold:
            should_drop = True
            reason = f"psi={psi:.3f}"

        # Check 2: Variance ratio (catches stuck-at-fault)
        ref_var = reference_df[sensor].var()
        cur_var = df[sensor].var()
        if ref_var > 0:
            var_ratio = cur_var / ref_var
            if var_ratio < var_ratio_threshold:
                should_drop = True
                reason = f"var_ratio={var_ratio:.3f}" if not reason else f"{reason},var_ratio={var_ratio:.3f}"

        if should_drop:
            cols_to_drop = [sensor]
            cols_to_drop += [c for c in df.columns
                             if c.startswith(f"{sensor}_rmean_")
                             or c.startswith(f"{sensor}_rstd_")]
            cols_to_drop = [c for c in cols_to_drop if c in df.columns]
            df = df.drop(columns=cols_to_drop)
            dropped.append(sensor)

    return df, dropped


def remediate_smoothing(corrupted_df: pd.DataFrame,
                         target_sensors: list,
                         window: int = 5,
                         **kwargs) -> pd.DataFrame:
    """SMOOTHING: apply rolling mean to denoise sensor readings."""
    df = corrupted_df.copy()
    df = df.sort_values(["engine_id", "cycle"])

    for sensor in target_sensors:
        if sensor not in df.columns:
            continue
        df[sensor] = df.groupby("engine_id")[sensor].transform(
            lambda x: x.rolling(window, min_periods=1, center=True).mean()
        )

    return df


def detect_concept_drift(corrupted_df: pd.DataFrame,
                          clean_df: pd.DataFrame,
                          target_sensors: list,
                          psi_threshold: float = 0.2,
                          corr_threshold: float = 0.15) -> dict:
    """RETRAIN_ALERT: detect concept drift and report."""
    detections = {"psi_alerts": [], "corr_alerts": [], "detected": False}

    for sensor in target_sensors:
        if sensor not in corrupted_df.columns:
            continue
        psi = compute_psi(
            clean_df[sensor].values,
            corrupted_df[sensor].values,
        )
        if psi > psi_threshold:
            detections["psi_alerts"].append({"sensor": sensor, "psi": psi})

    clean_corr = clean_df[target_sensors].corr()
    corrupt_corr = corrupted_df[target_sensors].corr()

    for sensor in target_sensors:
        dev = compute_correlation_deviation(corrupt_corr, clean_corr, sensor)
        if dev > corr_threshold:
            detections["corr_alerts"].append({"sensor": sensor, "deviation": dev})

    detections["detected"] = (
        len(detections["psi_alerts"]) > 0
        or len(detections["corr_alerts"]) > 0
    )
    detections["n_psi_alerts"] = len(detections["psi_alerts"])
    detections["n_corr_alerts"] = len(detections["corr_alerts"])

    return detections


# Strategy mapping
REMEDIATION_MAP = {
    CorruptionType.RANDOM_MISSING: "passthrough",
    CorruptionType.SENSOR_DRIFT: "smart_sensor_drop",
    CorruptionType.STUCK_AT_FAULT: "passthrough",
    CorruptionType.GAUSSIAN_NOISE: "smoothing",
    CorruptionType.CONCEPT_DRIFT: "retrain_alert",
}


# ---------------------------------------------------------------------------
# Experiment Runner
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    """Result of one corruption experiment scenario."""
    corruption_type: str
    severity: str
    remediation_strategy: str
    n_sensors_corrupted: int
    n_sensors_dropped: int
    rmse_clean: float
    rmse_corrupt: float
    rmse_recovered: float
    delta_rmse: float
    recovery_rate: float
    score_clean: float
    score_corrupt: float
    score_recovered: float
    concept_drift_detected: bool
    dropped_sensors: str  # comma-separated
    params: dict
    elapsed_sec: float


def run_single_scenario(data: dict,
                        corruption_type: CorruptionType,
                        severity: Severity,
                        baseline_metrics: dict,
                        max_rul: int = 125,
                        seed: int = 42) -> ExperimentResult:
    """Run one scenario with type-aware + severity-aware remediation."""
    t0 = time.time()
    train_df = data["train"].copy()

    strategy_name = REMEDIATION_MAP[corruption_type]

    # Select random subset of sensors to corrupt
    rng = np.random.default_rng(seed)
    n = min(N_CORRUPT_SENSORS, len(DEGRADATION_SENSORS))
    if corruption_type == CorruptionType.CONCEPT_DRIFT and n % 2 != 0:
        n = max(2, n - 1)
    corrupt_sensors = sorted(rng.choice(
        DEGRADATION_SENSORS, size=n, replace=False
    ).tolist())

    # --- Step 1: Apply corruption ---
    corruption_result = apply_corruption(
        train_df, corruption_type, severity,
        target_sensors=corrupt_sensors, seed=seed,
    )
    corrupted_df = corruption_result.corrupted_df

    # --- Step 2: Train on corrupted data (no governance) ---
    corrupt_model = train_xgboost_rul(corrupted_df, max_rul=max_rul)
    corrupt_metrics = evaluate_rul(
        corrupt_model["model"], data["test"], data["rul_true"],
        corrupt_model["feature_cols"], max_rul,
    )

    # --- Step 3: Type-aware + severity-aware remediation ---
    concept_drift_detected = False
    dropped_sensors = []

    if strategy_name == "retrain_alert":
        detection = detect_concept_drift(
            corrupted_df, train_df, corrupt_sensors,
        )
        concept_drift_detected = detection["detected"]

        if concept_drift_detected:
            rmse_recovered = baseline_metrics["rmse"]
            score_recovered = baseline_metrics["score"]
        else:
            rmse_recovered = corrupt_metrics["rmse"]
            score_recovered = corrupt_metrics["score"]

    elif strategy_name == "smart_sensor_drop":
        # Severity-aware: only drop sensors with PSI > threshold
        remediated_df, dropped_sensors = remediate_smart_sensor_drop(
            corrupted_df, corrupt_sensors,
            reference_df=train_df,
            psi_threshold=PSI_DROP_THRESHOLD,
        )

        if len(dropped_sensors) == 0:
            # No sensors severe enough to drop → passthrough
            remediated_df = corrupted_df.copy()

        governed_model = train_xgboost_rul(remediated_df, max_rul=max_rul)

        test_df = data["test"].copy()
        if len(dropped_sensors) > 0:
            drop_cols = [c for c in train_df.columns
                         if c not in remediated_df.columns
                         and c in test_df.columns]
            test_df = test_df.drop(columns=drop_cols)

        governed_metrics = evaluate_rul(
            governed_model["model"], test_df, data["rul_true"],
            governed_model["feature_cols"], max_rul,
        )
        rmse_recovered = governed_metrics["rmse"]
        score_recovered = governed_metrics["score"]

    elif strategy_name == "smoothing":
        remediated_df = remediate_smoothing(corrupted_df, corrupt_sensors)
        governed_model = train_xgboost_rul(remediated_df, max_rul=max_rul)
        governed_metrics = evaluate_rul(
            governed_model["model"], data["test"], data["rul_true"],
            governed_model["feature_cols"], max_rul,
        )
        rmse_recovered = governed_metrics["rmse"]
        score_recovered = governed_metrics["score"]

    else:  # passthrough
        rmse_recovered = corrupt_metrics["rmse"]
        score_recovered = corrupt_metrics["score"]

    # --- Compute Recovery Rate ---
    rmse_clean = baseline_metrics["rmse"]
    rmse_corrupt = corrupt_metrics["rmse"]
    delta = rmse_corrupt - rmse_clean

    if delta > 0:
        recovery = (rmse_corrupt - rmse_recovered) / delta
    else:
        recovery = 0.0

    elapsed = time.time() - t0

    return ExperimentResult(
        corruption_type=corruption_type.value,
        severity=severity.value,
        remediation_strategy=strategy_name,
        n_sensors_corrupted=len(corrupt_sensors),
        n_sensors_dropped=len(dropped_sensors),
        rmse_clean=rmse_clean,
        rmse_corrupt=round(rmse_corrupt, 4),
        rmse_recovered=round(rmse_recovered, 4),
        delta_rmse=round(delta, 4),
        recovery_rate=round(recovery, 4),
        score_clean=baseline_metrics["score"],
        score_corrupt=round(corrupt_metrics["score"], 2),
        score_recovered=round(score_recovered, 2),
        concept_drift_detected=concept_drift_detected,
        dropped_sensors=",".join(dropped_sensors),
        params=corruption_result.summary["params"],
        elapsed_sec=round(elapsed, 1),
    )


def run_full_experiment(data_dir: str = "data/raw/CMAPSSData",
                        subset: str = "FD001",
                        max_rul: int = 125,
                        output_dir: str = "experiments",
                        seed: int = 42) -> pd.DataFrame:
    """Run all 15 scenarios with type + severity aware remediation."""
    print(f"=== Corruption Impact Experiment ({subset}) ===")
    print(f"    Type-aware + severity-aware (PSI threshold={PSI_DROP_THRESHOLD})")
    print(f"    N_CORRUPT_SENSORS={N_CORRUPT_SENSORS}\n")

    # Load and preprocess
    print("Loading data...")
    data = preprocess_subset(data_dir, subset, max_rul)

    # Train clean baseline
    print("Training clean baseline...")
    baseline_model = train_xgboost_rul(data["train"], max_rul=max_rul)
    baseline_metrics = evaluate_rul(
        baseline_model["model"], data["test"], data["rul_true"],
        baseline_model["feature_cols"], max_rul,
    )
    print(f"  Baseline RMSE: {baseline_metrics['rmse']}")
    print(f"  Baseline Score: {baseline_metrics['score']}\n")

    # Run all scenarios
    results = []
    for ctype in CorruptionType:
        for severity in Severity:
            strategy = REMEDIATION_MAP[ctype]
            print(f"Running: {ctype.value} / {severity.value} "
                  f"[{strategy}]...")

            result = run_single_scenario(
                data, ctype, severity, baseline_metrics,
                max_rul=max_rul, seed=seed,
            )

            extra = ""
            if ctype == CorruptionType.CONCEPT_DRIFT:
                extra = f", detected={result.concept_drift_detected}"
            if result.n_sensors_dropped > 0:
                extra += f", dropped={result.n_sensors_dropped}/{result.n_sensors_corrupted}"

            print(f"  RMSE: {result.rmse_clean} -> {result.rmse_corrupt} "
                  f"-> {result.rmse_recovered} "
                  f"(recovery: {result.recovery_rate:.1%}"
                  f"{extra}, {result.elapsed_sec}s)")
            results.append(result)

    # Build results table
    results_df = pd.DataFrame([vars(r) for r in results])

    # Save
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    csv_name = f"corruption_experiment_{subset}.csv"
    results_df.to_csv(out_path / csv_name, index=False)

    json_name = f"corruption_experiment_{subset}.json"
    with open(out_path / json_name, "w") as f:
        json.dump({
            "subset": subset,
            "version": "type_severity_aware",
            "max_rul": max_rul,
            "n_corrupt_sensors": N_CORRUPT_SENSORS,
            "psi_drop_threshold": PSI_DROP_THRESHOLD,
            "baseline_rmse": baseline_metrics["rmse"],
            "baseline_score": baseline_metrics["score"],
            "n_scenarios": len(results),
            "results": [vars(r) for r in results],
        }, f, indent=2)

    # Print summary
    print(f"\n=== RESULTS SUMMARY ===\n")
    summary_cols = ["corruption_type", "severity", "remediation_strategy",
                    "n_sensors_dropped",
                    "rmse_corrupt", "rmse_recovered",
                    "delta_rmse", "recovery_rate"]
    print(results_df[summary_cols].to_string(index=False))

    # Key findings
    print("\n=== KEY FINDINGS ===")

    non_cd = results_df[results_df["corruption_type"] != "concept_drift"]
    cd = results_df[results_df["corruption_type"] == "concept_drift"]

    if len(non_cd) > 0:
        positive = non_cd[non_cd["recovery_rate"] > 0]
        negative = non_cd[non_cd["recovery_rate"] < 0]
        neutral = non_cd[non_cd["recovery_rate"] == 0]

        print(f"  Positive recovery: {len(positive)}/{len(non_cd)} scenarios")
        print(f"  Neutral (0%):      {len(neutral)}/{len(non_cd)} scenarios")
        print(f"  Negative (harm):   {len(negative)}/{len(non_cd)} scenarios")

        if len(positive) > 0:
            avg_pos = positive["recovery_rate"].mean()
            best = positive.loc[positive["recovery_rate"].idxmax()]
            print(f"  Avg recovery (positive): {avg_pos:.1%}")
            print(f"  Best: {best['corruption_type']} / "
                  f"{best['severity']} ({best['recovery_rate']:.1%})")

    if len(cd) > 0:
        detected = cd["concept_drift_detected"].sum()
        print(f"  Concept drift detection: {int(detected)}/{len(cd)}")

    return results_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Corruption impact experiment (type + severity aware)"
    )
    parser.add_argument("--data-dir", default="data/raw/CMAPSSData")
    parser.add_argument("--subset", default="FD001")
    parser.add_argument("--max-rul", type=int, default=125)
    parser.add_argument("--output-dir", default="experiments")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_full_experiment(
        args.data_dir, args.subset, args.max_rul,
        args.output_dir, args.seed,
    )
