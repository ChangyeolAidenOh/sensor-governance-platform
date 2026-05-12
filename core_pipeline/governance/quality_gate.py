"""
Quality Gate — pass/flag decision logic for sensor data.

Sits between raw ingestion and the analytics engine.
Data that fails the gate is flagged, logged, and optionally imputed
before entering the model pipeline.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from core_pipeline.governance.quality_metrics import (
    compute_all_sensor_health,
    health_scores_to_dataframe,
    SensorHealthScore,
)


class GateStatus(Enum):
    PASS = "pass"
    FLAG = "flag"
    BLOCK = "block"


class ImputationStrategy(Enum):
    FORWARD_FILL = "forward_fill"
    LINEAR_INTERPOLATION = "linear_interpolation"
    MEDIAN_FILL = "median_fill"
    MASK = "mask"  # drop sensor from feature set


@dataclass
class GateResult:
    """Result of quality gate evaluation."""
    status: GateStatus
    overall_score: float
    flagged_sensors: list[str]
    blocked_sensors: list[str]
    sensor_scores: pd.DataFrame
    actions_taken: list[dict] = field(default_factory=list)


class QualityGate:
    """Evaluates sensor data quality and decides pass/flag/block.

    Thresholds:
        composite >= flag_threshold  → PASS
        block_threshold <= composite < flag_threshold → FLAG (impute + warn)
        composite < block_threshold  → BLOCK (exclude sensor)
    """

    def __init__(self,
                 flag_threshold: float = 70.0,
                 block_threshold: float = 30.0,
                 weights: tuple = (0.3, 0.4, 0.3),
                 imputation_strategy: ImputationStrategy = ImputationStrategy.FORWARD_FILL):
        self.flag_threshold = flag_threshold
        self.block_threshold = block_threshold
        self.weights = weights
        self.imputation_strategy = imputation_strategy
        self.history: list[GateResult] = []

    def evaluate(self,
                 df_current: pd.DataFrame,
                 df_reference: pd.DataFrame,
                 sensor_cols: list) -> GateResult:
        """Evaluate data quality for the current window.

        Args:
            df_current: current window data
            df_reference: reference (baseline) window data
            sensor_cols: sensor columns to evaluate

        Returns:
            GateResult with status and per-sensor breakdown
        """
        scores = compute_all_sensor_health(
            df_current, df_reference, sensor_cols, self.weights
        )
        scores_df = health_scores_to_dataframe(scores)

        flagged = scores_df[
            (scores_df["composite"] < self.flag_threshold) &
            (scores_df["composite"] >= self.block_threshold)
        ]["sensor"].tolist()

        blocked = scores_df[
            scores_df["composite"] < self.block_threshold
        ]["sensor"].tolist()

        overall = scores_df["composite"].mean()

        if len(blocked) > 0:
            status = GateStatus.BLOCK
        elif len(flagged) > 0:
            status = GateStatus.FLAG
        else:
            status = GateStatus.PASS

        result = GateResult(
            status=status,
            overall_score=round(overall, 2),
            flagged_sensors=flagged,
            blocked_sensors=blocked,
            sensor_scores=scores_df,
        )

        self.history.append(result)
        return result

    def apply_remediation(self,
                          df: pd.DataFrame,
                          gate_result: GateResult) -> pd.DataFrame:
        """Apply imputation or masking based on gate result.

        Args:
            df: data to remediate
            gate_result: output of evaluate()

        Returns:
            remediated DataFrame
        """
        df = df.copy()

        # Blocked sensors: mask (set to NaN or drop)
        for sensor in gate_result.blocked_sensors:
            df[sensor] = np.nan
            gate_result.actions_taken.append({
                "sensor": sensor,
                "action": "masked",
                "reason": f"composite score below {self.block_threshold}",
            })

        # Flagged sensors: impute
        for sensor in gate_result.flagged_sensors:
            df = self._impute_sensor(df, sensor)
            gate_result.actions_taken.append({
                "sensor": sensor,
                "action": f"imputed ({self.imputation_strategy.value})",
                "reason": f"composite score below {self.flag_threshold}",
            })

        return df

    def _impute_sensor(self, df: pd.DataFrame, sensor: str) -> pd.DataFrame:
        """Apply imputation strategy to a single sensor."""
        if self.imputation_strategy == ImputationStrategy.FORWARD_FILL:
            df[sensor] = df.groupby("engine_id")[sensor].ffill()
            df[sensor] = df.groupby("engine_id")[sensor].bfill()

        elif self.imputation_strategy == ImputationStrategy.LINEAR_INTERPOLATION:
            df[sensor] = df.groupby("engine_id")[sensor].transform(
                lambda x: x.interpolate(method="linear").ffill().bfill()
            )

        elif self.imputation_strategy == ImputationStrategy.MEDIAN_FILL:
            median_val = df[sensor].median()
            df[sensor] = df[sensor].fillna(median_val)

        elif self.imputation_strategy == ImputationStrategy.MASK:
            df[sensor] = np.nan

        return df

    def get_history_summary(self) -> pd.DataFrame:
        """Summarize gate history for monitoring dashboard."""
        if not self.history:
            return pd.DataFrame()

        rows = []
        for i, result in enumerate(self.history):
            rows.append({
                "window": i,
                "status": result.status.value,
                "overall_score": result.overall_score,
                "n_flagged": len(result.flagged_sensors),
                "n_blocked": len(result.blocked_sensors),
                "n_actions": len(result.actions_taken),
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pipeline Integration
# ---------------------------------------------------------------------------

def run_governed_pipeline(train_df: pd.DataFrame,
                          sensor_cols: list,
                          reference_size: int = 30,
                          window_size: int = 20,
                          gate: Optional[QualityGate] = None
                          ) -> pd.DataFrame:
    """Run quality gate across all engines, remediating as needed.

    This is the main entry point for integrating governance
    into the model training pipeline.

    Args:
        train_df: full training data
        sensor_cols: sensor columns
        reference_size: baseline window size (first N cycles)
        window_size: evaluation window size
        gate: QualityGate instance (default: standard thresholds)

    Returns:
        Remediated DataFrame ready for model training
    """
    if gate is None:
        gate = QualityGate()

    remediated_frames = []

    for engine_id in train_df["engine_id"].unique():
        engine_df = train_df[train_df["engine_id"] == engine_id].copy()
        engine_df = engine_df.sort_values("cycle").reset_index(drop=True)

        if len(engine_df) < reference_size + window_size:
            remediated_frames.append(engine_df)
            continue

        reference = engine_df.iloc[:reference_size]
        result_parts = [engine_df.iloc[:reference_size]]

        for start in range(reference_size, len(engine_df), window_size):
            end = min(start + window_size, len(engine_df))
            window = engine_df.iloc[start:end]

            gate_result = gate.evaluate(window, reference, sensor_cols)

            if gate_result.status != GateStatus.PASS:
                window = gate.apply_remediation(window, gate_result)

            result_parts.append(window)

        remediated_frames.append(pd.concat(result_parts, ignore_index=True))

    return pd.concat(remediated_frames, ignore_index=True)
