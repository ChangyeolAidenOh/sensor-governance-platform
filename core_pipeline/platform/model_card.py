"""
Model Card Auto-Generation

Generates standardized Model Cards from experiment results.
Reference: Mitchell et al. (2019), "Model Cards for Model Reporting"

Usage:
    python -m core_pipeline.platform.model_card
"""

import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict


@dataclass
class ModelCard:
    """Standardized model documentation."""
    model_name: str
    model_type: str
    version: str
    created_at: str
    framework: str
    dataset: str
    subset: str
    n_train_samples: int
    n_test_samples: int
    n_features: int
    target: str
    max_rul: int
    test_rmse: float
    test_mae: float
    test_score: float
    best_val_rmse: float = None
    known_limitations: list = field(default_factory=list)
    failure_modes: list = field(default_factory=list)
    recommended_conditions: str = ""
    not_recommended: str = ""
    data_quality_required: str = ""
    drift_sensitivity: str = ""


def generate_xgboost_card() -> ModelCard:
    return ModelCard(
        model_name="XGBoost RUL Predictor",
        model_type="Gradient Boosted Trees",
        version="1.0.0",
        created_at=datetime.now().isoformat(),
        framework="xgboost",
        dataset="NASA C-MAPSS",
        subset="FD001",
        n_train_samples=20631,
        n_test_samples=100,
        n_features=98,
        target="Remaining Useful Life (cycles)",
        max_rul=125,
        test_rmse=18.81,
        test_mae=13.84,
        test_score=914.28,
        known_limitations=[
            "Trained on single operating condition (FD001, sea level)",
            "Cross-condition transfer degrades RMSE by +187% (FD002)",
            "Does not capture temporal order (each cycle treated independently)",
            "Sensitive to concept drift (RMSE +25.69 in corruption experiment)",
        ],
        failure_modes=[
            "Operating condition change: RMSE 18.81 -> 53.99 (+187%)",
            "Concept drift: RMSE 18.81 -> 44.50 (+137%)",
            "Sensor drift (high): RMSE 18.81 -> 23.20 (+23%)",
        ],
        recommended_conditions="Single operating condition (FD001-like). "
                                "Sensor Health Score >= 70 (Quality Gate PASS).",
        not_recommended="Multi-condition environments (FD002/FD004) without "
                        "condition normalization. Data with PSI > 0.2 on >30% of features.",
        data_quality_required="Quality Gate PASS (Sensor Health >= 70). "
                               "Missing rate < 20%. No concept drift detected.",
        drift_sensitivity="Concept drift: CRITICAL (100% detection, retrain required). "
                          "Sensor drift: MODERATE (80.1% recovery with PSI gate). "
                          "Gaussian noise: LOW (72-79% recovery with smoothing).",
    )


def generate_tft_card() -> ModelCard:
    return ModelCard(
        model_name="TFT RUL Predictor",
        model_type="Temporal Fusion Transformer",
        version="1.0.0",
        created_at=datetime.now().isoformat(),
        framework="pytorch",
        dataset="NASA C-MAPSS",
        subset="FD001",
        n_train_samples=20631,
        n_test_samples=100,
        n_features=14,
        target="Remaining Useful Life (cycles)",
        max_rul=125,
        test_rmse=16.39,
        test_mae=12.18,
        test_score=481.78,
        best_val_rmse=13.69,
        known_limitations=[
            "Learning instability: same hyperparameters -> RMSE 16-42 range",
            "VSN prone to winner-take-all on small datasets (80 engines)",
            "Requires multi-seed ensemble for reliable deployment",
            "MC Dropout coverage only 0.56 (target 0.90) — over-confident",
        ],
        failure_modes=[
            "Random init: RMSE ranges 16.39 (best) to 41.82 (worst)",
            "VSN collapse: single sensor captures >70% importance",
            "Val-Test gap: val RMSE 11.28 vs test RMSE 20.12 (Run 5)",
        ],
        recommended_conditions="Single operating condition with multi-seed training "
                                "(min 5 seeds, select by val RMSE). "
                                "Sensor Health Score >= 70.",
        not_recommended="Single-seed deployment. Multi-condition environments "
                        "without domain adaptation.",
        data_quality_required="Same as XGBoost. Additionally: "
                               "sequence length >= 30 cycles per engine.",
        drift_sensitivity="Variable Importance predicts noise sensitivity (5.3x) "
                          "but NOT drift sensitivity (1.0x). "
                          "Monitor top-importance sensors for noise, ALL sensors for drift.",
    )


def generate_bilstm_card() -> ModelCard:
    return ModelCard(
        model_name="Bi-LSTM RUL Predictor",
        model_type="Bidirectional LSTM",
        version="1.0.0",
        created_at=datetime.now().isoformat(),
        framework="pytorch",
        dataset="NASA C-MAPSS",
        subset="FD001",
        n_train_samples=20631,
        n_test_samples=100,
        n_features=14,
        target="Remaining Useful Life (cycles)",
        max_rul=125,
        test_rmse=20.11,
        test_mae=16.77,
        test_score=610.84,
        best_val_rmse=13.58,
        known_limitations=[
            "RMSE higher than XGBoost despite temporal modeling",
            "Score (610) better than XGBoost (914) — fewer late predictions",
            "MC Dropout uncertainty is over-confident (coverage 0.56 vs target 0.90)",
        ],
        failure_modes=[
            "Val-Test generalization gap: val 13.58 vs test 20.11",
            "Random initialization sensitivity across runs",
        ],
        recommended_conditions="When NASA Score optimization is priority over RMSE. "
                                "Single operating condition.",
        not_recommended="When RMSE is primary metric (use XGBoost or TFT instead).",
        data_quality_required="Quality Gate PASS. Sequence length >= 30 cycles.",
        drift_sensitivity="Similar to XGBoost for feature-level drift. "
                          "Temporal pattern disruption not separately quantified.",
    )


def save_card(card: ModelCard, output_dir: str = "models"):
    path = Path(output_dir)
    path.mkdir(exist_ok=True)
    filename = f"model_card_{card.model_name.lower().replace(' ', '_')}.json"
    filepath = path / filename
    with open(filepath, "w") as f:
        json.dump(asdict(card), f, indent=2, default=str)
    return filepath


def print_card(card: ModelCard):
    print(f"\n{'='*60}")
    print(f"MODEL CARD: {card.model_name}")
    print(f"{'='*60}")
    print(f"  Type:      {card.model_type}")
    print(f"  Version:   {card.version}")
    print(f"  Framework: {card.framework}")
    print(f"  Dataset:   {card.dataset} ({card.subset})")
    print(f"  Features:  {card.n_features}")

    print(f"\n  --- Performance ---")
    print(f"  RMSE:  {card.test_rmse}")
    print(f"  MAE:   {card.test_mae}")
    print(f"  Score: {card.test_score}")
    if card.best_val_rmse:
        print(f"  Val RMSE: {card.best_val_rmse}")

    print(f"\n  --- Limitations ---")
    for lim in card.known_limitations:
        print(f"  - {lim}")

    print(f"\n  --- Failure Modes ---")
    for fm in card.failure_modes:
        print(f"  - {fm}")

    print(f"\n  --- Usage ---")
    print(f"  Recommended: {card.recommended_conditions}")
    print(f"  Avoid:       {card.not_recommended}")

    print(f"\n  --- Governance ---")
    print(f"  Quality:  {card.data_quality_required}")
    print(f"  Drift:    {card.drift_sensitivity}")


if __name__ == "__main__":
    print("=== Model Card Generation ===\n")

    cards = [generate_xgboost_card(), generate_bilstm_card(), generate_tft_card()]

    for card in cards:
        print_card(card)
        filepath = save_card(card)
        print(f"\n  Saved: {filepath}")
