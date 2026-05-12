"""
Temporal Fusion Transformer (TFT) for RUL Prediction

Simplified implementation of "Temporal Fusion Transformers for
Interpretable Multi-horizon Time Series Forecasting" (Google, 2021)
adapted for RUL prediction on C-MAPSS.

Key component for this project: Variable Selection Network (VSN)
→ reveals which sensors are important at each time step
→ directly connects to Layer 1 governance (sensor priority)

Architecture: Input → VSN → GRN → LSTM encoder → Multi-head Attention → FC → RUL
"""

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional

from core_pipeline.data.preprocess import DEGRADATION_SENSORS
from core_pipeline.rul.xgboost_rul import cmapss_score
from core_pipeline.rul.bilstm_rul import RULSequenceDataset


# ---------------------------------------------------------------------------
# TFT Building Blocks
# ---------------------------------------------------------------------------

class GatedResidualNetwork(nn.Module):
    """GRN: nonlinear processing with skip connection and gating."""

    def __init__(self, d_input: int, d_hidden: int, d_output: int,
                 dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_input, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_output)
        self.gate = nn.Linear(d_hidden, d_output)
        self.skip = nn.Linear(d_input, d_output) if d_input != d_output else nn.Identity()
        self.norm = nn.LayerNorm(d_output)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = self.skip(x)
        h = F.elu(self.fc1(x))
        h = self.dropout(h)
        h = self.fc2(h) * torch.sigmoid(self.gate(F.elu(self.fc1(x))))
        return self.norm(h + residual)


class VariableSelectionNetwork(nn.Module):
    """VSN: learns which input variables are most relevant.

    Outputs per-variable importance weights at each time step.
    This is the key interpretability component of TFT.
    """

    def __init__(self, n_features: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features

        # Per-variable GRN
        self.variable_grns = nn.ModuleList([
            GatedResidualNetwork(1, d_model, d_model, dropout)
            for _ in range(n_features)
        ])

        # Softmax selection weights
        self.selection_grn = GatedResidualNetwork(
            n_features * d_model, d_model, n_features, dropout
        )

    def forward(self, x):
        """
        Args:
            x: (B, L, n_features)

        Returns:
            selected: (B, L, d_model) — weighted combination
            weights: (B, L, n_features) — variable importance
        """
        B, L, F = x.shape

        # Process each variable independently
        var_outputs = []
        for i in range(self.n_features):
            var_input = x[:, :, i:i+1]  # (B, L, 1)
            var_out = self.variable_grns[i](var_input)  # (B, L, d_model)
            var_outputs.append(var_out)

        var_outputs = torch.stack(var_outputs, dim=2)  # (B, L, F, d_model)

        # Compute selection weights
        flat = var_outputs.reshape(B, L, -1)  # (B, L, F*d_model)
        weights = torch.softmax(
            self.selection_grn(flat), dim=-1
        )  # (B, L, F)

        # Weighted combination
        selected = (var_outputs * weights.unsqueeze(-1)).sum(dim=2)  # (B, L, d_model)

        return selected, weights


class TemporalAttention(nn.Module):
    """Multi-head self-attention over the temporal dimension."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, attn_weights = self.attention(x, x, x)
        return self.norm(x + self.dropout(attn_out)), attn_weights


# ---------------------------------------------------------------------------
# TFT Model
# ---------------------------------------------------------------------------

class TFT_RUL(nn.Module):
    """Temporal Fusion Transformer for RUL prediction.

    Pipeline: VSN → LSTM encoder → Temporal Attention → GRN → FC
    """

    def __init__(self, n_features: int, d_model: int = 64,
                 n_heads: int = 4, lstm_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()

        self.vsn = VariableSelectionNetwork(n_features, d_model, dropout)

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        self.post_lstm = nn.Linear(d_model * 2, d_model)

        self.attention = TemporalAttention(d_model, n_heads, dropout)

        self.output_grn = GatedResidualNetwork(d_model, d_model, d_model, dropout)

        self.fc_out = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(32, 1),
        )

    def forward(self, x, return_weights: bool = False):
        """
        Args:
            x: (B, L, n_features)
            return_weights: if True, also return variable importance

        Returns:
            rul: (B, 1)
            var_weights: (B, L, n_features) — if return_weights
        """
        # Variable Selection
        selected, var_weights = self.vsn(x)  # (B, L, d_model)

        # LSTM encoding
        lstm_out, _ = self.lstm(selected)  # (B, L, d_model*2)
        lstm_out = self.post_lstm(lstm_out)  # (B, L, d_model)

        # Temporal attention
        attn_out, _ = self.attention(lstm_out)  # (B, L, d_model)

        # Take last time step
        last = attn_out[:, -1, :]  # (B, d_model)

        # Output
        out = self.output_grn(last)
        rul = self.fc_out(out)

        if return_weights:
            return rul, var_weights
        return rul


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_tft_rul(train_df: pd.DataFrame,
                  sensor_cols: Optional[list] = None,
                  seq_len: int = 50,
                  d_model: int = 32,
                  n_heads: int = 2,
                  lstm_layers: int = 2,
                  dropout: float = 0.2,
                  batch_size: int = 256,
                  n_epochs: int = 100,
                  lr: float = 1e-3,
                  max_rul: int = 125,
                  val_split: float = 0.2,
                  device: str = "auto") -> dict:
    """Train TFT RUL model."""
    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    if sensor_cols is None:
        sensor_cols = [c for c in DEGRADATION_SENSORS if c in train_df.columns]

    n_features = len(sensor_cols)
    print(f"  Device: {device}")
    print(f"  Features: {n_features}, Seq len: {seq_len}, d_model: {d_model}")

    # Split by engine
    engine_ids = sorted(train_df["engine_id"].unique())
    n_val = max(1, int(len(engine_ids) * val_split))
    rng = np.random.default_rng(42)
    rng.shuffle(engine_ids)
    val_engines = set(engine_ids[:n_val])
    train_engines = set(engine_ids[n_val:])

    train_subset = train_df[train_df["engine_id"].isin(train_engines)]
    val_subset = train_df[train_df["engine_id"].isin(val_engines)]

    print(f"  Train engines: {len(train_engines)}, Val engines: {len(val_engines)}")

    # Datasets
    train_dataset = RULSequenceDataset(train_subset, seq_len, sensor_cols, mode="train")
    val_dataset = RULSequenceDataset(val_subset, seq_len, sensor_cols, mode="train")

    # Apply train normalizer to val
    val_dataset.sequences = (
        (val_dataset.sequences * val_dataset.std + val_dataset.mean) - train_dataset.mean
    ) / train_dataset.std

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = TFT_RUL(n_features, d_model, n_heads, lstm_layers, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )
    criterion = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    # Training loop
    history = {"train_loss": [], "val_loss": [], "val_rmse": []}
    best_val_rmse = float("inf")
    best_state = None
    epochs_no_improve = 0
    patience = 10

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        n_batches = 0

        for sequences, targets in train_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            pred = model(sequences)
            loss = criterion(pred, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        avg_train_loss = train_loss / max(n_batches, 1)

        # Validate
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(device)
                pred = model(sequences)
                val_preds.append(pred.cpu().numpy())
                val_targets.append(targets.numpy())

        val_preds = np.concatenate(val_preds).flatten()
        val_targets = np.concatenate(val_targets).flatten()
        val_preds = np.clip(val_preds, 0, max_rul)

        val_rmse = np.sqrt(np.mean((val_preds - val_targets) ** 2))
        scheduler.step(val_rmse)

        history["train_loss"].append(avg_train_loss)
        history["val_rmse"].append(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{n_epochs} — "
                  f"train: {avg_train_loss:.4f}, "
                  f"val RMSE: {val_rmse:.4f}"
                  f"{' *' if epochs_no_improve == 0 else ''}")

        if epochs_no_improve >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    model.eval()
    print(f"  Best val RMSE: {best_val_rmse:.4f}")

    return {
        "model": model,
        "device": device,
        "sensor_cols": sensor_cols,
        "seq_len": seq_len,
        "normalizer": (train_dataset.mean, train_dataset.std),
        "history": history,
        "best_val_rmse": round(best_val_rmse, 4),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_tft_rul(model: TFT_RUL,
                     test_df: pd.DataFrame,
                     rul_true: pd.Series,
                     sensor_cols: list,
                     seq_len: int,
                     normalizer: tuple,
                     max_rul: int = 125,
                     device: str = "cpu") -> dict:
    """Evaluate TFT on test set."""
    mean, std = normalizer

    model.eval()
    predictions = []
    all_var_weights = []

    for engine_id in sorted(test_df["engine_id"].unique()):
        engine_df = test_df[test_df["engine_id"] == engine_id].sort_values("cycle")
        values = engine_df[sensor_cols].values.astype(np.float32)

        if len(values) >= seq_len:
            seq = values[-seq_len:]
        else:
            pad = np.zeros((seq_len - len(values), len(sensor_cols)), dtype=np.float32)
            seq = np.vstack([pad, values])

        seq = (seq - mean) / std

        with torch.no_grad():
            x = torch.FloatTensor(seq).unsqueeze(0).to(device)
            pred, var_weights = model(x, return_weights=True)
            predictions.append(np.clip(pred.cpu().item(), 0, max_rul))
            all_var_weights.append(var_weights.cpu().numpy())

    y_pred = np.array(predictions)
    y_true = rul_true.values

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    score = cmapss_score(y_true, y_pred)

    # Aggregate variable importance
    var_weights = np.concatenate(all_var_weights, axis=0)  # (n_engines, L, F)
    avg_importance = var_weights.mean(axis=(0, 1))  # (F,)

    importance_df = pd.DataFrame({
        "sensor": sensor_cols,
        "importance": avg_importance,
    }).sort_values("importance", ascending=False)

    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "score": round(score, 2),
        "y_true": y_true,
        "y_pred": y_pred,
        "variable_importance": importance_df,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from core_pipeline.data.preprocess import preprocess_subset

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/CMAPSSData")
    parser.add_argument("--subset", default="FD001")
    parser.add_argument("--max-rul", type=int, default=125)
    parser.add_argument("--seq-len", type=int, default=50)  # 30 → 50
    parser.add_argument("--d-model", type=int, default=32)  # 64 → 32
    parser.add_argument("--n-heads", type=int, default=2)  # 4 → 2
    parser.add_argument("--n-epochs", type=int, default=100)  # 50 → 100
    parser.add_argument("--batch-size", type=int, default=128)  # 256 → 128
    parser.add_argument("--lr", type=float, default=5e-4)  # 1e-3 → 5e-4
    args = parser.parse_args()

    print(f"=== TFT RUL Prediction ({args.subset}) ===\n")

    print("Loading data...")
    data = preprocess_subset(args.data_dir, args.subset, args.max_rul)

    print(f"\nTraining (epochs={args.n_epochs}, d_model={args.d_model})...")
    t0 = time.time()
    result = train_tft_rul(
        data["train"],
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_rul=args.max_rul,
    )
    elapsed = time.time() - t0
    print(f"  Training time: {elapsed:.1f}s")

    print("\nEvaluating on test set...")
    metrics = evaluate_tft_rul(
        result["model"], data["test"], data["rul_true"],
        result["sensor_cols"], result["seq_len"],
        result["normalizer"], args.max_rul, result["device"],
    )
    print(f"  Test RMSE:  {metrics['rmse']}")
    print(f"  Test MAE:   {metrics['mae']}")
    print(f"  Test Score: {metrics['score']}")

    print("\n  Variable Importance (top 10):")
    print(metrics["variable_importance"].head(10).to_string(index=False))

    # Comparison
    print("\n=== COMPARISON ===")
    print(f"  {'Model':<20} {'RMSE':>8} {'MAE':>8} {'Score':>10}")
    print(f"  {'XGBoost':<20} {'18.81':>8} {'13.84':>8} {'914.28':>10}")
    print(f"  {'Bi-LSTM':<20} {'20.11':>8} {'16.77':>8} {'610.84':>10}")
    print(f"  {'TFT':<20} {metrics['rmse']:>8} {metrics['mae']:>8} {metrics['score']:>10}")
