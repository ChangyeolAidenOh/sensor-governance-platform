"""
Bi-LSTM RUL Prediction

Deep learning baseline for Remaining Useful Life prediction on C-MAPSS.
Uses bidirectional LSTM to capture temporal degradation patterns
that tabular models (XGBoost) cannot leverage.

Serves as the DL baseline before Temporal Fusion Transformer (SOTA).
"""

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Optional

from core_pipeline.data.preprocess import DEGRADATION_SENSORS, SENSOR_COLS
from core_pipeline.rul.xgboost_rul import cmapss_score


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RULSequenceDataset(Dataset):
    """Sliding window dataset for RUL prediction.

    For each engine, creates sequences of the last `seq_len` cycles
    at every time step. Pads with zeros if fewer cycles are available.
    """

    def __init__(self, df: pd.DataFrame, seq_len: int = 30,
                 sensor_cols: Optional[list] = None,
                 mode: str = "train"):
        """
        Args:
            df: preprocessed DataFrame with rul column
            seq_len: input sequence length
            sensor_cols: feature columns
            mode: 'train' (all cycles) or 'test' (last cycle per engine)
        """
        if sensor_cols is None:
            sensor_cols = [c for c in DEGRADATION_SENSORS if c in df.columns]

        self.seq_len = seq_len
        self.sensor_cols = sensor_cols
        n_features = len(sensor_cols)

        self.sequences = []
        self.targets = []

        for engine_id in sorted(df["engine_id"].unique()):
            engine_df = df[df["engine_id"] == engine_id].sort_values("cycle")
            values = engine_df[sensor_cols].values.astype(np.float32)
            rul_values = engine_df["rul"].values.astype(np.float32) if "rul" in engine_df.columns else None

            if mode == "test":
                # Only last cycle
                indices = [len(values) - 1]
            else:
                # All cycles
                indices = range(len(values))

            for idx in indices:
                # Extract sequence ending at idx
                start = max(0, idx - seq_len + 1)
                seq = values[start:idx + 1]

                # Pad if shorter than seq_len
                if len(seq) < seq_len:
                    pad = np.zeros((seq_len - len(seq), n_features), dtype=np.float32)
                    seq = np.vstack([pad, seq])

                self.sequences.append(seq)
                if rul_values is not None:
                    self.targets.append(rul_values[idx])

        self.sequences = np.array(self.sequences)
        self.targets = np.array(self.targets) if len(self.targets) > 0 else None

        # Normalize features
        self.mean = self.sequences.reshape(-1, n_features).mean(axis=0)
        self.std = self.sequences.reshape(-1, n_features).std(axis=0)
        self.std[self.std == 0] = 1.0
        self.sequences = (self.sequences - self.mean) / self.std

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.FloatTensor(self.sequences[idx])
        if self.targets is not None:
            target = torch.FloatTensor([self.targets[idx]])
            return seq, target
        return seq

    def get_normalizer(self):
        return self.mean, self.std


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BiLSTM_RUL(nn.Module):
    """Bidirectional LSTM for RUL prediction.

    Architecture: Bi-LSTM → Dropout → FC → ReLU → FC → output
    Takes the final hidden state from both directions.
    """

    def __init__(self, n_features: int, hidden_size: int = 64,
                 n_layers: int = 2, dropout: float = 0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0,
        )

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        """
        Args:
            x: (B, seq_len, n_features)
        Returns:
            rul: (B, 1)
        """
        lstm_out, (h_n, _) = self.lstm(x)

        # Concatenate final hidden states from both directions
        # h_n shape: (n_layers * 2, B, hidden_size)
        h_forward = h_n[-2]   # last layer, forward
        h_backward = h_n[-1]  # last layer, backward
        h_concat = torch.cat([h_forward, h_backward], dim=1)  # (B, hidden*2)

        rul = self.fc(h_concat)
        return rul


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_bilstm_rul(train_df: pd.DataFrame,
                     sensor_cols: Optional[list] = None,
                     seq_len: int = 30,
                     hidden_size: int = 64,
                     n_layers: int = 2,
                     dropout: float = 0.3,
                     batch_size: int = 256,
                     n_epochs: int = 30,
                     lr: float = 1e-3,
                     max_rul: int = 125,
                     val_split: float = 0.2,
                     device: str = "auto") -> dict:
    """Train Bi-LSTM RUL model.

    Args:
        train_df: training data with rul column
        sensor_cols: feature columns
        seq_len: input sequence length
        hidden_size: LSTM hidden dimension
        n_layers: number of LSTM layers
        dropout: dropout rate
        batch_size: training batch size
        n_epochs: number of epochs
        lr: learning rate
        max_rul: RUL clipping value
        val_split: fraction of engines for validation
        device: 'auto', 'mps', 'cuda', or 'cpu'

    Returns:
        dict with model, training history, dataset normalizer
    """
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
    print(f"  Features: {n_features}, Seq len: {seq_len}")

    # Split by engine (no data leakage)
    engine_ids = sorted(train_df["engine_id"].unique())
    n_val = max(1, int(len(engine_ids) * val_split))
    rng = np.random.default_rng(42)
    rng.shuffle(engine_ids)
    val_engines = set(engine_ids[:n_val])
    train_engines = set(engine_ids[n_val:])

    train_subset = train_df[train_df["engine_id"].isin(train_engines)]
    val_subset = train_df[train_df["engine_id"].isin(val_engines)]

    print(f"  Train engines: {len(train_engines)}, Val engines: {len(val_engines)}")

    # Create datasets
    train_dataset = RULSequenceDataset(train_subset, seq_len, sensor_cols, mode="train")
    val_dataset = RULSequenceDataset(val_subset, seq_len, sensor_cols, mode="train")

    # Apply train normalizer to val
    val_dataset.sequences = (
        (val_dataset.sequences * val_dataset.std + val_dataset.mean) - train_dataset.mean
    ) / train_dataset.std

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = BiLSTM_RUL(n_features, hidden_size, n_layers, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )
    criterion = nn.MSELoss()

    # Training loop
    history = {"train_loss": [], "val_loss": [], "val_rmse": []}
    best_val_rmse = float("inf")
    best_state = None
    epochs_no_improve = 0
    patience = 10

    for epoch in range(n_epochs):
        # Train
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
        val_loss = val_rmse ** 2

        scheduler.step(val_loss)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
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

    # Load best model
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

def evaluate_bilstm_rul(model: BiLSTM_RUL,
                        test_df: pd.DataFrame,
                        rul_true: pd.Series,
                        sensor_cols: list,
                        seq_len: int,
                        normalizer: tuple,
                        max_rul: int = 125,
                        device: str = "cpu") -> dict:
    """Evaluate Bi-LSTM on test set (predict RUL at last cycle per engine).

    Args:
        model: trained BiLSTM_RUL
        test_df: test data
        rul_true: ground truth RUL per engine
        sensor_cols: feature columns
        seq_len: sequence length
        normalizer: (mean, std) from training
        max_rul: clipping value
        device: device string

    Returns:
        dict with RMSE, MAE, Score, predictions
    """
    mean, std = normalizer

    model.eval()
    predictions = []

    for engine_id in sorted(test_df["engine_id"].unique()):
        engine_df = test_df[test_df["engine_id"] == engine_id].sort_values("cycle")
        values = engine_df[sensor_cols].values.astype(np.float32)

        # Take last seq_len cycles
        if len(values) >= seq_len:
            seq = values[-seq_len:]
        else:
            pad = np.zeros((seq_len - len(values), len(sensor_cols)), dtype=np.float32)
            seq = np.vstack([pad, values])

        # Normalize with training stats
        seq = (seq - mean) / std

        # Predict
        with torch.no_grad():
            x = torch.FloatTensor(seq).unsqueeze(0).to(device)
            pred = model(x).cpu().item()
            pred = np.clip(pred, 0, max_rul)
            predictions.append(pred)

    y_pred = np.array(predictions)
    y_true = rul_true.values

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    score = cmapss_score(y_true, y_pred)

    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "score": round(score, 2),
        "y_true": y_true,
        "y_pred": y_pred,
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
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    print(f"=== Bi-LSTM RUL Prediction ({args.subset}) ===\n")

    print("Loading data...")
    data = preprocess_subset(args.data_dir, args.subset, args.max_rul)

    print(f"\nTraining (epochs={args.n_epochs}, seq_len={args.seq_len})...")
    t0 = time.time()
    result = train_bilstm_rul(
        data["train"],
        seq_len=args.seq_len,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_rul=args.max_rul,
    )
    elapsed = time.time() - t0
    print(f"  Training time: {elapsed:.1f}s")

    print("\nEvaluating on test set...")
    metrics = evaluate_bilstm_rul(
        result["model"], data["test"], data["rul_true"],
        result["sensor_cols"], result["seq_len"],
        result["normalizer"], args.max_rul, result["device"],
    )
    print(f"  Test RMSE:  {metrics['rmse']}")
    print(f"  Test MAE:   {metrics['mae']}")
    print(f"  Test Score: {metrics['score']}")

    # Comparison
    print("\n=== COMPARISON ===")
    print(f"  {'Model':<20} {'RMSE':>8} {'MAE':>8} {'Score':>10}")
    print(f"  {'XGBoost':<20} {'18.81':>8} {'13.84':>8} {'914.28':>10}")
    print(f"  {'Bi-LSTM':<20} {metrics['rmse']:>8} {metrics['mae']:>8} {metrics['score']:>10}")
