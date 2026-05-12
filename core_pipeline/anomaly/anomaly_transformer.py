"""
Anomaly Transformer — SOTA Time Series Anomaly Detection

Implementation of "Anomaly Transformer: Time Series Anomaly Detection
with Association Discrepancy" (ICLR 2022 Spotlight, Xu et al.)

Key idea: normal points associate broadly across the series, while
anomaly points concentrate attention on adjacent time points.
The Association Discrepancy (KL divergence between prior-association
and series-association) serves as the anomaly criterion.

Evaluation note: the original paper uses point-adjust F1 which has
known issues (TiSAT: random guess > SOTA). We evaluate with both
standard F1 and point-adjust for transparency.
"""

import math
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional

from core_pipeline.data.preprocess import DEGRADATION_SENSORS
from core_pipeline.anomaly.isolation_forest import (
    create_anomaly_labels,
    evaluate_anomaly_detection,
    evaluate_multiple_thresholds,
    analyze_engine_detections,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CMAPSSWindowDataset(Dataset):
    """Sliding window dataset for C-MAPSS sensor data."""

    def __init__(self, df: pd.DataFrame, window_size: int = 50,
                 sensor_cols: Optional[list] = None, stride: int = 1):
        if sensor_cols is None:
            sensor_cols = [c for c in DEGRADATION_SENSORS if c in df.columns]

        self.window_size = window_size
        self.sensor_cols = sensor_cols

        # Build windows per engine (no cross-engine windows)
        self.windows = []
        self.window_indices = []  # (engine_id, start_idx) for mapping back

        for engine_id in sorted(df["engine_id"].unique()):
            engine_df = df[df["engine_id"] == engine_id].sort_values("cycle")
            values = engine_df[sensor_cols].values.astype(np.float32)

            for start in range(0, len(values) - window_size + 1, stride):
                self.windows.append(values[start:start + window_size])
                self.window_indices.append((engine_id, start))

        self.windows = np.array(self.windows)

        # Normalize
        self.mean = self.windows.reshape(-1, len(sensor_cols)).mean(axis=0)
        self.std = self.windows.reshape(-1, len(sensor_cols)).std(axis=0)
        self.std[self.std == 0] = 1.0
        self.windows = (self.windows - self.mean) / self.std

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.windows[idx])


# ---------------------------------------------------------------------------
# Model Components
# ---------------------------------------------------------------------------

class TriangularCausalMask:
    """Causal mask for attention (not used in anomaly detection,
    but kept for API compatibility)."""
    def __init__(self, B, L, device="cpu"):
        mask = torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()
        self.mask = mask.unsqueeze(0).expand(B, -1, -1)


class AnomalyAttention(nn.Module):
    """Anomaly-Attention mechanism.

    Computes two types of association:
    1. Prior-association: learned Gaussian kernel (distance-based)
    2. Series-association: standard self-attention (content-based)

    The discrepancy between these two reveals anomalies.
    """

    def __init__(self, d_model: int, n_heads: int, d_keys: int = None,
                 attention_dropout: float = 0.0):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        self.n_heads = n_heads
        self.d_keys = d_keys
        self.scale = d_keys ** -0.5

        self.W_Q = nn.Linear(d_model, d_keys * n_heads)
        self.W_K = nn.Linear(d_model, d_keys * n_heads)
        self.W_V = nn.Linear(d_model, d_keys * n_heads)
        self.out_proj = nn.Linear(d_keys * n_heads, d_model)

        # Learnable prior: sigma parameter for Gaussian kernel
        self.sigma = nn.Parameter(torch.ones(1, n_heads, 1, 1))

        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, x):
        """
        Args:
            x: (B, L, d_model)

        Returns:
            output: (B, L, d_model)
            series_assoc: (B, H, L, L) — self-attention weights
            prior_assoc: (B, H, L, L) — Gaussian prior weights
        """
        B, L, _ = x.shape
        H = self.n_heads

        Q = self.W_Q(x).view(B, L, H, self.d_keys).transpose(1, 2)
        K = self.W_K(x).view(B, L, H, self.d_keys).transpose(1, 2)
        V = self.W_V(x).view(B, L, H, self.d_keys).transpose(1, 2)

        # Series association (standard attention)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        series_assoc = self.dropout(torch.softmax(scores, dim=-1))

        # Prior association (Gaussian kernel based on distance)
        distances = torch.abs(
            torch.arange(L, device=x.device).float().unsqueeze(0) -
            torch.arange(L, device=x.device).float().unsqueeze(1)
        )  # (L, L)
        sigma = torch.clamp(self.sigma, min=1e-4)
        prior = torch.exp(-0.5 * (distances / sigma) ** 2)
        prior_assoc = prior / (prior.sum(dim=-1, keepdim=True) + 1e-8)
        prior_assoc = prior_assoc.expand(B, -1, -1, -1)

        # Output
        out = torch.matmul(series_assoc, V)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        out = self.out_proj(out)

        return out, series_assoc, prior_assoc


class AnomalyTransformerBlock(nn.Module):
    """Single Anomaly Transformer encoder block."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int = None,
                 dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or d_model * 4

        self.attention = AnomalyAttention(d_model, n_heads,
                                           attention_dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        attn_out, series, prior = self.attention(x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x, series, prior


class AnomalyTransformerModel(nn.Module):
    """Anomaly Transformer for time series anomaly detection.

    Architecture: input projection → N transformer blocks → output projection
    Training: minimax on reconstruction loss and association discrepancy
    Inference: association discrepancy as anomaly score
    """

    def __init__(self, n_features: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, d_ff: int = 256, dropout: float = 0.1,
                 window_size: int = 50):
        super().__init__()

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_embed = nn.Parameter(
            torch.randn(1, window_size, d_model) * 0.02
        )

        self.layers = nn.ModuleList([
            AnomalyTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.output_proj = nn.Linear(d_model, n_features)

    def forward(self, x):
        """
        Args:
            x: (B, L, n_features)

        Returns:
            recon: (B, L, n_features) — reconstruction
            series_list: list of (B, H, L, L) — per-layer series associations
            prior_list: list of (B, H, L, L) — per-layer prior associations
        """
        h = self.input_proj(x) + self.pos_embed[:, :x.size(1), :]

        series_list = []
        prior_list = []

        for layer in self.layers:
            h, series, prior = layer(h)
            series_list.append(series)
            prior_list.append(prior)

        recon = self.output_proj(h)
        return recon, series_list, prior_list


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------

def association_discrepancy(series_list, prior_list):
    eps = 1e-8
    disc = 0
    for series, prior in zip(series_list, prior_list):
        series_safe = series.clamp(min=eps)
        prior_safe = prior.clamp(min=eps)

        # Symmetric KL: KL(series||prior) + KL(prior||series)
        kl = (series_safe * (series_safe.log() - prior_safe.log())).sum(dim=-1).mean(dim=1)
        kl_reverse = (prior_safe * (prior_safe.log() - series_safe.log())).sum(dim=-1).mean(dim=1)

        disc = disc + (kl + kl_reverse)

    disc = disc / len(series_list)
    return disc


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_anomaly_transformer(dataset: CMAPSSWindowDataset,
                               n_features: int,
                               d_model: int = 128,
                               n_heads: int = 8,
                               n_layers: int = 3,
                               d_ff: int = 256,
                               window_size: int = 50,
                               batch_size: int = 32,
                               n_epochs: int = 30,
                               lr: float = 5e-5,
                               lambda_disc: float = 1.0,
                               device: str = "auto") -> dict:
    """Train Anomaly Transformer with minimax strategy.

    Phase 1 (minimize): minimize reconstruction loss + association discrepancy
    Phase 2 (maximize): maximize association discrepancy (amplify normal-abnormal gap)
    """
    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    print(f"  Device: {device}")

    model = AnomalyTransformerModel(
        n_features=n_features, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_ff=d_ff, window_size=window_size,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        drop_last=True)

    model.train()
    for epoch in range(n_epochs):
        epoch_recon_loss = 0
        epoch_disc_loss = 0
        n_batches = 0

        for batch in loader:
            batch = batch.to(device)

            # --- Phase 1: Minimize reconstruction + discrepancy ---
            optimizer.zero_grad()
            recon, series_list, prior_list = model(batch)

            recon_loss = F.mse_loss(recon, batch)
            disc = association_discrepancy(series_list, prior_list)
            disc_loss = disc.mean()

            loss_min = recon_loss - lambda_disc * disc_loss
            loss_min.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # --- Phase 2: Maximize discrepancy ---
            optimizer.zero_grad()
            recon, series_list, prior_list = model(batch)

            disc = association_discrepancy(series_list, prior_list)
            disc_loss = disc.mean()

            loss_max = lambda_disc * disc_loss
            loss_max.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_recon_loss += recon_loss.item()
            epoch_disc_loss += disc_loss.item()
            n_batches += 1

        avg_recon = epoch_recon_loss / max(n_batches, 1)
        avg_disc = epoch_disc_loss / max(n_batches, 1)
        print(f"  Epoch {epoch+1}/{n_epochs} — "
              f"recon: {avg_recon:.4f}, disc: {avg_disc:.4f}")

    return {"model": model, "device": device}


# ---------------------------------------------------------------------------
# Inference & Scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_windows(model: AnomalyTransformerModel,
                  dataset: CMAPSSWindowDataset,
                  device: str = "cpu",
                  batch_size: int = 128) -> np.ndarray:
    """Score all windows using association discrepancy.

    Returns:
        scores: (n_windows, window_size) — per-timestep anomaly scores
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_scores = []
    for batch in loader:
        batch = batch.to(device)
        recon, series_list, prior_list = model(batch)
        disc = association_discrepancy(series_list, prior_list)
        all_scores.append(disc.cpu().numpy())

    return np.concatenate(all_scores, axis=0)


def map_scores_to_cycles(window_scores: np.ndarray,
                          dataset: CMAPSSWindowDataset,
                          df: pd.DataFrame) -> pd.DataFrame:
    """Map per-window scores back to individual cycles.

    Each cycle may appear in multiple windows. We take the max score
    across all windows containing that cycle.

    Args:
        window_scores: (n_windows, window_size)
        dataset: the dataset with window_indices
        df: original DataFrame to add scores to

    Returns:
        DataFrame with anomaly_score column
    """
    df = df.copy()
    df["anomaly_score"] = 0.0
    df = df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)

    # Build engine index mapping
    engine_start_idx = {}
    current_idx = 0
    for engine_id in sorted(df["engine_id"].unique()):
        engine_start_idx[engine_id] = current_idx
        current_idx += len(df[df["engine_id"] == engine_id])

    score_accumulator = np.zeros(len(df))
    count_accumulator = np.zeros(len(df))

    for w_idx, (engine_id, start) in enumerate(dataset.window_indices):
        global_start = engine_start_idx[engine_id] + start
        w_size = window_scores.shape[1]

        for t in range(w_size):
            global_idx = global_start + t
            if global_idx < len(df):
                score_accumulator[global_idx] = max(
                    score_accumulator[global_idx],
                    window_scores[w_idx, t]
                )
                count_accumulator[global_idx] += 1

    df["anomaly_score"] = score_accumulator
    # Binary prediction: top 25% of nonzero scores flagged as anomaly
    nonzero_scores = score_accumulator[score_accumulator > 0]
    if len(nonzero_scores) > 0:
        threshold = np.percentile(nonzero_scores, 75)
    else:
        threshold = np.percentile(score_accumulator, 75)
    df["anomaly_pred_binary"] = (score_accumulator > threshold).astype(int)

    return df


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
    parser.add_argument("--rul-threshold", type=int, default=50)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    print(f"=== Anomaly Transformer ({args.subset}) ===\n")

    print("Loading data...")
    data = preprocess_subset(args.data_dir, args.subset, args.max_rul)
    train_df = data["train"]

    sensor_cols = [c for c in DEGRADATION_SENSORS if c in train_df.columns]
    n_features = len(sensor_cols)
    print(f"  Sensors: {n_features}")

    print(f"Creating windows (size={args.window_size})...")
    dataset = CMAPSSWindowDataset(
        train_df, window_size=args.window_size,
        sensor_cols=sensor_cols, stride=1,
    )
    print(f"  Windows: {len(dataset)}")

    print(f"\nTraining (epochs={args.n_epochs})...")
    t0 = time.time()
    result = train_anomaly_transformer(
        dataset, n_features=n_features,
        d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, window_size=args.window_size,
        batch_size=args.batch_size, n_epochs=args.n_epochs, lr=args.lr,
    )
    elapsed = time.time() - t0
    print(f"  Training time: {elapsed:.1f}s")

    print("\nScoring...")
    window_scores = score_windows(
        result["model"], dataset, result["device"],
    )
    print(f"  Window scores shape: {window_scores.shape}")

    print("Mapping scores to cycles...")
    scored_df = map_scores_to_cycles(window_scores, dataset, train_df)

    print(f"\nEvaluation (RUL threshold={args.rul_threshold}):")
    metrics = evaluate_anomaly_detection(scored_df, args.rul_threshold)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\nMulti-threshold evaluation:")
    multi = evaluate_multiple_thresholds(scored_df)
    print(multi.to_string(index=False))

    print("\nPer-engine detection summary:")
    engine_summary = analyze_engine_detections(scored_df, args.rul_threshold)
    detected = engine_summary["detected"].sum()
    total = len(engine_summary)
    print(f"  Engines detected: {detected}/{total}")

    lead_times = engine_summary["lead_time"].dropna()
    if len(lead_times) > 0:
        print(f"  Lead time - mean: {lead_times.mean():.1f}, "
              f"median: {lead_times.median():.1f}, "
              f"min: {lead_times.min():.0f}, "
              f"max: {lead_times.max():.0f} cycles")

    # --- Comparison summary ---
    print("\n=== COMPARISON vs BASELINES ===")
    print(f"  {'Model':<25} {'AUROC':>8} {'F1(RUL50)':>10} {'Detection':>10}")
    print(f"  {'Rolling Z-score':<25} {'0.327':>8} {'0.456':>10} {'100/100':>10}")
    print(f"  {'Isolation Forest':<25} {'0.955':>8} {'0.789':>10} {'100/100':>10}")
    print(f"  {'Anomaly Transformer':<25} {metrics['auroc']:>8} {metrics['f1']:>10} "
          f"{f'{detected}/{total}':>10}")
