import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import mlflow
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

load_dotenv()
engine = create_engine(os.getenv("DB_URL"))

FEATURES = [
    "realized_vol_5d", "realized_vol_20d", "vol_lag1", "vol_lag2", "vol_lag5",
    "vol_momentum", "price_momentum_5d", "price_momentum_20d",
    "volume_surge", "log_return",
    "avg_sentiment", "sentiment_std", "positive_ratio", "negative_ratio",
    "news_volume", "sentiment_lag1", "sentiment_lag2",
    "sentiment_velocity", "news_volume_lag1"
]

SEQ_LEN = 20       # look back 20 trading days
TARGET = "vol_spike_2d"


# ─────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────

class VolatilityDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_len: int = SEQ_LEN):
        self.seq_len = seq_len
        self.samples = []

        for ticker in df["ticker"].unique():
            ticker_df = df[df["ticker"] == ticker].sort_values("date").reset_index(drop=True)
            X = ticker_df[FEATURES].values.astype(np.float32)
            y = ticker_df[TARGET].values.astype(np.float32)

            for i in range(seq_len, len(ticker_df)):
                self.samples.append((
                    X[i - seq_len:i],   # sequence of past 20 days
                    y[i]                # spike label for today
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X, y = self.samples[idx]
        return torch.tensor(X), torch.tensor(y)


# ─────────────────────────────────────────
# TFT-style model (simplified but genuine)
# ─────────────────────────────────────────

class TemporalAttentionModel(nn.Module):
    """
    Simplified Temporal Fusion Transformer:
    - Input projection per timestep
    - Multi-head self-attention over time
    - Gated residual connections
    - Final classification head
    """
    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)

        # Multi-head attention over time steps
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Gated residual network
        self.grn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )

        # Layer norm
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        projected = self.input_proj(x)                          # (batch, seq, hidden)
        projected = self.norm1(projected)

        # Self-attention
        attn_out, attn_weights = self.attention(
            projected, projected, projected
        )                                                        # (batch, seq, hidden)

        # Gated residual
        gate = self.grn(attn_out)
        out = self.norm2(projected + gate * attn_out)           # residual connection

        # Use last timestep for prediction
        last = out[:, -1, :]                                     # (batch, hidden)
        pred = self.classifier(last).squeeze(-1)                 # (batch,)

        return pred, attn_weights


# ─────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────

def train():
    print("Loading data...")
    df = pd.read_sql("SELECT * FROM master_features", engine)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df[FEATURES] = df[FEATURES].fillna(0)

    # Normalize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df[FEATURES] = scaler.fit_transform(df[FEATURES])

    # Train/val split — last 60 days per ticker = validation
    df["date"] = pd.to_datetime(df["date"])
    cutoff = df["date"].max() - pd.Timedelta(days=60)
    train_df = df[df["date"] <= cutoff]
    val_df   = df[df["date"] > cutoff]

    print(f"Train: {len(train_df)} rows | Val: {len(val_df)} rows")

    train_dataset = VolatilityDataset(train_df)
    val_dataset   = VolatilityDataset(val_df)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False)

    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # Model
    model = TemporalAttentionModel(
        input_size=len(FEATURES),
        hidden_size=64,
        num_heads=4,
        dropout=0.1
    )

    # Class imbalance weight
    spike_ratio = (df[TARGET] == 0).sum() / (df[TARGET] == 1).sum()
    pos_weight = torch.tensor([spike_ratio], dtype=torch.float32)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    mlflow.set_experiment("volatility_tft")

    with mlflow.start_run(run_name="temporal_attention_model"):
        mlflow.log_params({
            "seq_len": SEQ_LEN,
            "hidden_size": 64,
            "num_heads": 4,
            "dropout": 0.1,
            "epochs": 30,
            "model": "TemporalAttentionModel"
        })

        best_auc = 0
        patience_counter = 0

        for epoch in range(30):
            # ── Train ──
            model.train()
            train_losses = []
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                preds, _ = model(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())

            # ── Validate ──
            model.eval()
            val_preds, val_actuals = [], []
            val_losses = []
            all_attn_weights = []

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    preds, attn = model(X_batch)
                    loss = criterion(preds, y_batch)
                    val_losses.append(loss.item())
                    val_preds.extend(preds.numpy())
                    val_actuals.extend(y_batch.numpy())
                    all_attn_weights.append(attn.numpy())

            val_preds = np.array(val_preds)
            val_actuals = np.array(val_actuals)
            val_binary = (val_preds > 0.5).astype(int)

            try:
                auc = roc_auc_score(val_actuals, val_preds)
            except:
                auc = 0.0

            train_loss = np.mean(train_losses)
            val_loss   = np.mean(val_losses)
            scheduler.step(val_loss)

            print(f"Epoch {epoch+1:02d} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"AUC: {auc:.3f}")

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auc": auc
            }, step=epoch)

            # Save best model
            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), "data/processed/tft_best.pt")
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= 7:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # ── Final metrics ──
        model.load_state_dict(torch.load("data/processed/tft_best.pt"))
        model.eval()

        final_preds, final_actuals, final_attns = [], [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                preds, attn = model(X_batch)
                final_preds.extend(preds.numpy())
                final_actuals.extend(y_batch.numpy())
                final_attns.append(attn.mean(dim=1).numpy())  # avg over heads

        final_preds   = np.array(final_preds)
        final_actuals = np.array(final_actuals)
        final_binary  = (final_preds > 0.5).astype(int)

        metrics = {
            "final_auc":       roc_auc_score(final_actuals, final_preds),
            "final_precision": precision_score(final_actuals, final_binary, zero_division=0),
            "final_recall":    recall_score(final_actuals, final_binary, zero_division=0),
            "final_f1":        f1_score(final_actuals, final_binary, zero_division=0)
        }

        print(f"\nFinal TFT Metrics:")
        for k, v in metrics.items():
            print(f"   {k}: {v:.3f}")
            mlflow.log_metric(k, v)

        # ── Attention visualization ──
        attn_matrix = np.concatenate(final_attns, axis=0)  # (samples, seq, seq)
        avg_attn = attn_matrix.mean(axis=0)                 # (seq, seq)
        temporal_attn = avg_attn.mean(axis=0)               # (seq,) — attention per past day

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Temporal attention
        days_back = list(range(SEQ_LEN, 0, -1))
        axes[0].bar(days_back, temporal_attn, color="steelblue", alpha=0.8)
        axes[0].set_xlabel("Days in the past")
        axes[0].set_ylabel("Average attention weight")
        axes[0].set_title("Temporal Attention — Which Past Days Matter Most?")
        axes[0].invert_xaxis()

        # Prediction distribution
        axes[1].hist(final_preds[final_actuals == 0], bins=40,
                     alpha=0.6, label="No spike", color="green")
        axes[1].hist(final_preds[final_actuals == 1], bins=40,
                     alpha=0.6, label="Spike", color="red")
        axes[1].set_xlabel("Predicted probability")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Prediction Distribution\n(Spike vs No-Spike separation)")
        axes[1].legend()

        plt.suptitle(f"Temporal Attention Model — Val AUC: {metrics['final_auc']:.3f}",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig("data/processed/tft_interpretation.png", dpi=150)
        mlflow.log_artifact("data/processed/tft_interpretation.png")

        print(f"\nTraining complete | Best AUC: {best_auc:.3f}")
        print("Charts saved to data/processed/tft_interpretation.png")

    return model, metrics


if __name__ == "__main__":
    train()