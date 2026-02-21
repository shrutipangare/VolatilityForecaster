import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (classification_report, roc_auc_score,
                             precision_score, recall_score, f1_score)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

load_dotenv()
engine = create_engine(os.getenv("DB_URL"))

FEATURES = [
    "realized_vol_5d", "realized_vol_20d", "vol_lag1", "vol_lag2", "vol_lag5",
    "vol_momentum", "price_momentum_5d", "price_momentum_20d",
    "volume_surge", "log_return", "day_of_week",
    "avg_sentiment", "sentiment_std", "positive_ratio", "negative_ratio",
    "news_volume", "sentiment_lag1", "sentiment_lag2", "news_volume_lag1",
    "sentiment_velocity"
]
TARGET = "vol_spike_2d"

def load_data():
    df = pd.read_sql("SELECT * FROM master_features", engine)
    df = df.sort_values(["ticker", "date"])
    return df

def evaluate(y_true, y_pred, y_prob):
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0)
    }

def train():
    df = load_data()
    X = df[FEATURES].fillna(0)
    y = df[TARGET]

    print(f"Class balance — Spikes: {y.sum()} ({y.mean():.1%}) | Calm: {(~y.astype(bool)).sum()}")

    # Time series cross validation — never leak future data into training
    tscv = TimeSeriesSplit(n_splits=5)

    mlflow.set_experiment("volatility_xgboost")

    with mlflow.start_run(run_name="xgboost_baseline"):
        params = {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": int((y == 0).sum() / (y == 1).sum()),  # handle class imbalance
            "random_state": 42,
            "eval_metric": "auc"
        }
        mlflow.log_params(params)

        fold_metrics = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = XGBClassifier(**params)
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      verbose=False)

            y_pred = model.predict(X_val)
            y_prob = model.predict_proba(X_val)[:, 1]

            metrics = evaluate(y_val, y_pred, y_prob)
            fold_metrics.append(metrics)
            print(f"Fold {fold+1} — AUC: {metrics['roc_auc']:.3f} | "
                  f"Precision: {metrics['precision']:.3f} | "
                  f"Recall: {metrics['recall']:.3f} | "
                  f"F1: {metrics['f1']:.3f}")

        # Average metrics across folds
        avg_metrics = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
        print(f"\nAverage CV Metrics:")
        for k, v in avg_metrics.items():
            print(f"   {k}: {v:.3f}")
            mlflow.log_metric(k, v)

        # Train final model on all data
        final_model = XGBClassifier(**params)
        final_model.fit(X, y, verbose=False)

        # Feature importance plot
        importance = pd.Series(
            final_model.feature_importances_, index=FEATURES
        ).sort_values(ascending=False)

        print(f"\nTop 10 Features:")
        print(importance.head(10))

        # Save plot
        fig, ax = plt.subplots(figsize=(10, 6))
        importance.head(15).plot(kind="barh", ax=ax)
        ax.set_title("XGBoost Feature Importance — Volatility Spike Prediction")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig("data/processed/feature_importance.png")
        mlflow.log_artifact("data/processed/feature_importance.png")

        # Log model
        mlflow.xgboost.log_model(final_model, "xgboost_model")
        print("\nModel trained and logged to MLflow")

        return final_model, avg_metrics

if __name__ == "__main__":
    model, metrics = train()