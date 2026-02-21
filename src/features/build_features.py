import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()
engine = create_engine(os.getenv("DB_URL"))

def build_master_features() -> pd.DataFrame:
    print("Loading market data...")
    market = pd.read_sql("SELECT * FROM market_data", engine)
    market["date"] = pd.to_datetime(market["date"]).dt.date

    print("Loading sentiment data...")
    sentiment = pd.read_sql("SELECT * FROM daily_sentiment", engine)
    sentiment["date"] = pd.to_datetime(sentiment["date"]).dt.date

    print("Merging...")
    df = pd.merge(market, sentiment, on=["ticker", "date"], how="left")

    # Fill missing sentiment with neutral (days with no news)
    sentiment_cols = ["avg_sentiment", "sentiment_std", "positive_ratio",
                      "negative_ratio", "news_volume", "sentiment_velocity"]
    df[sentiment_cols] = df[sentiment_cols].fillna(0)

    print("Engineering additional features...")

    # Lagged sentiment features (yesterday's and 2 days ago sentiment)
    df = df.sort_values(["ticker", "date"])
    df["sentiment_lag1"] = df.groupby("ticker")["avg_sentiment"].shift(1)
    df["sentiment_lag2"] = df.groupby("ticker")["avg_sentiment"].shift(2)
    df["news_volume_lag1"] = df.groupby("ticker")["news_volume"].shift(1)

    # Lagged volatility features
    df["vol_lag1"] = df.groupby("ticker")["realized_vol_5d"].shift(1)
    df["vol_lag2"] = df.groupby("ticker")["realized_vol_5d"].shift(2)
    df["vol_lag5"] = df.groupby("ticker")["realized_vol_5d"].shift(5)

    # Vol momentum — is volatility rising or falling?
    df["vol_momentum"] = df["realized_vol_5d"] - df["realized_vol_20d"]

    # Price momentum
    df["price_momentum_5d"] = df.groupby("ticker")["close"].pct_change(5)
    df["price_momentum_20d"] = df.groupby("ticker")["close"].pct_change(20)

    # Volume surge — is trading volume unusually high?
    df["volume_ma20"] = df.groupby("ticker")["volume"].transform(
        lambda x: x.rolling(20).mean()
    )
    df["volume_surge"] = df["volume"] / df["volume_ma20"]

    # Day of week (volatility patterns differ Mon vs Fri)
    df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek

    # Drop rows with NaN target or key features
    df = df.dropna(subset=["vol_spike_2d", "vol_lag1", "sentiment_lag1"])

    print(f" Master feature table built — {len(df)} rows, {len(df.columns)} columns")
    return df

if __name__ == "__main__":
    df = build_master_features()
    df.to_sql("master_features", engine, if_exists="replace", index=False)
    print(" Saved to master_features table")
    print(df[["ticker", "date", "vol_spike_2d", "avg_sentiment", "realized_vol_5d"]].tail(10))