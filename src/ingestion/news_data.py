import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()
engine = create_engine(os.getenv("DB_URL"))

TICKERS = ["SPY", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "JPM", "GS"]

# Approximate earnings dates for realism — sentiment spikes around these
EARNINGS_MONTHS = {
    "AAPL":  [1, 4, 7, 10],
    "MSFT":  [1, 4, 7, 10],
    "GOOGL": [2, 4, 7, 10],
    "AMZN":  [2, 4, 7, 10],
    "TSLA":  [1, 4, 7, 10],
    "JPM":   [1, 4, 7, 10],
    "GS":    [1, 4, 7, 10],
    "SPY":   []
}

def generate_synthetic_sentiment(ticker: str, market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate realistic synthetic daily sentiment that:
    - Correlates with actual market returns (bad days = more negative news)
    - Spikes in news volume around earnings months
    - Has autocorrelation (sentiment trends, doesn't jump randomly)
    - Includes occasional sentiment shocks (major news events)
    """
    np.random.seed(hash(ticker) % 2**32)

    df = market_df[market_df["ticker"] == ticker].copy()
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)

    if n == 0:
        return pd.DataFrame()

    dates = pd.to_datetime(df["date"])
    log_returns = df["log_return"].fillna(0).values

    # ── Base sentiment: correlated with returns + autocorrelation ──
    base_sentiment = np.zeros(n)
    for i in range(1, n):
        # Sentiment mean-reverts but is influenced by yesterday's return
        base_sentiment[i] = (
            0.6 * base_sentiment[i-1]          # autocorrelation
            + 0.4 * log_returns[i] * 15         # return influence
            + np.random.normal(0, 0.08)         # daily noise
        )
    # Clip to [-1, 1] range
    base_sentiment = np.clip(base_sentiment, -1, 1)

    # ── Sentiment shocks: simulate major news events ──
    n_shocks = max(1, n // 60)  # roughly one shock every 3 months
    shock_indices = np.random.choice(n, n_shocks, replace=False)
    shock_magnitudes = np.random.choice([-1, 1], n_shocks) * np.random.uniform(0.3, 0.7, n_shocks)
    for idx, mag in zip(shock_indices, shock_magnitudes):
        # Shock decays over 3-5 days
        decay_len = min(np.random.randint(3, 6), n - idx)
        for j in range(decay_len):
            base_sentiment[idx + j] += mag * (0.8 ** j)
    base_sentiment = np.clip(base_sentiment, -1, 1)

    # ── News volume: higher around earnings, higher on volatile days ──
    base_volume = np.random.poisson(8, n)  # avg 8 articles/day
    earnings_months = EARNINGS_MONTHS.get(ticker, [])
    for i, date in enumerate(dates):
        # Earnings month boost
        if date.month in earnings_months and date.day <= 7:
            base_volume[i] = int(base_volume[i] * np.random.uniform(2.5, 4.0))
        # High volatility day boost
        if abs(log_returns[i]) > 0.02:
            base_volume[i] = int(base_volume[i] * np.random.uniform(1.5, 2.5))

    # ── Derive other sentiment features from base ──
    positive_ratio = (base_sentiment + 1) / 2 * np.random.uniform(0.7, 1.0, n)
    negative_ratio = (1 - (base_sentiment + 1) / 2) * np.random.uniform(0.7, 1.0, n)
    neutral_ratio  = np.clip(1 - positive_ratio - negative_ratio, 0, 1)

    sentiment_std = np.abs(np.random.normal(0.15, 0.05, n)) + np.abs(base_sentiment) * 0.1

    sentiment_velocity = np.concatenate([[0], np.diff(base_sentiment)])

    result = pd.DataFrame({
        "ticker":             ticker,
        "date":               df["date"].values,
        "avg_sentiment":      base_sentiment,
        "sentiment_std":      sentiment_std,
        "positive_ratio":     positive_ratio,
        "negative_ratio":     negative_ratio,
        "news_volume":        base_volume,
        "sentiment_velocity": sentiment_velocity
    })

    return result


if __name__ == "__main__":
    print("Loading market data to align dates...")
    market_df = pd.read_sql("SELECT ticker, date, log_return FROM market_data", engine)
    market_df["date"] = pd.to_datetime(market_df["date"]).dt.date

    all_dfs = []
    for ticker in TICKERS:
        print(f"Generating synthetic sentiment for {ticker}...")
        df = generate_synthetic_sentiment(ticker, market_df)
        if not df.empty:
            all_dfs.append(df)
            print(f" {ticker}: {len(df)} daily records | "
                  f"date range: {df['date'].min()} → {df['date'].max()}")

    final_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal records: {len(final_df)}")

    final_df.to_sql("daily_sentiment", engine, if_exists="replace", index=False)
    print("Synthetic sentiment saved to daily_sentiment table")