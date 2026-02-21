import yfinance as yf
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()


engine = create_engine(os.getenv("DB_URL"))

TICKERS = ["SPY", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "JPM", "GS"]

def fetch_market_data(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df["ticker"] = ticker
    df["log_return"] = np.log(df["close"] / df["close"].shift(1)) #Instead of raw price changes, we use log returns because they're statistically better behaved theyre symmetric, additive across time, and normally distributed enough for modeling. shift(1) means yesterday's close, so computing today's return relative to yesterday
    
    # Realized volatility 5-day and 20-day rolling

    df["realized_vol_5d"] = df["log_return"].rolling(5).std() * np.sqrt(252) # annualizes it (252 trading days in a year) so it's expressed as an annual percentage
    df["realized_vol_20d"] = df["log_return"].rolling(20).std() * np.sqrt(252)
    
    # Target variable: vol spike in next 2 days
    # A spike = realized vol exceeds 90th percentile
    vol_90th = df["realized_vol_5d"].quantile(0.90)
    df["vol_spike_2d"] = (
        df["realized_vol_5d"].shift(-2) > vol_90th
    ).astype(int)
    
    df.reset_index(inplace=True)
    df.rename(columns={"Date": "date"}, inplace=True)
    return df.dropna()

def save_to_db(df: pd.DataFrame, table: str = "market_data"):
    df.to_sql(table, engine, if_exists="append", index=False)
    print(f"Saved {len(df)} rows to {table}")

if __name__ == "__main__":
    for ticker in TICKERS:
        print(f"Fetching {ticker}...")
        df = fetch_market_data(ticker)
        save_to_db(df)
    print("Market data ingestion complete")