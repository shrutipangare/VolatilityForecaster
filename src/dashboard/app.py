import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
#from sqlalchemy import create_engine
#from dotenv import load_dotenv
import os
#import torch
#import torch.nn as nn
#from sklearn.preprocessing import StandardScaler
#import joblib
#import mlflow.xgboost

#load_dotenv()
#engine = create_engine(os.getenv("DB_URL"))


st.set_page_config(
    page_title="Volatility Spike Forecaster",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Volatility Spike Forecaster")
st.markdown("*Predicting market turbulence 48 hours ahead using news sentiment + price signals*")

# ─────────────────────────────────────────
# Load data
# ─────────────────────────────────────────

@st.cache_data
#def load_data():
#    market = pd.read_sql("SELECT * FROM market_data", engine)
#    sentiment = pd.read_sql("SELECT * FROM daily_sentiment", engine)
#    features = pd.read_sql("SELECT * FROM master_features", engine)
#    market["date"] = pd.to_datetime(market["date"])
#    sentiment["date"] = pd.to_datetime(sentiment["date"])
#    features["date"] = pd.to_datetime(features["date"])
#    return market, sentiment, features
@st.cache_data
def load_data():
    base = os.path.join(os.path.dirname(__file__), "../../data/processed")
    market = pd.read_csv(f"{base}/market_data.csv")
    sentiment = pd.read_csv(f"{base}/daily_sentiment.csv")
    features = pd.read_csv(f"{base}/master_features.csv")
    market["date"] = pd.to_datetime(market["date"])
    sentiment["date"] = pd.to_datetime(sentiment["date"])
    features["date"] = pd.to_datetime(features["date"])
    return market, sentiment, features

market_df, sentiment_df, features_df = load_data()

TICKERS = sorted(market_df["ticker"].unique().tolist())

FEATURES = [
    "realized_vol_5d", "realized_vol_20d", "vol_lag1", "vol_lag2", "vol_lag5",
    "vol_momentum", "price_momentum_5d", "price_momentum_20d",
    "volume_surge", "log_return",
    "avg_sentiment", "sentiment_std", "positive_ratio", "negative_ratio",
    "news_volume", "sentiment_lag1", "sentiment_lag2",
    "sentiment_velocity", "news_volume_lag1"
]

# ─────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────

st.sidebar.header("⚙️ Controls")
selected_ticker = st.sidebar.selectbox("Select Ticker", TICKERS)
lookback_days = st.sidebar.slider("Lookback days", 30, 365, 90)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Performance")
st.sidebar.metric("XGBoost AUC", "0.813", delta="baseline")
st.sidebar.metric("TFT AUC", "0.720", delta="-0.093")
st.sidebar.markdown("---")
st.sidebar.markdown("### 🗂️ Data")
st.sidebar.metric("Tickers tracked", len(TICKERS))
st.sidebar.metric("Trading days", len(market_df[market_df["ticker"] == selected_ticker]))

# ─────────────────────────────────────────
# Section 1 — Price + Volatility overview
# ─────────────────────────────────────────

st.header(f"📉 {selected_ticker} — Price & Volatility")

ticker_market = market_df[market_df["ticker"] == selected_ticker].sort_values("date")
ticker_market = ticker_market.tail(lookback_days)

col1, col2, col3, col4 = st.columns(4)

latest = ticker_market.iloc[-1]
prev   = ticker_market.iloc[-2]

col1.metric("Latest Close",
            f"${latest['close']:.2f}",
            f"{((latest['close']/prev['close'])-1)*100:.2f}%")
col2.metric("Realized Vol (5d)",
            f"{latest['realized_vol_5d']:.1%}",
            f"{latest['realized_vol_5d'] - prev['realized_vol_5d']:.1%}")
col3.metric("Realized Vol (20d)",
            f"{latest['realized_vol_20d']:.1%}")
# Show recent spike activity instead of latest label
recent_spikes = ticker_market.tail(30)["vol_spike_2d"].sum()
col4.metric(
    "Spikes (last 30d)",
    f"{int(recent_spikes)} days",
    delta=f"{recent_spikes/30:.0%} spike rate"
)

# Price chart
fig_price = go.Figure()
fig_price.add_trace(go.Scatter(
    x=ticker_market["date"], y=ticker_market["close"],
    name="Close Price", line=dict(color="#2196F3", width=2)
))

# Highlight spike days
spike_days = ticker_market[ticker_market["vol_spike_2d"] == 1]
fig_price.add_trace(go.Scatter(
    x=spike_days["date"], y=spike_days["close"],
    mode="markers", name="Volatility Spike",
    marker=dict(color="red", size=8, symbol="circle")
))

fig_price.update_layout(
    title=f"{selected_ticker} Price with Volatility Spike Labels",
    xaxis_title="Date", yaxis_title="Price ($)",
    height=350, template="plotly_dark"
)
st.plotly_chart(fig_price, use_container_width=True)

# Volatility chart
fig_vol = go.Figure()
fig_vol.add_trace(go.Scatter(
    x=ticker_market["date"], y=ticker_market["realized_vol_5d"],
    name="5-day Vol", line=dict(color="#FF9800", width=2)
))
fig_vol.add_trace(go.Scatter(
    x=ticker_market["date"], y=ticker_market["realized_vol_20d"],
    name="20-day Vol", line=dict(color="#9C27B0", width=1.5, dash="dash")
))

# Shade spike zones
for _, row in spike_days.iterrows():
    fig_vol.add_vrect(
        x0=row["date"], x1=row["date"] + pd.Timedelta(days=2),
        fillcolor="red", opacity=0.15, line_width=0
    )

fig_vol.update_layout(
    title="Realized Volatility — Spike Zones Highlighted",
    xaxis_title="Date", yaxis_title="Annualized Volatility",
    height=300, template="plotly_dark"
)
st.plotly_chart(fig_vol, use_container_width=True)

# ─────────────────────────────────────────
# Section 2  Sentiment signals
# ─────────────────────────────────────────

st.header("📰 News Sentiment Signals")

ticker_sentiment = sentiment_df[sentiment_df["ticker"] == selected_ticker].sort_values("date")
ticker_sentiment = ticker_sentiment.tail(lookback_days)

col1, col2 = st.columns(2)

with col1:
    fig_sent = go.Figure()
    colors = ["red" if s < 0 else "green" for s in ticker_sentiment["avg_sentiment"]]
    fig_sent.add_trace(go.Bar(
        x=ticker_sentiment["date"],
        y=ticker_sentiment["avg_sentiment"],
        marker_color=colors,
        name="Daily Sentiment"
    ))
    fig_sent.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
    fig_sent.update_layout(
        title="Daily Sentiment Score",
        height=300, template="plotly_dark"
    )
    st.plotly_chart(fig_sent, use_container_width=True)

with col2:
    fig_vol_news = go.Figure()
    fig_vol_news.add_trace(go.Bar(
        x=ticker_sentiment["date"],
        y=ticker_sentiment["news_volume"],
        marker_color="#03A9F4",
        name="News Volume"
    ))
    fig_vol_news.update_layout(
        title="Daily News Volume",
        height=300, template="plotly_dark"
    )
    st.plotly_chart(fig_vol_news, use_container_width=True)

# ─────────────────────────────────────────
# Section 3 Sentiment vs Volatility correlation
# ─────────────────────────────────────────

st.header("Sentiment vs Volatility Relationship")

merged = pd.merge(
    ticker_market[["date", "realized_vol_5d", "vol_spike_2d"]],
    ticker_sentiment[["date", "avg_sentiment", "news_volume"]],
    on="date", how="inner"
)

col1, col2 = st.columns(2)

with col1:
    fig_scatter = px.scatter(
        merged, x="avg_sentiment", y="realized_vol_5d",
        color="vol_spike_2d",
        color_continuous_scale=["green", "red"],
        title="Sentiment vs Realized Volatility",
        labels={"avg_sentiment": "Sentiment Score",
                "realized_vol_5d": "Realized Vol (5d)",
                "vol_spike_2d": "Spike"},
        template="plotly_dark", height=350
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with col2:
    # Rolling correlation
    merged = merged.sort_values("date")
    merged["rolling_corr"] = (
        merged["avg_sentiment"]
        .rolling(20)
        .corr(merged["realized_vol_5d"])
    )
    fig_corr = go.Figure()
    fig_corr.add_trace(go.Scatter(
        x=merged["date"], y=merged["rolling_corr"],
        line=dict(color="#E91E63", width=2),
        name="20-day Rolling Correlation"
    ))
    fig_corr.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
    fig_corr.update_layout(
        title="Rolling Correlation: Sentiment vs Volatility",
        height=350, template="plotly_dark"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# ─────────────────────────────────────────
# Section 4 Cross-ticker comparison
# ─────────────────────────────────────────

st.header("Cross-Ticker Spike Frequency")

spike_summary = features_df.groupby("ticker").agg(
    spike_rate=("vol_spike_2d", "mean"),
    avg_vol=("realized_vol_5d", "mean"),
    avg_sentiment=("avg_sentiment", "mean")
).reset_index()

spike_summary["spike_rate_pct"] = spike_summary["spike_rate"] * 100

col1, col2 = st.columns(2)

with col1:
    fig_bar = px.bar(
        spike_summary.sort_values("spike_rate_pct", ascending=True),
        x="spike_rate_pct", y="ticker",
        orientation="h",
        title="Volatility Spike Rate by Ticker (%)",
        color="spike_rate_pct",
        color_continuous_scale="RdYlGn_r",
        template="plotly_dark", height=350
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    fig_bubble = px.scatter(
        spike_summary,
        x="avg_sentiment", y="avg_vol",
        size="spike_rate_pct", color="ticker",
        title="Avg Sentiment vs Avg Volatility (bubble = spike rate)",
        template="plotly_dark", height=350
    )
    st.plotly_chart(fig_bubble, use_container_width=True)

# ─────────────────────────────────────────
# Section 5 Feature importance
# ─────────────────────────────────────────

st.header("Model Insights")

col1, col2 = st.columns(2)

with col1:
    st.subheader("XGBoost Feature Importance")
    try:
        img = open("data/processed/feature_importance.png", "rb").read()
        st.image(img)
    except:
        st.info("Run xgboost_model.py first to generate this chart")

with col2:
    st.subheader("Temporal Attention Model")
    try:
        img = open("data/processed/tft_interpretation.png", "rb").read()
        st.image(img)
    except:
        st.info("Run tft_model.py first to generate this chart")

# ─────────────────────────────────────────
# Footer
# ─────────────────────────────────────────

st.markdown("---")
st.markdown(
    "Built with XGBoost + Temporal Attention Model | "
    "Data: Yahoo Finance + Synthetic Sentiment | "
    "Stack: Python · PostgreSQL · MLflow · Streamlit | " 
    "Project by Shruti"
)