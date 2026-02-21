# VolatilityForecaster - NLP-Augmented Market Risk Prediction
A machine learning system for predicting equity market volatility spikes 48 hours ahead using news sentiment signals and price-based technical features. The project combines natural language processing with time series modeling, deployed as an interactive analytics dashboard.

---

## Motivation

Predicting the direction of stock prices is largely intractable. Predicting periods of elevated turbulence, however, is a more tractable and arguably more useful problem. Risk managers, options traders, and portfolio managers need to know when markets are likely to become volatile, not necessarily which direction they will move. This project addresses that problem by treating volatility spike prediction as a binary classification task with a 48-hour forecast horizon.

---

## Project Architecture

    volatility-forecaster/
    |-- data/
    |   |-- raw/                  Raw API data
    |   |-- processed/            Feature tables, model artifacts, charts
    |-- src/
    |   |-- ingestion/
    |   |   |-- market_data.py    Pulls OHLCV data via yfinance, computes realized volatility
    |   |   |-- news_data.py      News ingestion and synthetic sentiment generation
    |   |-- features/
    |   |   |-- sentiment.py      FinBERT sentiment scoring and daily aggregation
    |   |   |-- build_features.py Merges all signals into master feature table
    |   |-- models/
    |   |   |-- xgboost_model.py  XGBoost classifier with TimeSeriesSplit CV
    |   |   |-- tft_model.py      Custom Temporal Attention Model in PyTorch
    |   |-- dashboard/
    |       |-- app.py            Streamlit analytics dashboard
    |-- mlruns/                   MLflow experiment tracking
    |-- requirements.txt
    |-- .env

---

## Data Pipeline

### Market Data

Daily OHLCV data for 8 US equities and ETFs (SPY, AAPL, MSFT, GOOGL, AMZN, TSLA, JPM, GS) is sourced via the yfinance library, covering a 2-year lookback window. The following features are computed at ingestion time.

Log returns are calculated as the natural log of the ratio of consecutive closing prices. This formulation is preferred over simple returns for its additive properties across time and better approximation of normality.

Realized volatility is computed as the rolling standard deviation of log returns over 5-day and 20-day windows, annualized by multiplication with the square root of 252 (trading days per year). This serves as both a feature and the basis for the target variable.

The target variable, vol_spike_2d, is a binary indicator equal to 1 when the 5-day realized volatility two days into the future exceeds the 90th historical percentile for that ticker. This operationalizes the forecasting objective as detection of anomalous turbulence events.

### Sentiment Data

News sentiment is modeled using FinBERT (ProsusAI/finbert), a BERT-based model fine-tuned on financial corpora. Unlike general-purpose sentiment models, FinBERT is sensitive to domain-specific language including terms such as headwinds, writedown, beat, and miss.

Sentiment scores are aggregated from article level to daily ticker level, producing the following features: average sentiment score, sentiment standard deviation (a proxy for market disagreement), sentiment velocity (rate of change), news volume, and the ratio of positive and negative articles.

Note: The current implementation uses synthetic sentiment data designed to replicate the statistical properties of real financial news, including return correlation, autocorrelation, earnings-period volume spikes, and episodic sentiment shocks. Replacing this with a real news provider (Alpha Vantage, Refinitiv) requires only a change to the ingestion script.

---

## Feature Engineering

The master feature table combines market and sentiment signals with additional derived features:

Lagged volatility features (1, 2, and 5 days) capture the persistence structure of volatility, a well-documented empirical regularity in financial markets. Volatility momentum, defined as the difference between short-term and long-term realized volatility, captures whether the market is entering or exiting a turbulent regime.

Price momentum over 5-day and 20-day windows captures trend signals. A volume surge ratio, computed as the ratio of daily volume to its 20-day moving average, identifies unusual trading activity that often precedes price dislocations.

Lagged sentiment features (1 and 2 days) allow the model to detect whether negative news sentiment precedes volatility with a delay, as is often the case when markets process information gradually.

---

## Models

### XGBoost Baseline

An XGBoost classifier was trained using 5-fold TimeSeriesSplit cross-validation, which respects the temporal ordering of data and prevents look-ahead bias. Class imbalance (approximately 10 percent of days are spike days) is addressed via the scale_pos_weight parameter set to the negative-to-positive class ratio.

Results:

    ROC-AUC   0.813
    Precision 0.601
    Recall    0.502
    F1        0.472

Six of the top fifteen features by importance are sentiment-derived, including sentiment_lag2, sentiment_velocity, avg_sentiment, sentiment_std, and positive_ratio. This indicates that news sentiment contributes measurable signal beyond price history alone.

### Temporal Attention Model

A custom sequence model was implemented in PyTorch, incorporating multi-head self-attention over a 20-day lookback window, gated residual connections, layer normalization, and a binary classification head. The architecture is conceptually aligned with the Temporal Fusion Transformer but implemented from scratch to avoid dependency constraints.

Results:

    ROC-AUC   0.720
    Precision 0.500
    Recall    0.250
    F1        0.333

The underperformance relative to XGBoost is consistent with the known data requirements of deep learning models. With approximately 3,400 training samples, the inductive biases of gradient boosted trees are better suited to the problem. The attention model is expected to outperform with substantially more historical data.

---

## Experiment Tracking

All experiments are tracked using MLflow. Each run records hyperparameters, cross-validation metrics per fold and averaged, feature importance visualizations as artifacts, and the serialized model. Experiments can be inspected via the MLflow UI.

    mlflow ui --port 5000

---

## Dashboard

An interactive Streamlit dashboard provides the following views:

Price and volatility overview with spike day annotations and shaded spike zones on the volatility chart. News sentiment signals including daily sentiment direction and news volume over time. Sentiment versus volatility relationship analysis including scatter plots and 20-day rolling correlation. Cross-ticker spike rate comparison and a sentiment versus average volatility bubble chart. Model insights with embedded feature importance and attention model interpretation charts.

    streamlit run src/dashboard/app.py

---

## Setup Instructions

Prerequisites: Python 3.10 or higher, PostgreSQL 15, Homebrew (Mac).

    git clone https://github.com/YOUR_USERNAME/volatility-forecaster.git
    cd volatility-forecaster

    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

    brew install postgresql@15
    brew services start postgresql@15
    psql postgres

Inside the psql shell:

    CREATE DATABASE volatility_db;
    CREATE USER voluser WITH PASSWORD 'volpass123';
    GRANT ALL PRIVILEGES ON DATABASE volatility_db TO voluser;
    GRANT ALL ON SCHEMA public TO voluser;

Create a .env file in the project root:

    DB_URL=postgresql://voluser:volpass123@localhost:5432/volatility_db

Run the pipeline in sequence:

    python src/ingestion/market_data.py
    python src/ingestion/news_data.py
    python src/features/build_features.py
    python src/models/xgboost_model.py
    python src/models/tft_model.py
    streamlit run src/dashboard/app.py

---

## Technology Stack

    Language        Python 3.12
    Database        PostgreSQL 15 via SQLAlchemy
    NLP             FinBERT (HuggingFace Transformers)
    ML Framework    XGBoost, PyTorch
    MLOps           MLflow
    Dashboard       Streamlit, Plotly
    Data Sources    yfinance, SEC EDGAR RSS

---

## Limitations and Future Work

The current sentiment data is synthetic. Replacing it with a real news provider would require an API key from Alpha Vantage, Refinitiv, or similar services.

An ablation study comparing model performance with and without sentiment features would formally isolate their marginal contribution to predictive accuracy.

The model could be extended to hourly data for more precise short-horizon forecasting, and to include VIX as an additional market-wide fear signal.

A backtesting engine computing Sharpe ratio and maximum drawdown on signals generated by the model would translate the ML metrics into the language of finance practitioners.

Production deployment would involve scheduled data ingestion via Apache Airflow, a REST API serving predictions via FastAPI, and model drift monitoring using population stability index.
