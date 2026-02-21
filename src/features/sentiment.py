import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import torch
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()

engine = create_engine(os.getenv("DB_URL"))

# Load FinBERT
print("Loading FinBERT model...")
tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
model.eval()
print("FinBERT loaded")

def get_sentiment(text: str) -> dict:
    """Returns positive, negative, neutral scores for a piece of text."""
    if not text or len(text.strip()) < 10:
        return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    scores = softmax(outputs.logits, dim=1).squeeze()
    # label order: positive, negative, neutral
    return {
        "positive": scores[0].item(),
        "negative": scores[1].item(),
        "neutral": scores[2].item(),
        "sentiment_score": scores[0].item() - scores[1].item()  # net sentiment
    }

def process_sentiment_batch(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for i, row in df.iterrows():
        if i % 50 == 0:
            print(f"Processing article {i}/{len(df)}...")
        sentiment = get_sentiment(str(row["text"]))
        results.append(sentiment)

    sentiment_df = pd.DataFrame(results)
    return pd.concat([df.reset_index(drop=True), sentiment_df], axis=1)

def aggregate_daily_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate article-level sentiment to daily ticker-level features."""
    df["date"] = pd.to_datetime(df["published_at"]).dt.date

    daily = df.groupby(["ticker", "date"]).agg(
        avg_sentiment=("sentiment_score", "mean"),
        sentiment_std=("sentiment_score", "std"),      # disagreement signal
        positive_ratio=("positive", "mean"),
        negative_ratio=("negative", "mean"),
        news_volume=("sentiment_score", "count"),       # how many articles
        sentiment_velocity=("sentiment_score", lambda x: x.diff().mean())  # rate of change
    ).reset_index()

    return daily

if __name__ == "__main__":
    print("Loading news from DB...")
    df = pd.read_sql("SELECT * FROM news_raw", engine)
    print(f"Loaded {len(df)} articles")

    print("Running FinBERT sentiment analysis...")
    df_sentiment = process_sentiment_batch(df)

    print("Aggregating to daily level...")
    daily_sentiment = aggregate_daily_sentiment(df_sentiment)

    print("Saving to DB...")
    daily_sentiment.to_sql("daily_sentiment", engine, if_exists="replace", index=False)
    print(f"Sentiment pipeline complete — {len(daily_sentiment)} daily records saved")