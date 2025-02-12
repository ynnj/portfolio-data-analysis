import pandas as pd
import requests
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import sqlite3
import os

# Download VADER lexicon (only needed once)
nltk.download("vader_lexicon")

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Function to fetch news (Yahoo Finance API as an example)
def fetch_financial_news():
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": "stock market",  # Change to a specific stock if needed
        "apiKey": "YOUR_NEWS_API_KEY",
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 20,
    }
    response = requests.get(url, params=params)
    data = response.json()
    return data["articles"]

# Process news and compute sentiment
def analyze_news():
    articles = fetch_financial_news()
    news_df = pd.DataFrame(articles)[["publishedAt", "title", "description"]]
    news_df["publishedAt"] = pd.to_datetime(news_df["publishedAt"])
    
    # Compute sentiment scores
    news_df["sentiment_score"] = news_df["title"].apply(lambda text: sia.polarity_scores(text)["compound"])
    
    return news_df

# Save news sentiment to database
def save_news_to_db(news_df):
    db_path = os.path.join(os.path.dirname(__file__), "../data/trades.db")
    conn = sqlite3.connect(db_path)
    
    # Store news sentiment
    news_df.to_sql("news_sentiment", conn, if_exists="replace", index=False)
    
    conn.commit()
    conn.close()
    print("✅ News sentiment saved to database.")

news_df = analyze_news()
save_news_to_db(news_df)
