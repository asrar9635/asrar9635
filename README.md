import pandas as pd
import numpy as np
#import yfinance as yf
import matplotlib.pyplot as plt
#from transformers import pipeline
from sklearn.preprocessing import MinMaxScaler
from nltk.tokenize import word_tokenize
import nltk

# Download NLTK data
#nltk.download('punkt')

# --- STEP 1: Load Financial News Data ---
def load_financial_news():
    # Example dataset (replace this with real data sources)
    data = {
        'date': ['2024-11-01', '2024-11-02', '2024-11-03'],
        'headline': [
            "Stock market hits a record high amid positive earnings reports.",
            "Tech stocks tumble after disappointing quarterly results.",
            "Federal Reserve signals rate cuts to support the economy."
        ]
    }
    news_df = pd.DataFrame(data)
    news_df['date'] = pd.to_datetime(news_df['date'])
    return news_df

# --- STEP 2: Perform Sentiment Analysis ---
def perform_sentiment_analysis(news_df):
    # Load pre-trained sentiment analysis pipeline (Hugging Face)
    sentiment_analyzer = pipeline("sentiment-analysis")
   
    # Analyze sentiment for each headline
    news_df['sentiment'] = news_df['headline'].apply(lambda x: sentiment_analyzer(x)[0]['label'])
    news_df['score'] = news_df['headline'].apply(lambda x: sentiment_analyzer(x)[0]['score'])
   
    # Convert sentiment labels to numerical values
    news_df['sentiment_numeric'] = news_df['sentiment'].map({'POSITIVE': 1, 'NEGATIVE': -1, 'NEUTRAL': 0})
    return news_df

# --- STEP 3: Fetch Stock Price Data ---
def fetch_stock_prices(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data.reset_index(inplace=True)
    return stock_data

# --- STEP 4: Merge and Visualize Data ---
def merge_and_visualize_data(stock_data, news_df):
    # Merge stock prices with sentiment data
    merged_data = pd.merge(stock_data, news_df, left_on='Date', right_on='date', how='left')
   
    # Plot stock prices and sentiment
    plt.figure(figsize=(12, 6))
    plt.plot(merged_data['Date'], merged_data['Close'], label="Stock Price", color='blue')
    plt.scatter(merged_data['Date'], merged_data['sentiment_numeric'] * 50 + merged_data['Close'], label="Sentiment", color='red')
    plt.title("Stock Prices vs Sentiment")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

    return merged_data

# --- MAIN FUNCTION ---
if __name__ == "__main__":
    # 1. Load financial news data
    news_df = load_financial_news()
    print("Financial News Data:")
    print(news_df)

    # 2. Perform sentiment analysis
   # news_df = perform_sentiment_analysis(news_df)
    print("\nSentiment Analysis Results:")
    print(news_df)

    # 3. Fetch stock price data
    #stock_data = fetch_stock_prices(ticker="AAPL", start_date="2024-11-01", end_date="2024-11-04")
    print("\nStock Price Data:")
    print("price")
    print("data")

    # 4. Merge and visualize data
    #merged_data = merge_and_visualize_data(stock data, news df)
    print("\nMerged Data:")
    print("merged")
    print("data")
