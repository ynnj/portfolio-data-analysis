import yfinance as yf
from datetime import datetime

def get_ticker_price(ticker: str, time: str = "15:30", date: str = None):
    """
    Fetches the stock price of a given ticker at a specified time and its daily high/low range.
    
    Parameters:
        ticker (str): Stock ticker symbol.
        time (str): Time in HH:MM format (24-hour clock) for fetching historical price.
        date (str): Date in YYYY-MM-DD format (optional, defaults to today if not provided).
        
    Returns:
        dict: Contains the stock price at the specified time, daily high, and daily low.
    """
    if date is None:
        date = datetime.today().strftime('%Y-%m-%d')
    
    stock = yf.Ticker(ticker)
    
    # Fetch intraday historical data
    historical_data = stock.history(period="1d", interval="1m")
    
    if historical_data.empty:
        return {"error": "No data available for the specified ticker and date."}
    
    # Find the closest timestamp
    target_time = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
    historical_data.index = historical_data.index.tz_localize(None)  # Remove timezone
    
    # Get the closest available timestamp using asof()
    nearest_time = historical_data.index[historical_data.index.get_indexer([target_time], method='nearest')][0]
    price_at_time = historical_data.loc[nearest_time]
    
    # Get daily high and low
    daily_high = historical_data['High'].max()
    daily_low = historical_data['Low'].min()
    
    return {
        "ticker": ticker,
        "date": date,
        "time": time,
        "price_at_time": round(price_at_time['Close'], 2),
        "daily_high": round(daily_high, 2),
        "daily_low": round(daily_low, 2)
    }

# Example usage:
# result = get_ticker_price("AAPL", "10:30")
# print(result)


# Example usage:
result = get_ticker_price("AAPL", "10:30","2025-02-10")
print(result)
