import yfinance as yf
from datetime import datetime
import pandas as pd

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

def get_asset_min_max(df, date, symbol):
    """
    Retrieves the min and max prices for a given asset (symbol) on a specific date,
    using Yahoo Finance if the data is not in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing transaction data.
        date (str or datetime.date): The date for which to retrieve min/max prices.
        symbol (str): The asset symbol.

    Returns:
        tuple: A tuple containing (min_price, max_price), or (None, None) if not found.
    """

    try:
        # Ensure the date is in datetime.date format
        if not isinstance(date, pd.Timestamp):
            date = pd.to_datetime(date).date()

        # Filter the DataFrame for the given date and symbol
        filtered_df = df[(df['TradeDate'] == date) & (df['symbol'] == symbol)]

        if not filtered_df.empty:
            # Calculate min and max prices from the DataFrame
            min_price = filtered_df[['price_buy','price_sell']].min().min()
            max_price = filtered_df[['price_buy','price_sell']].max().max()
            return min_price, max_price
        else:
            # If not found in DataFrame, try Yahoo Finance
            date_str = date.strftime("%Y-%m-%d")
            start_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            end_date = start_date + datetime.timedelta(days=1)
            yahoo_data = yf.download(symbol, start=start_date, end=end_date, progress=False) #added progress=False to suppress print.

            if not yahoo_data.empty:
                high = yahoo_data['High'][0]
                low = yahoo_data['Low'][0]
                return low, high  # Yahoo provides low, high, so return in the same order.
            else:
                return None, None  # No data found in Yahoo Finance

    except Exception as e:
        print(f"Error: {e}")
        return None, None  # Return None if an error occurs

# Example usage:
# result = get_ticker_price("AAPL", "10:30","2025-02-10")


# Sample Usage
# Create a sample DataFrame (replace with your actual data)
# data = {
#     'TradeDate': pd.to_datetime(['2023-10-26', '2023-10-27']).date,
#     'symbol': ['AAPL', 'MSFT'],
#     'price_buy': [150.0, 300.0],
#     'price_sell': [155.0, 310.0]
# }
# df = pd.DataFrame(data)

# # Example 1: Data exists in the DataFrame
# date1 = '2023-10-26'
# symbol1 = 'AAPL'
# min1, max1 = get_asset_min_max(df, date1, symbol1)
# if min1 is not None:
#     print(f"Example 1: {symbol1} on {date1} - Min: {min1}, Max: {max1}")
# else:
#     print(f"Example 1: No data found for {symbol1} on {date1}")

def get_ticker_range(df):
    """
    Retrieves the high and low price range for each ticker in the given DataFrame.
    
    Parameters:
        df (pd.DataFrame): A DataFrame containing 'TradeDate' and 'Symbol' columns.

    Returns:
        pd.DataFrame: A DataFrame with 'Symbol', 'TradeDate', 'High', and 'Low'.
    """
    results = []
    
    for _, row in df.iterrows():
        symbol = row['symbol']
        trade_date = row['TradeDate']
        
        # Fetch historical data for the given date
        stock = yf.Ticker(symbol)
        history = stock.history(start=trade_date, end=pd.to_datetime(trade_date) + pd.Timedelta(days=1))
        
        if not history.empty:
            high = history['High'].iloc[0]
            low = history['Low'].iloc[0]
            atr= high-low
        else:
            high, low, atr = None, None, None
        
        results.append({'symbol': symbol, 'TradeDate': trade_date, 'atr': atr})
    
    return pd.DataFrame(results)

# Example usage:
df = pd.DataFrame({
    'TradeDate': ['2025-02-12'],
    'symbol': ['TSLA']
})
result_df = get_ticker_range(df)
print(result_df)
