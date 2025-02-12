import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Define date range (adjustable)
start_date = "2000-01-01"  # Adjust start date as needed
end_date = "2025-02-12"    # Set to today's date or another desired endpoint

# Download SPY market data
spy = yf.download("SPY", start=start_date, end=end_date)

# Check if data exists
if spy.empty:
    raise ValueError("SPY data download failed! No data found for given date range.")

# Flatten multi-index columns if needed
if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = spy.columns.get_level_values(0)

print("SPY Data Columns:", spy.columns)  # Debug print

# Ensure correct price column
price_column = "Adj Close" if "Adj Close" in spy.columns else "Close"

# Generate market features
spy["returns"] = spy[price_column].pct_change().fillna(0)  # Daily returns
spy["volatility"] = spy["returns"].rolling(5).std().fillna(0)  # 5-day rolling volatility
spy["trend"] = spy["returns"].rolling(10).mean().fillna(0)  # 10-day momentum
spy["log_volume"] = np.log(spy["Volume"] + 1)  # Log volume (proxy for market activity)

# Drop NaN values (important before clustering)
spy.dropna(inplace=True)

# Select features for clustering
features = spy[["volatility", "trend", "log_volume"]]

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # 3 market regimes
spy["market_regime"] = kmeans.fit_predict(features)

# Plot Market Regimes
plt.figure(figsize=(12, 6))
plt.scatter(spy.index, spy["returns"], c=spy["market_regime"], cmap="viridis", alpha=0.5)
plt.title("SPY Market Regimes (Clustered)")
plt.xlabel("Date")
plt.ylabel("Daily Returns")
plt.colorbar(label="Market Regime")
plt.show()
