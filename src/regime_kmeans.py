import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Define date range
start_date = "2015-01-01"
end_date = "2025-02-12"

# Download SPY and Tesla market data
spy = yf.download("SPY", start=start_date, end=end_date)
tsla = yf.download("TSLA", start=start_date, end=end_date)

# Check if data exists
if spy.empty or tsla.empty:
    raise ValueError("SPY or TSLA data download failed! Check the date range.")

# Flatten multi-index columns if needed
if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = spy.columns.get_level_values(0)
if isinstance(tsla.columns, pd.MultiIndex):
    tsla.columns = tsla.columns.get_level_values(0)

# Detect the correct price column
price_column_spy = "Adj Close" if "Adj Close" in spy.columns else "Close"
price_column_tsla = "Adj Close" if "Adj Close" in tsla.columns else "Close"

# Generate market features for SPY
spy["returns"] = spy[price_column_spy].pct_change().fillna(0)  # Daily returns
spy["volatility"] = spy["returns"].rolling(5).std().fillna(0)  # 5-day rolling volatility
spy["trend"] = spy["returns"].rolling(10).mean().fillna(0)  # 10-day momentum
spy["log_volume"] = np.log(spy["Volume"] + 1)  # Log volume (proxy for market activity)

# Drop NaN values before clustering
spy.dropna(inplace=True)

# Select features for clustering
features = spy[["volatility", "trend", "log_volume"]]

# Apply K-Means clustering with 5 clusters
kmeans = KMeans(n_clusters=5, random_state=42)  # Increased clusters for better granularity
spy["market_regime"] = kmeans.fit_predict(features)

# Merge Tesla data with SPY regimes
tsla = tsla[[price_column_tsla]].rename(columns={price_column_tsla: "TSLA Price"})
tsla = tsla.merge(spy[[price_column_spy, "market_regime"]], left_index=True, right_index=True, how="left")

# Plot Tesla & SPY Prices with Market Regimes
plt.figure(figsize=(12, 6))

# Plot Tesla's price
plt.plot(tsla.index, tsla["TSLA Price"], color="black", label="Tesla Price", linewidth=1.5)

# Plot SPY price
plt.plot(spy.index, spy[price_column_spy], color="blue", label="SPY Price", linewidth=1.5, alpha=0.7)

# Shade background based on SPY market regimes
regime_colors = {0: "lightgreen", 1: "lightcoral", 2: "lightblue", 3: "gold", 4: "purple"}

for regime in spy["market_regime"].unique():
    subset = spy[spy["market_regime"] == regime]
    plt.axvspan(subset.index.min(), subset.index.max(), color=regime_colors[regime], alpha=0.3)

plt.title("Tesla & SPY Prices vs SPY Market Regimes")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()
