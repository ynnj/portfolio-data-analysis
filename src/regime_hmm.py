import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from hmmlearn.hmm import GaussianHMM
from datetime import timedelta

# Define date range
start_date = "2015-01-01"
end_date = "2025-01-01"

# Download SPY & TSLA data
spy = yf.download("SPY", start=start_date, end=end_date)
tsla = yf.download("TSLA", start=start_date, end=end_date)

# Ensure correct column names
price_column = "Adj Close" if "Adj Close" in spy.columns else "Close"

# Compute features for SPY
spy["returns"] = spy[price_column].pct_change().fillna(0)
spy["volatility"] = spy["returns"].rolling(5).std().fillna(0)
spy["trend"] = spy["returns"].rolling(10).mean().fillna(0)
spy["log_volume"] = np.log(spy["Volume"] + 1).fillna(0)

# Drop NaN values (important for HMM fitting)
spy.dropna(inplace=True)

# Prepare features for HMM clustering
features = spy[["returns", "volatility", "trend", "log_volume"]].dropna().values

# Apply Hidden Markov Model (HMM) for regime classification
n_regimes = 3  # Adjust number of regimes
hmm = GaussianHMM(n_components=n_regimes, covariance_type="diag", n_iter=2000, tol=1e-4, random_state=42)

# Initialize probabilities to avoid zero sums
hmm.startprob_ = np.full(n_regimes, 1.0 / n_regimes)
hmm.transmat_ = np.full((n_regimes, n_regimes), 1.0 / n_regimes)

# Fit the model
hmm.fit(features)

# Predict market regimes
spy["market_regime"] = hmm.predict(features)

# Define regime colors
hmm_colors = {0: "lightgreen", 1: "lightcoral", 2: "lightblue"}
spy["color"] = spy["market_regime"].map(hmm_colors)

# Prepare TSLA prices for plotting
tsla_price = tsla["Adj Close"] if "Adj Close" in tsla.columns else tsla["Close"]

# Create legend patches
regime_labels = {
    0: "Strong Bullish 📈",
    1: "Bearish 📉",
    2: "Sideways ⚖️"
}
legend_patches = [mpatches.Patch(color=hmm_colors[i], label=regime_labels[i]) for i in hmm_colors]

# Plot Market Regimes with SPY and TSLA prices
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot market regimes
scatter = ax1.scatter(spy.index, spy["returns"], c=spy["color"], alpha=0.5, label="SPY Returns")
ax1.set_xlabel("Date")
ax1.set_ylabel("SPY Returns")
ax1.set_title("SPY Market Regimes (HMM Clustering)")

# Add second y-axis for TSLA price
ax2 = ax1.twinx()
ax2.plot(tsla.index, tsla_price, color="black", label="TSLA Price", linewidth=1.5, alpha=0.8)
ax2.set_ylabel("TSLA Price")

# Add legend
ax1.legend(handles=legend_patches, title="Market Regimes", loc="upper left")
ax2.legend(loc="upper right")

plt.show()
