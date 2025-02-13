import os
import subprocess
import time
print("🚀 Running Trade Data Pipeline...")

# Step 1: Fetch and store trades
# print("📡 Fetching trade executions from IBKR...")
# subprocess.run(["python3", "data/get_transactions2.py"])
# subprocess.run(["python3", "data/process_transactions2.py"])


# Step 2: Analyze trades
print("📊 Running trade analysis...")
subprocess.run(["python3", "src/trade_analysis.py"])

# Step 3: Launch the Streamlit dashboard
print("📈 Opening trade dashboard...")
subprocess.run(["streamlit", "run", "dashboard/dashboard.py"])

print("✅ Process Complete!")

# ✅ News sentiment (Optional): Use NLP to analyze financial news
# ✅ Volatility regime: Is this in a low/high volatility market?