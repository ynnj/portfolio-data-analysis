import os
import subprocess

print("🚀 Running Trade Data Pipeline...")

# Step 1: Fetch and store trades
print("📡 Fetching trade executions from IBKR...")
subprocess.run(["python3", "src/store_trades.py"])

# Step 2: Analyze trades
print("📊 Running trade analysis...")
subprocess.run(["python3", "src/analyze_trades.py"])

# Step 3: Launch the Streamlit dashboard
print("📈 Opening trade dashboard...")
subprocess.run(["streamlit", "run", "dashboard/dashboard.py"])

print("✅ Process Complete!")
