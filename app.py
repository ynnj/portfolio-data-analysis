import os
import subprocess
import time


# Define the paths to your scripts inside the "data" folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, "src")  # Path to the "data" folder

GET_STATS_SCRIPT = os.path.join(SRC_DIR, "trade_analysis.py")

def run_pipeline(account_type):

    # Step 1: Analyze trades
    print("📊 Running trade analysis...")
    subprocess.run(["python3", GET_STATS_SCRIPT, account_type])

    # # Step 2: Launch the Streamlit dashboard
    # print("📈 Opening trade dashboard...")
    # subprocess.run(["streamlit", "run", "dashboard/dashboard.py"])

    print("✅ Process Complete!")

# Run for both REAL and PAPER accounts
run_pipeline("real")
# run_pipeline("paper")

subprocess.run(["streamlit", "run", "dashboard/dashboard.py"])


# ✅ News sentiment (Optional): Use NLP to analyze financial news
# ✅ Volatility regime: Is this in a low/high volatility market?