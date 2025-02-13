import subprocess
import os

# Define the paths to your scripts inside the "data" folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")  # Path to the "data" folder

GET_TRANSACTIONS_SCRIPT = os.path.join(DATA_DIR, "get_transactions.py")
PROCESS_TRANSACTIONS_SCRIPT = os.path.join(DATA_DIR, "process_transactions.py")

def run_pipeline(account_type):
    print(f"🚀 Running data pipeline for {account_type.upper()} account...")

    # Step 1: Fetch new transactions
    print(f"📥 Running get_transactions.py for {account_type}...")
    subprocess.run(["python3", GET_TRANSACTIONS_SCRIPT, account_type])

    # Step 2: Process transactions
    print(f"🔄 Running process_transactions.py for {account_type}...")
    subprocess.run(["python3", PROCESS_TRANSACTIONS_SCRIPT, account_type])

    print(f"✅ Data pipeline run complete for {account_type.upper()}.\n")

# Run for both REAL and PAPER accounts
run_pipeline("real")
run_pipeline("paper")
