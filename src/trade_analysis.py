import sqlite3
import pandas as pd
import os

# Connect to SQLite
db_path = os.path.join(os.path.dirname(__file__), "../data/trades.db")
conn = sqlite3.connect(db_path)

# Load data into a Pandas DataFrame
df = pd.read_sql("SELECT * FROM trades", conn)
conn.close()

# Ensure data exists
if df.empty:
    print("⚠️ No trades found.")
    exit()

# Convert price to numeric
df['execution_price'] = pd.to_numeric(df['execution_price'])

# Calculate key statistics
total_trades = len(df)
winning_trades = len(df[df['action'] == 'BUY'])  # Simplified example
losing_trades = total_trades - winning_trades
win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
average_price = df['execution_price'].mean()

# Print summary
print(f"📊 Trade Summary")
print(f"Total Trades: {total_trades}")
print(f"Winning Trades: {winning_trades} ({win_rate:.2f}%)")
print(f"Average Price: {average_price:.2f}")

# 0 16 * * * /usr/bin/python3 /path/to/store_trades.py
