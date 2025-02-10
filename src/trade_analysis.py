import sqlite3
import pandas as pd
import os
from datetime import datetime

# Connect to SQLite
DB_PATH = os.path.join(os.path.dirname(__file__), "../data/trades.db")
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Ensure trade_metrics table has a date column
cursor.execute('''
    CREATE TABLE IF NOT EXISTS trade_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT,
        total_trades INTEGER,
        win_rate REAL,
        profit_factor REAL,
        average_win REAL,
        average_loss REAL,
        average_holding_period REAL,
        cumulative_pnl REAL,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
''')

# Load merged trades
df = pd.read_sql("SELECT * FROM merged_trades", conn)

# Ensure data exists
if df.empty:
    print("⚠️ No trades found.")
    conn.close()
    exit()

# Convert necessary columns
df['realized_pnl'] = pd.to_numeric(df['realized_pnl'])
df['execution_time_sell'] = pd.to_datetime(df['execution_time_sell'])

# Compute Cumulative Net P&L
df = df.sort_values(by="execution_time_sell")
df['cumulative_pnl'] = df['realized_pnl'].cumsum()

# Compute key statistics
total_trades = len(df)
winning_trades = len(df[df['realized_pnl'] > 0])
losing_trades = len(df[df['realized_pnl'] < 0])
win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

average_win = df[df['realized_pnl'] > 0]['realized_pnl'].mean() if winning_trades > 0 else 0
average_loss = df[df['realized_pnl'] < 0]['realized_pnl'].mean() if losing_trades > 0 else 0
profit_factor = abs(average_win / average_loss) if average_loss != 0 else float('inf')

average_holding_period = df['holding_period'].mean()
cumulative_pnl = df['cumulative_pnl'].iloc[-1]  # Last value of cumulative P&L

# Get today's date
today = datetime.today().strftime('%Y-%m-%d')

# Insert new metrics
cursor.execute('''
    INSERT INTO trade_metrics (date, total_trades, win_rate, profit_factor, average_win, average_loss, average_holding_period, cumulative_pnl)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
''', (today, total_trades, win_rate, profit_factor, average_win, average_loss, average_holding_period, cumulative_pnl))

conn.commit()
conn.close()

print("✅ Trade metrics updated for", today)
