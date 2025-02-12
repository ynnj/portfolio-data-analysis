import sqlite3
import pandas as pd
import numpy as np
import os

# Connect to SQLite
DB_PATH = os.path.join(os.path.dirname(__file__), "../data/paper_all_transactions.db")
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Load merged trade data
df = pd.read_sql("SELECT * FROM merged_trades", conn)

if df.empty:
    print("⚠️ No merged trades found.")
    conn.close()
    exit()

# Ensure proper data types
df['net_pnl'] = pd.to_numeric(df['net_pnl'])
df['holding_period'] = pd.to_numeric(df['holding_period'])
df['execution_time_sell'] = pd.to_datetime(df['execution_time_sell'])

# Compute key metrics
total_trades = len(df)
winning_trades = df[df['net_pnl'] > 0]
losing_trades = df[df['net_pnl'] <= 0]

win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
average_win = winning_trades['net_pnl'].mean() if not winning_trades.empty else 0
average_loss = losing_trades['net_pnl'].mean() if not losing_trades.empty else 0
profit_factor = abs(average_win / average_loss) if average_loss != 0 else 0
expectancy = (win_rate / 100 * average_win) + ((1 - win_rate / 100) * average_loss)
avg_holding_time = df['holding_period'].mean()

# 📌 **New Metrics**
returns = df['net_pnl']
sharpe_ratio = returns.mean() / returns.std() if returns.std() != 0 else 0
downside_returns = returns[returns < 0]
sortino_ratio = returns.mean() / downside_returns.std() if len(downside_returns) > 0 and downside_returns.std() != 0 else 0
max_drawdown = (df['net_pnl'].cumsum().cummax() - df['net_pnl'].cumsum()).max()
volatility_pnl = returns.std()

# Profit consistency (Profitable trading days / total trading days)
df['trade_date'] = df['execution_time_sell'].dt.date
daily_pnl = df.groupby('trade_date')['net_pnl'].sum()
profitable_days = (daily_pnl > 0).sum()
profit_consistency = (profitable_days / len(daily_pnl)) * 100 if len(daily_pnl) > 0 else 0

# Kelly Criterion
kelly_criterion = (win_rate / 100) - ((1 - win_rate / 100) / (average_win / abs(average_loss))) if average_loss != 0 else 0

# Win/Loss Streaks
df['win'] = df['net_pnl'] > 0
df['streak'] = df['win'].groupby((df['win'] != df['win'].shift()).cumsum()).cumsum()
longest_win_streak = df[df['win']]['streak'].max() if not df[df['win']].empty else 0
longest_loss_streak = df[~df['win']]['streak'].max()
longest_loss_streak = 0 if pd.isna(longest_loss_streak) else int(longest_loss_streak)


# Ensure metrics table exists
cursor.execute('''
    CREATE TABLE IF NOT EXISTS trade_metrics (
        date TEXT PRIMARY KEY,
        total_trades INTEGER,
        win_rate REAL,
        profit_factor REAL,
        average_win REAL,
        average_loss REAL,
        expectancy REAL,
        sharpe_ratio REAL,
        sortino_ratio REAL,
        max_drawdown REAL,
        volatility_pnl REAL,
        profit_consistency REAL,
        kelly_criterion REAL,
        avg_holding_time REAL,
        longest_win_streak INTEGER,
        longest_loss_streak INTEGER
    )
''')

# Insert metrics into the database
metrics_data = (
    str(pd.Timestamp.today().date()), total_trades, win_rate, profit_factor, 
    average_win, average_loss, expectancy, sharpe_ratio, sortino_ratio, 
    max_drawdown, volatility_pnl, profit_consistency, kelly_criterion, 
    avg_holding_time, longest_win_streak, longest_loss_streak
)

cursor.execute('''
    INSERT INTO trade_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(date) DO UPDATE SET
        total_trades=excluded.total_trades,
        win_rate=excluded.win_rate,
        profit_factor=excluded.profit_factor,
        average_win=excluded.average_win,
        average_loss=excluded.average_loss,
        expectancy=excluded.expectancy,
        sharpe_ratio=excluded.sharpe_ratio,
        sortino_ratio=excluded.sortino_ratio,
        max_drawdown=excluded.max_drawdown,
        volatility_pnl=excluded.volatility_pnl,
        profit_consistency=excluded.profit_consistency,
        kelly_criterion=excluded.kelly_criterion,
        avg_holding_time=excluded.avg_holding_time,
        longest_win_streak=excluded.longest_win_streak,
        longest_loss_streak=excluded.longest_loss_streak
''', metrics_data)

print(f"✅ Metrics updated for {metrics_data[0]}")
conn.commit()
conn.close()
