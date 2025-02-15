import pandas as pd
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "../data/real_all_transactions.db")

def load_data():
    """Loads trade data and metrics from the database."""
    conn = sqlite3.connect(DB_PATH)
    metrics = pd.read_sql("SELECT * FROM trade_metrics ORDER BY date DESC LIMIT 1", conn)
    df = pd.read_sql("SELECT * FROM merged_trades", conn)
    conn.close()
    
    if df.empty or metrics.empty:
        raise ValueError("No trade data found.")
    
    df['execution_time_sell'] = pd.to_datetime(df['execution_time_sell'])
    df = df.sort_values(by="execution_time_sell")
    df['cumulative_pnl'] = df['net_pnl'].cumsum()
    
    return df, metrics.iloc[0]

def calculate_avg_pnl_by_weekday(df):
    """Computes average daily P&L per weekday."""
    df['trade_date'] = df['execution_time_sell'].dt.date
    df['weekday'] = df['execution_time_sell'].dt.day_name()
    total_pnl_by_weekday = df.groupby('weekday')['net_pnl'].sum()
    num_weekdays = df.groupby('weekday')['trade_date'].nunique()
    avg_pnl_by_weekday = (total_pnl_by_weekday / num_weekdays).reset_index()
    avg_pnl_by_weekday.columns = ['weekday', 'average_daily_pnl']
    return avg_pnl_by_weekday.sort_values(by="average_daily_pnl", ascending=False)

def calculate_avg_pnl_by_symbol(df):
    """Computes average P&L per asset symbol."""
    avg_pnl_by_symbol = df.groupby('symbol')['net_pnl'].mean().reset_index()
    return avg_pnl_by_symbol.sort_values(by="net_pnl", ascending=False)

def calculate_total_pl_per_subcategory(df):
    """Computes total P&L per subcategory."""
    pl_per_subcategory = df.groupby('subcategory')['net_pnl'].sum().reset_index()
    pl_per_subcategory = pl_per_subcategory.rename(columns={'net_pnl': 'Total P/L'})
    total_pl = pl_per_subcategory['Total P/L'].sum()
    total_row = pd.DataFrame({'subcategory': ['Total'], 'Total P/L': [total_pl]})
    return pd.concat([pl_per_subcategory, total_row], ignore_index=True)

def calculate_performance_over_time(df):
    """Calculates P&L performance for different time periods."""
    today = pd.Timestamp.now().floor('D')
    start_of_week = today - pd.Timedelta(days=today.weekday())
    start_of_month = today.replace(day=1)
    start_of_year = today.replace(month=1, day=1)

    pl_today = df[df['execution_time_sell'] >= today]['net_pnl'].sum()
    pl_this_week = df[df['execution_time_sell'] >= start_of_week]['net_pnl'].sum()
    pl_this_month = df[df['execution_time_sell'] >= start_of_month]['net_pnl'].sum()
    pl_ytd = df[df['execution_time_sell'] >= start_of_year]['net_pnl'].sum()

    return pd.DataFrame({
        'Time Period': ['Today', 'This Week', 'This Month', 'Year-to-Date'],
        'Total P/L': [pl_today, pl_this_week, pl_this_month, pl_ytd]
    })

def calculate_consecutive_performance(df):
    """Calculates max consecutive wins/losses and their profits/losses."""
    wins = losses = max_wins = max_losses = 0
    current_profit_streak = current_loss_streak = 0
    max_consecutive_profit = max_consecutive_loss = 0

    for pnl in df['net_pnl']:
        if pnl > 0:
            wins += 1
            losses = 0
            current_profit_streak += pnl
            max_wins = max(max_wins, wins)
            max_consecutive_profit = max(max_consecutive_profit, current_profit_streak)
        elif pnl < 0:
            losses += 1
            wins = 0
            current_loss_streak += pnl
            max_losses = max(max_losses, losses)
            max_consecutive_loss = min(max_consecutive_loss, current_loss_streak)
        else:
            wins = losses = current_profit_streak = current_loss_streak = 0

    return {
        'Max Wins': max_wins,
        'Max Losses': max_losses,
        'Max Consecutive Profit': max_consecutive_profit,
        'Max Consecutive Loss': max_consecutive_loss
    }

def calculate_trade_duration(df):
    """Calculates the average duration of winning and losing trades."""
    df['execution_time_buy'] = pd.to_datetime(df['execution_time_buy'])
    df['execution_time_sell'] = pd.to_datetime(df['execution_time_sell'])
    df['duration'] = (df['execution_time_sell'] - df['execution_time_buy']).dt.total_seconds() / 3600
    
    winning_trades = df[df['net_pnl'] > 0]
    losing_trades = df[df['net_pnl'] < 0]

    return {
        'Winning Trades': winning_trades['duration'].mean(),
        'Losing Trades': losing_trades['duration'].mean(),
        'All Trades': df['duration'].mean()
    }

def calculate_kelly_criterion(win_rate, avg_win_loss_ratio):
    """Calculates the Kelly Criterion for optimal position sizing."""
    if avg_win_loss_ratio == 0 or win_rate == 0:
        return 0
    kelly_fraction = (win_rate * (avg_win_loss_ratio + 1) - 1) / avg_win_loss_ratio
    return max(0, min(1, kelly_fraction))

def calculate_money_management(df, account_size=10000):
    """Computes optimal position size based on Kelly Criterion."""
    win_rate = (df['net_pnl'] > 0).sum() / len(df) if len(df) > 0 else 0
    avg_win_loss_ratio = abs(df[df['net_pnl'] > 0]['net_pnl'].mean() / df[df['net_pnl'] < 0]['net_pnl'].mean()) if len(df[df['net_pnl'] < 0]) > 0 else 0
    kelly_criterion = calculate_kelly_criterion(win_rate, avg_win_loss_ratio)
    avg_price_buy = df['price_buy'].mean() if len(df) > 0 else 0
    position_size = (account_size * kelly_criterion) / avg_price_buy if avg_price_buy > 0 else 0

    return {
        'Kelly Criterion': kelly_criterion,
        'Position Size (Shares)': position_size
    }

def calculate_risk_return(df):
    """Computes risk-return metrics for winning and losing trades."""
    winning_trades = df[df['net_pnl'] > 0]
    losing_trades = df[df['net_pnl'] < 0]

    avg_return_wins = winning_trades['net_pnl'].mean() if len(winning_trades) > 0 else 0
    avg_risk_wins = winning_trades['net_pnl'].std() if len(winning_trades) > 0 else 0
    avg_return_losses = losing_trades['net_pnl'].mean() if len(losing_trades) > 0 else 0
    avg_risk_losses = losing_trades['net_pnl'].std() if len(losing_trades) > 0 else 0
    avg_return_all = df['net_pnl'].mean()
    avg_risk_all = df['net_pnl'].std()
    expected_return = ((df['net_pnl'] > 0).sum() / len(df)) * avg_return_wins + ((df['net_pnl'] < 0).sum() / len(df)) * avg_return_losses

    return pd.DataFrame({
        'Trade Type': ['Winning Trades', 'Losing Trades', 'All Trades', 'Expected'],
        'Average Return': [avg_return_wins, avg_return_losses, avg_return_all, expected_return],
        'Average Risk': [avg_risk_wins, avg_risk_losses, avg_risk_all, None]
    })

if __name__ == "__main__":
    df, latest_metrics = load_data()
    print("Total Trades:", latest_metrics['total_trades'])
    print("Win Rate:", latest_metrics['win_rate'])
    print("Profit Factor:", latest_metrics['profit_factor'])
    print("Cumulative PnL:", df['cumulative_pnl'].iloc[-1])
    print("Max Drawdown:", latest_metrics['max_drawdown'])
