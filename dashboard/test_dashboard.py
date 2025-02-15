import streamlit as st
import pandas as pd
import sqlite3
import os
import plotly.express as px

# Set Streamlit page config
st.set_page_config(page_title="Trade Performance Dashboard", layout="wide")

# Connect to SQLite
DB_PATH = os.path.join(os.path.dirname(__file__), "../data/real_all_transactions.db")
conn = sqlite3.connect(DB_PATH)

# Load Metrics (Latest Snapshot)
metrics = pd.read_sql("SELECT * FROM trade_metrics ORDER BY date DESC LIMIT 1", conn)
df = pd.read_sql("SELECT * FROM merged_trades", conn)
conn.close()

# Ensure data exists
if df.empty or metrics.empty:
    st.warning("⚠️ No trade data found.")
    st.stop()

# Convert datetime columns
df['execution_time_sell'] = pd.to_datetime(df['execution_time_sell'])
df = df.sort_values(by="execution_time_sell")

# Compute cumulative P&L
df['cumulative_pnl'] = df['net_pnl'].cumsum()
latest_metrics = metrics.iloc[0]

# Extract trade date and weekday
df['trade_date'] = df['execution_time_sell'].dt.date
df['weekday'] = df['execution_time_sell'].dt.day_name()

# Compute average P&L per weekday
total_pnl_by_weekday = df.groupby('weekday')['net_pnl'].sum()
num_weekdays = df.groupby('weekday')['trade_date'].nunique()
avg_pnl_by_weekday = (total_pnl_by_weekday / num_weekdays).reset_index()
avg_pnl_by_weekday.columns = ['weekday', 'average_daily_pnl']
avg_pnl_by_weekday = avg_pnl_by_weekday.sort_values(by="average_daily_pnl", ascending=False)

# Compute average P&L per asset symbol
avg_pnl_by_symbol = df.groupby('symbol')['net_pnl'].mean().reset_index()
avg_pnl_by_symbol = avg_pnl_by_symbol.sort_values(by="net_pnl", ascending=False)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Trades", int(latest_metrics['total_trades']))
col2.metric("Win Rate (%)", f"{latest_metrics['win_rate']:.2f}")
col3.metric("Profit Factor", f"{latest_metrics['profit_factor']:.2f}")
col4.metric("PnL", f"{df['cumulative_pnl'].iloc[-1]:.2f}")
col5.metric("Max Drawdown ($)", f"{latest_metrics['max_drawdown']:.2f}")

# Sidebar filters
st.sidebar.header("🔍 Filters")
selected_symbols = st.sidebar.multiselect("Select Asset Symbols", df['symbol'].unique(), default=df['symbol'].unique())
date_range = st.sidebar.date_input("Select Date Range", [df['trade_date'].min(), df['trade_date'].max()])

# Apply filters
df_filtered = df[(df['symbol'].isin(selected_symbols)) & (df['trade_date'].between(date_range[0], date_range[1]))]

st.title("📊 Trade Performance Dashboard")

# Display Latest Key Metrics in an expander
with st.expander("📌 **Key Metrics**", expanded=True):
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Trades", int(latest_metrics['total_trades']))
    col2.metric("Win Rate (%)", f"{latest_metrics['win_rate']:.2f}")
    col3.metric("Profit Factor", f"{latest_metrics['profit_factor']:.2f}")
    col4.metric("Sharpe Ratio", f"{latest_metrics['sharpe_ratio']:.2f}")
    col5.metric("Max Drawdown ($)", f"{latest_metrics['max_drawdown']:.2f}")

# 📈 **Cumulative Net P&L Chart**
st.subheader("📈 Cumulative Net P&L")
fig_cum_pnl = px.line(
    df_filtered, x="execution_time_sell", y="cumulative_pnl",
    title="Cumulative Net P&L",
    labels={"execution_time_sell": "Date", "cumulative_pnl": "Cumulative P&L ($)"},
    line_shape="spline"
)
fig_cum_pnl.update_xaxes(title_text="Date", tickformat="%Y-%m-%d", tickangle=-45)
st.plotly_chart(fig_cum_pnl, use_container_width=True)

# 📊 **Average Net P&L by Weekday**
st.subheader("📊 Average Net P&L by Weekday")
fig_avg_pnl_weekday = px.bar(
    avg_pnl_by_weekday, x="weekday", y="average_daily_pnl",
    title="Average Daily Net P&L by Weekday",
    labels={"weekday": "Weekday", "average_daily_pnl": "Avg Daily Net P&L ($)"},
    text_auto=True, color="average_daily_pnl"
)
st.plotly_chart(fig_avg_pnl_weekday, use_container_width=True)

# 📊 **Average Net P&L by Asset Symbol**
st.subheader("📊 Average Net P&L by Asset Symbol")
fig_avg_pnl_symbol = px.bar(
    avg_pnl_by_symbol, x="symbol", y="net_pnl", 
    title="Average Net P&L by Asset Symbol",
    labels={"symbol": "Asset Symbol", "net_pnl": "Avg Net P&L ($)"},
    text_auto=True, color="net_pnl"
)
st.plotly_chart(fig_avg_pnl_symbol, use_container_width=True)

# 📋 **Trade Data Table**
st.subheader("📋 Recent Trades")
st.dataframe(
    df_filtered[['execution_time_sell', 'symbol', 'net_pnl', 'cumulative_pnl', 'holding_period']]
    .rename(columns={'execution_time_sell': 'Execution Date', 'net_pnl': 'Net P&L ($)', 'cumulative_pnl': 'Cum P&L ($)', 'holding_period': 'Holding (m)'})
    .style.applymap(lambda x: 'background-color: red' if isinstance(x, (int, float)) and x < 0 else '', subset=['Net P&L ($)'])
)

####### Calculate total PL per subcategory
pl_per_subcategory = df.groupby('subcategory')['net_pnl'].sum().reset_index()
pl_per_subcategory = pl_per_subcategory.rename(columns={'net_pnl': 'Total P/L'})

total_pl = pl_per_subcategory['Total P/L'].sum()
total_row = pd.DataFrame({'subcategory': ['Total'], 'Total P/L': [total_pl]})
pl_per_subcategory = pd.concat([pl_per_subcategory, total_row], ignore_index=True)
pl_per_subcategory['Total P/L'] = pl_per_subcategory['Total P/L'].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)

st.subheader("Total P/L per Subcategory")
st.dataframe(pl_per_subcategory)

####### DISPLAY WINNING AND LOSING TRADES
df_copy = df.copy()
df_copy['pnl_percent'] = ((df_copy['net_pnl'] / df_copy['price_buy']) * 100).replace([float('inf'), -float('inf')], 0)
# Sort by net P/L to get top winners and losers
trades_df_sorted = df_copy.sort_values('net_pnl', ascending=False)

top_3_winners = trades_df_sorted.head(3)[['net_pnl', 'pnl_percent']]
top_3_losers = trades_df_sorted.tail(3)[['net_pnl', 'pnl_percent']]

# Formatting
for df_copy in [top_3_winners, top_3_losers]:
    for col in df_copy.columns:
        df_copy[col] = df_copy[col].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)


st.subheader("Top 3 Winning Trades")
st.dataframe(top_3_winners)
st.dataframe(top_3_losers)

####### PERFORMANCE PER PERIOD
# Get today's date
today = pd.Timestamp.now().floor('D') # current time truncated to the start of the day

# Calculate start of week (Monday)
start_of_week = today - pd.Timedelta(days=today.weekday())

# Calculate start of month
start_of_month = today.replace(day=1)

# Calculate start of year
start_of_year = today.replace(month=1, day=1)

# Filter and calculate total P/L for each period
df.to_csv("output.csv", index=False)
pl_today = df[df['execution_time_sell'] >= today]['net_pnl'].sum()
pl_this_week = df[df['execution_time_sell'] >= start_of_week]['net_pnl'].sum()
pl_this_month = df[df['execution_time_sell'] >= start_of_month]['net_pnl'].sum()
pl_ytd = df[df['execution_time_sell'] >= start_of_year]['net_pnl'].sum()


# Create a DataFrame for the results
time_periods = ['Today', 'This Week', 'This Month', 'Year-to-Date']
pl_values = [pl_today, pl_this_week, pl_this_month, pl_ytd]

pl_over_time = pd.DataFrame({'Time Period': time_periods, 'Total P/L': pl_values})

#Formatting
pl_over_time['Total P/L'] = pl_over_time['Total P/L'].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)


st.subheader("Total P/L Over Time")
st.dataframe(pl_over_time)


####### CONSECUTIVE PERFORMANCE

def calculate_consecutive(series):
    """Calculates max consecutive wins/losses and profit/loss, along with trade counts."""
    wins = 0
    losses = 0
    max_wins = 0
    max_losses = 0
    max_wins_trades = 0  # Track trades during max win streak
    max_losses_trades = 0  # Track trades during max loss streak
    consecutive_profits = []
    consecutive_losses = []
    current_profit_streak = 0
    current_loss_streak = 0
    max_consecutive_profit = 0
    max_consecutive_loss = 0

    wins_trades = 0
    losses_trades = 0
    max_consecutive_profit_trades = 0
    max_consecutive_loss_trades = 0

    for i, pnl in enumerate(series):
        if pnl > 0:
            wins += 1
            losses = 0  # Reset losses
            wins_trades +=1
            losses_trades = 0
            current_profit_streak += pnl
            current_loss_streak = 0 # reset loss streak
            max_wins = max(max_wins, wins)
            if max_wins == wins:
                max_wins_trades = wins_trades
            consecutive_profits.append(current_profit_streak)
            max_consecutive_profit = max(max_consecutive_profit, current_profit_streak)
            if max_consecutive_profit == current_profit_streak:
                max_consecutive_profit_trades = wins_trades

        elif pnl < 0:
            losses += 1
            wins = 0  # Reset wins
            losses_trades +=1
            wins_trades = 0
            current_loss_streak += pnl
            current_profit_streak = 0 # reset profit streak
            max_losses = max(max_losses, losses)
            if max_losses == losses:
                max_losses_trades = losses_trades
            consecutive_losses.append(current_loss_streak)
            max_consecutive_loss = min(max_consecutive_loss, current_loss_streak) # min because loss is negative
            if max_consecutive_loss == current_loss_streak:
                max_consecutive_loss_trades = losses_trades
        else:
            wins = 0
            losses = 0
            wins_trades = 0
            losses_trades = 0
            current_profit_streak = 0
            current_loss_streak = 0

    return max_wins, max_losses, max_wins_trades, max_losses_trades, max_consecutive_profit, max_consecutive_loss, max_consecutive_profit_trades, max_consecutive_loss_trades

# Calculate metrics
total_trades = len(df)
net_pl = df['net_pnl'].sum()
max_drawdown = df['net_pnl'].cumsum().min()
max_wins, max_losses, max_wins_trades, max_losses_trades, max_consecutive_profit, max_consecutive_loss, max_consecutive_profit_trades, max_consecutive_loss_trades = calculate_consecutive(df['net_pnl'])

# Create the metrics DataFrame
metrics_data = {
    'Metric': ['Max Drawdown', 'Consecutive Wins', 'Consecutive Losses', 'Max Consecutive Profit', 'Max Consecutive Loss'],
    'Value': [max_drawdown, max_wins, max_losses, max_consecutive_profit, max_consecutive_loss],
    'Total Trades': ['', max_wins_trades, max_losses_trades, max_consecutive_profit_trades, max_consecutive_loss_trades] # Add trade counts
}
metrics_df = pd.DataFrame(metrics_data)

# Formatting
metrics_df['Value'] = metrics_df['Value'].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)

st.dataframe(metrics_df)


####### DURATION PERFORMANCE
df_copy = df.copy()

# Convert execution times to datetime if they aren't already
df_copy['execution_time_buy'] = pd.to_datetime(df_copy['execution_time_buy'])
df_copy['execution_time_sell'] = pd.to_datetime(df_copy['execution_time_sell'])


# Calculate trade duration
df_copy['duration'] = (df_copy['execution_time_sell'] - df_copy['execution_time_buy']).dt.total_seconds() / 3600  # In hours

# Separate winning and losing trades
winning_trades = df_copy[df_copy['net_pnl'] > 0]
losing_trades = df_copy[df_copy['net_pnl'] < 0]

# Calculate average durations
avg_duration_all = df_copy['duration'].mean()
avg_duration_wins = winning_trades['duration'].mean() if len(winning_trades) > 0 else 0 # Handle cases where there are no winning trades
avg_duration_losses = losing_trades['duration'].mean() if len(losing_trades) > 0 else 0 # Handle cases where there are no losing trades


# Create the duration DataFrame
duration_data = {
    'Trade Type': ['Winning Trades', 'Losing Trades', 'All Trades'],
    'Average Duration (Hours)': [avg_duration_wins, avg_duration_losses, avg_duration_all]
}
duration_df = pd.DataFrame(duration_data)

# Formatting
duration_df['Average Duration (Hours)'] = duration_df['Average Duration (Hours)'].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)

st.subheader("Average Trade Duration")
st.dataframe(duration_df)



####### MONEY MANAGEMENT

def calculate_kelly_criterion(win_rate, avg_win_loss_ratio):
    """Calculates the Kelly Criterion."""
    if avg_win_loss_ratio == 0 or win_rate == 0: # avoid division by zero
        return 0
    kelly_fraction = (win_rate * (avg_win_loss_ratio + 1) - 1) / avg_win_loss_ratio
    return max(0, min(1, kelly_fraction))  # Ensure kelly_fraction is between 0 and 1

def calculate_position_size(account_size, kelly_fraction, price_buy, risk_per_trade_percent):
    """Calculates the position size in shares."""
    if price_buy == 0: # avoid division by zero
        return 0
    position_size_shares = (account_size * kelly_fraction) / price_buy
    return position_size_shares

# Calculate metrics (replace with your actual account size)
account_size = 10000  # Example account size
win_rate = (df['net_pnl'] > 0).sum() / len(df) if len(df) > 0 else 0
avg_win_loss_ratio = abs(df[df['net_pnl'] > 0]['net_pnl'].mean() / df[df['net_pnl'] < 0]['net_pnl'].mean()) if len(df[df['net_pnl'] > 0]) > 0 and len(df[df['net_pnl'] < 0]) > 0 else 0

kelly_criterion = calculate_kelly_criterion(win_rate, avg_win_loss_ratio)
# Assume price_buy is the entry price of the next trade, you'll need to fetch this dynamically
# For demonstration, I'm using the mean of price_buy. You'll have to change this.
average_price_buy = df['price_buy'].mean() if len(df) > 0 else 0
position_size = calculate_position_size(account_size, kelly_criterion, average_price_buy, risk_per_trade_percent=0.01)

# Create the money management DataFrame
money_management_data = {
    'Metric': ['Kelly Criterion', 'Position Size (Shares)'],
    'Value': [kelly_criterion, position_size]
}
money_management_df = pd.DataFrame(money_management_data)

# Formatting
money_management_df['Value'] = money_management_df['Value'].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)

st.subheader("Money Management")
st.dataframe(money_management_df)



####### RISK/RETURN METRICS
avg_return_wins = winning_trades['net_pnl'].mean() if len(winning_trades) > 0 else 0
avg_risk_wins = winning_trades['net_pnl'].std() if len(winning_trades) > 0 else 0  # Risk is standard deviation
avg_return_losses = losing_trades['net_pnl'].mean() if len(losing_trades) > 0 else 0
avg_risk_losses = losing_trades['net_pnl'].std() if len(losing_trades) > 0 else 0

# All Trades
avg_return_all = df['net_pnl'].mean()
avg_risk_all = df['net_pnl'].std()

# Expected Return (assuming historical win rate is a good predictor)
win_rate = (df['net_pnl'] > 0).sum() / len(df) if len(df) > 0 else 0
expected_return = (win_rate * avg_return_wins) + ((1 - win_rate) * avg_return_losses)


# Create the risk/return DataFrame
risk_return_data = {
    'Trade Type': ['Winning Trades', 'Losing Trades', 'All Trades', 'Expected'],
    'Average Return': [avg_return_wins, avg_return_losses, avg_return_all, expected_return],
    'Average Risk (Standard Deviation)': [avg_risk_wins, avg_risk_losses, avg_risk_all, '']  # Risk for expected return is not usually calculated this way
}
risk_return_df = pd.DataFrame(risk_return_data)

# Formatting
for col in ['Average Return', 'Average Risk (Standard Deviation)']:
    if col in risk_return_df.columns:
        risk_return_df[col] = risk_return_df[col].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)


st.subheader("Risk/Return Metrics")
st.dataframe(risk_return_df)