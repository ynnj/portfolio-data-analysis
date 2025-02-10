import streamlit as st
import pandas as pd
import sqlite3
import os
import plotly.express as px

# Connect to SQLite
DB_PATH = os.path.join(os.path.dirname(__file__), "../data/trades.db")
conn = sqlite3.connect(DB_PATH)

# Load Metrics (Historical Data)
metrics = pd.read_sql("SELECT * FROM trade_metrics ORDER BY date ASC", conn)
df = pd.read_sql("SELECT * FROM merged_trades", conn)
conn.close()

# Ensure data exists
if df.empty or metrics.empty:
    st.warning("⚠️ No trade data found.")
    st.stop()

# Convert date column to datetime
metrics['date'] = pd.to_datetime(metrics['date'])
df['execution_time_sell'] = pd.to_datetime(df['execution_time_sell'])
df = df.sort_values(by="execution_time_sell")
# Compute cumulative P&L
df['cumulative_pnl'] = df['realized_pnl'].cumsum()

# Streamlit Dashboard
st.title("📊 Trade Performance Dashboard")

# Display Latest Key Metrics
latest_metrics = metrics.iloc[-1]
col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)
col7, _ = st.columns([1, 2])
col1.metric("Total Trades", int(latest_metrics['total_trades']))
col2.metric("Win Rate (%)", f"{latest_metrics['win_rate']:.2f}")
col3.metric("Profit Factor", f"{latest_metrics['profit_factor']:.2f}")
col4.metric("Average Win ($)", f"{latest_metrics['average_win']:.2f}")
col5.metric("Average Loss ($)", f"{latest_metrics['average_loss']:.2f}")
col6.metric("Avg Holding Period (min)", f"{latest_metrics['average_holding_period']:.2f}")
col7.metric("Cumulative PL", f"{latest_metrics['average_holding_period']:.2f}")


# Cumulative Net P&L Over Time from merged_trades
fig_cum_pnl_trades = px.line(
    df, 
    x="execution_time_sell", 
    y="cumulative_pnl",
    title="Cumulative Net P&L Over Time",
    labels={"execution_time_sell": "Execution Date", "cumulative_pnl": "Cumulative P&L ($)"},
    line_shape="hv",  # Step-like line for financial data
)

# Improve x-axis formatting
fig_cum_pnl_trades.update_xaxes(title_text="Date", tickformat="%Y-%m-%d", tickangle=-45)
fig_cum_pnl_trades.update_yaxes(title_text="Cumulative P&L ($)")

# Display the chart
st.plotly_chart(fig_cum_pnl_trades)

# Show cumulative P&L table
st.write("### 📋 Cumulative Net P&L Table")
st.dataframe(df[['execution_time_sell', 'realized_pnl', 'cumulative_pnl']].rename(
    columns={'execution_time_sell': 'Execution Date', 'realized_pnl': 'Realized P&L ($)', 'cumulative_pnl': 'Cumulative P&L ($)'}
))

# Win/Loss Breakdown Pie Chart
win_loss_counts = df['win_loss'].value_counts()
fig_pie = px.pie(names=win_loss_counts.index, values=win_loss_counts.values, title="Win/Loss Breakdown")
st.plotly_chart(fig_pie)

# Cumulative Net P&L Over Time
fig_cum_pnl = px.line(metrics, x="date", y="cumulative_pnl", title="Cumulative Net P&L Over Time")
st.plotly_chart(fig_cum_pnl)

# Win Rate Over Time
fig_win_rate = px.line(metrics, x="date", y="win_rate", title="Win Rate Over Time")
st.plotly_chart(fig_win_rate)

# Profit Factor Over Time
fig_profit_factor = px.line(metrics, x="date", y="profit_factor", title="Profit Factor Over Time")
st.plotly_chart(fig_profit_factor)

st.write("### Trade Data Table")
st.dataframe(df)
