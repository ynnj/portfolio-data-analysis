import streamlit as st
import pandas as pd
import sqlite3
import os

# Load trades
db_path = os.path.join(os.path.dirname(__file__), "../data/trades.db")
conn = sqlite3.connect(db_path)
df = pd.read_sql("SELECT * FROM trades", conn)
conn.close()

# Streamlit App
st.title("📊 Trade Performance Dashboard")
st.write(df)

# Win/Loss Pie Chart
win_count = len(df[df['action'] == 'BUY'])
loss_count = len(df[df['action'] == 'SELL'])
st.write("### Win/Loss Breakdown")
st.bar_chart({"Wins": win_count, "Losses": loss_count})
