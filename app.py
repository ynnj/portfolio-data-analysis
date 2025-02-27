import os
import sys
import subprocess
import time
import streamlit as st
import pandas as pd
from db_utils import fetch_data, execute_query, get_merged_trades_df, get_metrics_df
from src.get_metrics import *
from dashboard import dashboard

# Load data
try:
    df=get_merged_trades_df()
    df_metrics=get_metrics_df()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

st.set_page_config(layout="wide")

# Sidebar Navigation
st.sidebar.header('📊 Key Metrics')
dashboard.display_metrics_sidebar(df,df_metrics)
view = st.sidebar.radio("Select a view:", ("Dashboard", "Trades", "Calendar","Trade Analysis"))

if view == "Dashboard":
    st.title("📊 Trade Performance Dashboard")

    dashboard.display_metrics_dashboard(df, df_metrics)
    dashboard.display_aggrid_pivot(df)
    dashboard.plot_cumulative_pnl(df)
    
    # # Plot Average PnL per Day and Hour
    day_chart, hour_chart = dashboard.plot_pnl_per_day_and_hour(df)
    col1, col2 = st.columns(2)  # Create two columns
    with col1:
        st.altair_chart(day_chart)
    with col2:
        st.altair_chart(hour_chart)
    
    # Display assets and asset type tables
    assets_df, asset_type_df = dashboard.plot_assets(df)
    col1, col2 = st.columns(2)  # Create two columns
    with col1:
        st.dataframe(assets_df, hide_index=True)
    with col2:
        st.dataframe(asset_type_df, hide_index=True)

elif view == "Trades":
    st.title("Trades")
    dashboard.display_transactions(df)

elif view == "Calendar":
    st.title("Calendar")
    dashboard.display_calendar_metrics(df)
    dashboard.display_calendar(df)

elif view == "Trade Analysis":
    st.title("Trade Analysis")
    dashboard.display_trade_clusters(df)


