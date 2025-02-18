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


st.title("📊 Portfolio Data Analysis")
# Set Streamlit page config
# st.set_page_config(page_title="Trade Performance Dashboard", layout="wide",initial_sidebar_state="expanded")

# Sidebar navigation
view = st.sidebar.radio("Select a view:", ("Dashboard", "Trade Analysis"))

if view == "Dashboard":
    st.title("📊 Trade Performance Dashboard")
    # Display cumulative PnL graph
    dashboard.plot_cumulative_pnl(df)
    
    # # Plot Average PnL per Day and Hour
    # day_chart, hour_chart = dashboard.plot_pnl_per_day_and_hour(df)
    # st.altair_chart(day_chart)
    # st.altair_chart(hour_chart)
    
    # # Display assets and asset type tables
    # assets_df, asset_type_df = dashboard.plot_assets(df)
    # st.dataframe(assets_df)
    # st.dataframe(asset_type_df)

elif view == "Trade Analysis":
    st.title("Trade Analysis")
    
    # Run clustering and show analysis
    df_clustered = dashboard.display_metrics(df,df_metrics)
    st.dataframe(df_metrics)



# # Test database connection
# st.subheader("🔍 Checking Database Connection...")
# test_query = "SELECT name FROM sqlite_master WHERE type='table';"
# tables = fetch_data(test_query)

# if tables is not None:
#     st.success("✅ Connection successful!")
#     st.write("### Available Tables:", tables)
#     st.dataframe(get_merged_trades_df())
#     st.dataframe(get_metrics_df())
# else:
#     st.error("❌ Failed to connect to the database.")
#     st.stop()


# # Load data
# try:
#     df=get_merged_trades_df()
#     latest_metrics=get_metrics_df()
# except Exception as e:
#     st.error(f"Error loading data: {e}")
#     st.stop()

# # Make sure to extract a scalar value from 'latest_metrics' DataFrame
# st.sidebar.metric("Total Trades", latest_metrics['total_trades'].iloc[0])






# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # my_project/
# sys.path.append(PROJECT_ROOT)


# import db_utils # Absolute import
# from src.get_metrics import *

# print(hello world)

# ... rest of your code

# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# SRC_DIR = os.path.join(SCRIPT_DIR, "src")  

# GET_STATS_SCRIPT = os.path.join(SRC_DIR, "trade_analysis.py")

# def run_pipeline(account_type):

#     # Step 1: Analyze trades
#     print("📊 Running trade analysis...")
#     # subprocess.run(["python3", GET_STATS_SCRIPT, account_type])

#     # # Step 2: Launch the Streamlit dashboard
#     # print("📈 Opening trade dashboard...")
#     # subprocess.run(["streamlit", "run", "dashboard/dashboard.py"])

#     print("✅ Process Complete!")

# # Run for both REAL and PAPER accounts
# run_pipeline("real")
# run_pipeline("paper")

# subprocess.run(["streamlit", "run", "dashboard/dashboard.py"])

