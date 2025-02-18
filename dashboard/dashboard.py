import streamlit as st
import pandas as pd
import plotly.express as px
import altair as alt
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import sys
import os
# Add the src folder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from get_metrics import *


def load_data(db_utils):
    metrics = db_utils.fetch_data("SELECT * FROM trade_metrics ORDER BY date DESC LIMIT 1")
    df = db_utils.fetch_data("SELECT * FROM merged_trades")
    
    if df is None or df.empty or metrics is None or metrics.empty:
        st.warning("⚠️ No trade data found.")
        st.stop()
        
    df['execution_time_sell'] = pd.to_datetime(df['execution_time_sell'])
    df = df.sort_values(by="execution_time_sell")
    return df, metrics

def plot_cumulative_pnl(df):
    df['cumulative_pnl'] = df['net_pnl'].cumsum()
    fig_cum_pnl = px.area(df, x="execution_time_sell", y="cumulative_pnl")
    fig_cum_pnl.update_xaxes(showgrid=False, title_text="", showticklabels=False)
    fig_cum_pnl.update_yaxes(showgrid=False, title_text="")
    fig_cum_pnl.update_traces(fillcolor="skyblue", opacity=0.7)
    fig_cum_pnl.update_layout(plot_bgcolor='white')
    st.plotly_chart(fig_cum_pnl, use_container_width=True)

def plot_pnl_per_day_and_hour(df):
    pnl_per_day = calculate_avg_pnl_by_weekday(df)
    pnl_per_hour = calculate_avg_pnl_by_hour(df)
    
    chart_width = 600
    chart_height = 500
    
    # Plot for Average PnL per Day
    day_chart = alt.Chart(pnl_per_day).mark_bar().encode(
        x=alt.X('average_daily_pnl', axis=alt.Axis(title="Average P&L")),
        y=alt.Y('weekday', axis=alt.Axis(title="Weekday")),
        color=alt.condition(alt.datum.average_daily_pnl > 0, alt.value('green'), alt.value('red'))
    ).properties(
        title='Average PnL per Weekday',
        width=chart_width,
        height=chart_height
    )
    
    # Plot for Average PnL per Hour
    hour_chart = alt.Chart(pnl_per_hour).mark_bar().encode(
        x=alt.X('Average PnL', axis=alt.Axis(title="Average P&L")),
        y=alt.Y('Hour:O', axis=alt.Axis(title="Hour of the Day")),
        color=alt.condition(alt.datum['Average PnL'] > 0, alt.value('green'), alt.value('red'))
    ).properties(
        title='Average PnL per Hour',
        width=chart_width,
        height=chart_height
    )
    
    return day_chart, hour_chart

def plot_assets(df):
    assets_df = calculate_pnl_by_symbol(df)
    asset_type_df = calculate_pnl_by_subcategory(df)
    
    return assets_df, asset_type_df

def display_metrics(df, metrics):

    # profit_factor = calculate_profit_factor(df)
    # avg_win = "${:,.0f}".format(calculate_avg_win(df))
    # trade_durations = calculate_trade_duration(df)
    # avg_loss = "-${:,.0f}".format(-1*calculate_avg_loss(df))

    win_rate = "{:,.0f}%".format(calculate_win_rate(metrics).iloc[0])
        # win_rate = "{:,.0f}%".format(calculate_win_rate(metrics))
    print(win_rate)
    tot_trades = metrics['total_trades'].iloc[0]  # Or use .sum() if it's a series you want to sum
    st.sidebar.metric("Total Trades", int(tot_trades))
    st.sidebar.metric("Win Rate", win_rate)
    # st.sidebar.metric("Profit Factor", profit_factor)
    # st.sidebar.metric("Avg Win", avg_win)
    # st.sidebar.metric("Avg Loss", avg_loss)

def trade_analysis(df):
    # Apply clustering and analysis here
    numeric_cols = ['net_pnl', 'holding_period', 'weekday']
    weekday_encoder = LabelEncoder()
    df['weekday'] = weekday_encoder.fit_transform(df['weekday'])
    df = pd.get_dummies(df, columns=['subcategory'])
    df = apply_kmeans(df, 3)
    
    return df
