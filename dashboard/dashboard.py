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
    df['execution_time_sell'] = pd.to_datetime(df['execution_time_sell'])

    chart = alt.Chart(df).mark_line().encode(
        x=alt.X('execution_time_sell:T', axis=alt.Axis(title=None, labels=True, ticks=True)),  # Remove x-axis labels and ticks
        y=alt.Y('cumulative_pnl:Q', axis=alt.Axis(title=None, labels=True, ticks=True))      # Remove y-axis labels and ticks
    ).properties(
        width=600  # Adjust width as needed
    )

    st.altair_chart(chart, use_container_width=True)


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


    assets_df['pnl'] = assets_df['pnl'].apply(lambda x: f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}")
    assets_df['pnl %'] = assets_df['pnl %'].map('{:,.0f}%'.format)
    assets_df['weighted'] = (assets_df['weighted'] * 100).map('{:,.0f}%'.format)

    assets_df = assets_df.rename(columns={
    'symbol': 'Symbol',
    'pnl': 'Total P&L',
    'total_trades': 'Trades',
    'pnl %': 'P&L Percentage',
    'weighted': 'Weight'
    })


    asset_type_df['total_pnl'] = asset_type_df['total_pnl'].apply(lambda x: f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}")
    asset_type_df['pnl %'] = asset_type_df['pnl %'].map('{:,.0f}%'.format)
    asset_type_df['weighted'] = (asset_type_df['weighted'] * 100).map('{:,.0f}%'.format)

    asset_type_df = asset_type_df.rename(columns={
    'subcategory': 'Type',
    'total_pnl': 'Total P&L',
    'total_trades': 'Trades',
    'pnl %': 'P&L Percentage',
    'weighted': 'Weight'
    })
    
    return assets_df, asset_type_df

def display_metrics_sidebar(df, metrics):
    profit_factor = calculate_profit_factor(df)
    tot_pnl="${:,.0f}".format(calculate_pnl_total(df))
    win_rate = "{:,.0f}%".format(calculate_win_rate(metrics).iloc[0])
    
    tot_trades = metrics['total_trades'].iloc[0]  
    st.sidebar.metric("PnL", tot_pnl)
    st.sidebar.metric("Total Trades", int(tot_trades))
    st.sidebar.metric("Win Rate", win_rate)
    st.sidebar.metric("Profit Factor", profit_factor)

def trade_analysis(df):
    # Apply clustering and analysis here
    numeric_cols = ['net_pnl', 'holding_period', 'weekday']
    weekday_encoder = LabelEncoder()
    df['weekday'] = weekday_encoder.fit_transform(df['weekday'])
    df = pd.get_dummies(df, columns=['subcategory'])
    df = apply_kmeans(df, 3)
    
    return df

def display_metrics_dashboard(df, metrics):
    a, b, c, d, e = st.columns(5)
    f,g,h,i,j = st.columns(5)
    win_rate="{:,.0f}%".format(calculate_win_rate(metrics).iloc[0])
    profit_factor=calculate_profit_factor(df)
    avg_win="${:,.0f}".format(calculate_avg_win(df))
    trade_durations = calculate_trade_duration(df)
    avg_loss="-${:,.0f}".format(-1*calculate_avg_loss(df))
    trade_durations = calculate_trade_duration(df)
    avg_win_hold= f"{trade_durations['Winning Trades']:,.0f}m"
    avg_loss_hold=f"{trade_durations['Losing Trades']:,.0f}m"
    top_win="${:,.0f}".format(calculate_top_win(df))
    top_loss="${:,.0f}".format(calculate_top_loss(df))
    trade_consecutive=calculate_consecutive_performance(df)
    streak_win=trade_consecutive['Max Wins']
    streak_loss=trade_consecutive['Max Losses']

    a.metric("Win Rate", win_rate, border=True)
    b.metric("Avg win", avg_win, border=True)
    c.metric("Avg win hold", avg_win_hold, border=True)
    d.metric("Top win", top_win, border=True)
    e.metric("Win streak", streak_win, border=True)
    
    f.metric("Profit factor", profit_factor, border=True)
    g.metric("Avg loss", avg_loss, border=True)
    h.metric("Avg loss hold", avg_loss_hold, border=True) 
    i.metric("Top loss", top_loss, border=True)
    j.metric("Loss streak", streak_loss, border=True)