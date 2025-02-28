import streamlit as st
import pandas as pd
import numpy as np
import random
import plotly.express as px
import altair as alt
import calplot 
import plotly.express as px
from datetime import datetime, timedelta
from streamlit_calendar import calendar
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from st_aggrid import AgGrid,GridUpdateMode, GridOptionsBuilder,DataReturnMode,JsCode
import sys
import os
import json
# Add the src folder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from get_metrics import *
from get_ticker_data import *


def load_data(db_utils):
    metrics = db_utils.fetch_data("SELECT * FROM trade_metrics ORDER BY date DESC LIMIT 1")
    # print(metrics)
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
    # profit_factor = round(float(metrics['profit_factor'].iloc[-1])*100)
    profit_factor = "{:,.0f}%".format(metrics['profit_factor'].iloc[-1]*100)
    tot_pnl="${:,.0f}".format(calculate_pnl_total(df))
    win_rate = "{:,.0f}%".format(calculate_win_rate(metrics).iloc[-1])
    
    tot_trades = metrics['total_trades'].iloc[-1]
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
    win_rate="{:,.0f}%".format(calculate_win_rate(metrics).iloc[-1])
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

def display_transactions(df_filtered):
    df_filtered['execution_time_sell'] = pd.to_datetime(df_filtered['execution_time_sell'], errors='coerce')
    df_filtered['TradeDate'] = df_filtered['execution_time_sell'].dt.date
    df_filtered['Result'] = np.where(df_filtered['net_pnl'] > 0, 'Win', 'Lose')
    df_filtered['net_pnl_percentage'] = ((df_filtered['price_sell']-df_filtered['price_buy']) / df_filtered['price_buy'])
    df_filtered['net_pnl_percentage_formatted'] = df_filtered['net_pnl_percentage'].apply(lambda x: f"{x:.0%}" if pd.notna(x) else "") #Handle nan values

    # Add dummy stars and views history (replace with your actual data if available)
    df_filtered['stars'] = [random.randint(0, 1000) for _ in range(len(df_filtered))]
    df_filtered['views_history'] = [[random.randint(0, 5000) for _ in range(30)] for _ in range(len(df_filtered))]

    # 1. Define column display names, order, and formatting
    column_definitions = {
        "TradeDate": {"display_name": "Date", "format": st.column_config.DateColumn},
        "symbol": {"display_name": "Symbol", "format": st.column_config.TextColumn},
        "OPT": {"display_name": "OPT", "format": st.column_config.TextColumn},
        "shares": {"display_name": "Shares", "format": lambda x: st.column_config.NumberColumn(x, format="%d")},
        "price_buy": {"display_name": "Buy Price", "format": lambda x: st.column_config.NumberColumn(x, format="$%.2f")},
        "price_sell": {"display_name": "Sell Price", "format": lambda x: st.column_config.NumberColumn(x, format="$%.2f")},
        "net_pnl": {"display_name": "Net P&L", "format": lambda x: st.column_config.NumberColumn(x, format="$%.2f")},
        "net_pnl_percentage_formatted": {"display_name": "Net P&L", "format": st.column_config.TextColumn},
        "holding_period": {"display_name": "Hold", "format": lambda x: st.column_config.NumberColumn(x, format="%d")},
        "views_history": {"display_name": "Views (past 30 days)", "format": lambda x: st.column_config.LineChartColumn(x, y_min=0, y_max=5000)},
        "stars": {"display_name": "Stars", "format": lambda x: st.column_config.NumberColumn(x, format="%d ⭐")},
    }

    # 2. Create default columns list in the desired order
    default_columns = [col for col in column_definitions if col in df_filtered.columns]


    views_history_data = df_filtered['views_history'].copy()

    # 3. Create df_to_display with ONLY default columns
    df_to_display = df_filtered[default_columns].copy()  # Use default_columns directly

    if 'views_history' in default_columns and 'views_history' in df_filtered.columns:  # Check if both exist
        df_to_display['views_history'] = views_history_data

    # 4. Create column_config using the ordered column_definitions
    column_config = {}
    for col in default_columns:  # Iterate over default_columns
        if col in column_definitions:
            definition = column_definitions[col]
            column_config[col] = definition["format"](definition["display_name"])

    return st.dataframe(df_to_display, column_config=column_config, hide_index=True, height=1000)

def display_calendar_metrics(df):
    pnl_data = calculate_pnl_by_period(df)

    # Safely access net_pnl values
    total_pnl_year = pnl_data["last_pnl_by_year"].get("net_pnl", 0)
    total_pnl_month = pnl_data["last_pnl_by_month"].get("net_pnl", 0)
    total_pnl_week = pnl_data["last_pnl_by_week"].get("net_pnl", 0)
    total_pnl_day = pnl_data["last_pnl_by_day"].get("net_pnl", 0)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("This Year", f"${total_pnl_year:,.0f}")
    col2.metric("This Month", f"${total_pnl_month:,.0f}")
    col3.metric("This Week", f"${total_pnl_week:,.0f}")
    col4.metric("Today", f"${total_pnl_day:,.0f}")

def display_calendar(df):

    df['execution_time_sell'] = pd.to_datetime(df['execution_time_sell'], errors='coerce')
    df.set_index('execution_time_sell', inplace=True)
    df = df.sort_index()

    def create_calendar_heatmap(data):
        if data.empty:
            st.warning("No data available for the selected month and year.")
            return None
        
        data['net_pnl'] = pd.to_numeric(data['net_pnl'], errors='coerce')
        fig = calplot.calplot(data['net_pnl'], cmap='RdYlGn', colorbar=True)
        return fig[0]  # Extract figure from tuple

    fig = create_calendar_heatmap(df)
    if fig:
        st.pyplot(fig)


    # CALENDAR

    df_grouped = df.resample('D').agg({'net_pnl': 'sum', 'symbol': 'count'})
    df_grouped.rename(columns={'symbol': 'total_trades'}, inplace=True)

    # Split into positive and negative PnL
    df_grouped['positive_pnl'] = df_grouped['net_pnl'].apply(lambda x: x if x > 0 else 0)
    df_grouped['negative_pnl'] = df_grouped['net_pnl'].apply(lambda x: x if x < 0 else 0)

    # Streamlit UI
    st.header("Filter Options")
    selected_year = st.selectbox("Select Year", sorted(df_grouped.index.year.unique()), index=0)
    selected_month = st.selectbox("Select Month", range(1, 13), index=datetime.now().month - 1, format_func=lambda x: datetime(2000, x, 1).strftime('%B'))

    # Filter by selected month and year
    filtered_df = df_grouped[(df_grouped.index.year == selected_year) & (df_grouped.index.month == selected_month)]

    # Convert to event format (separate positive & negative PnL)
    events = []

    for row in filtered_df.itertuples():
        # Positive PnL event
        if row.positive_pnl > 0:
            events.append({
                "title": f"${round(row.positive_pnl)} [{row.total_trades}]",
                "start": row.Index.strftime("%Y-%m-%d"),
                "color": "green",
            })

        # Negative PnL event
        if row.negative_pnl < 0:
            events.append({
                "title": f"-${round(row.negative_pnl*-1)} [{row.total_trades}]",
                "start": row.Index.strftime("%Y-%m-%d"),
                "color": "red",
            })

    # Display interactive calendar
    calendar(events=events, options={"initialView": "dayGridMonth"}, key="pnl_calendar")

    # Show raw data
    if st.checkbox("Show Raw Data"):
        st.dataframe(filtered_df.reset_index())


def display_trade_clusters(df):
    df['execution_time_sell'] = pd.to_datetime(df['execution_time_sell'])
    df = df.sort_values(by="execution_time_sell")
    df['weekday'] = df['execution_time_sell'].dt.day_name()
    numeric_cols = ['net_pnl', 'holding_period', 'weekday']  # Removed 'shares', replaced with 'subcategory'

    # Convert 'weekday' into numeric format using Label Encoding
    weekday_encoder = LabelEncoder()
    df['weekday'] = weekday_encoder.fit_transform(df['weekday'])

    # Convert 'subcategory' into numeric using One-Hot Encoding
    df = pd.get_dummies(df, columns=['subcategory'])

    # Update feature list to include newly created one-hot-encoded subcategory columns
    numeric_cols.extend([col for col in df.columns if col.startswith("subcategory_")])

    # Perform clustering
    def apply_kmeans(df, n_clusters=3):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[numeric_cols])  # Scale only numeric data
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df['Cluster'] = kmeans.fit_predict(scaled_data)
        return df, kmeans

    # Streamlit UI
    st.header('Settings')
    n_clusters = st.slider('Number of Clusters', min_value=2, max_value=5, value=3)

    df, model = apply_kmeans(df, n_clusters)

    # Compute cluster characteristics (including total trades per cluster)
    cluster_summary = df.groupby('Cluster')[numeric_cols].mean()
    cluster_summary['Total Trades'] = df['Cluster'].value_counts()

    col1, col2 = st.columns(2)  # Create two columns
    with col1:
        st.write("## Cluster Characteristics")
        st.write(cluster_summary)

        # Assign meaning based on cluster properties
        cluster_labels = {}
        for cluster in df['Cluster'].unique():
            mean_pnl = cluster_summary.loc[cluster, 'net_pnl']
            mean_duration = cluster_summary.loc[cluster, 'holding_period']

            if mean_pnl > 20:
                label = "High-Gain Trades"
            elif mean_pnl < -20:
                label = "High-Loss Trades"
            elif mean_duration > 60:
                label = "Long-Term Trades"
            else:
                label = "Stable Trades"
            
            cluster_labels[cluster] = label

        df['Cluster Label'] = df['Cluster'].map(cluster_labels)

    with col2:

        st.write("## Cluster Interpretation")
        st.dataframe(df[['Cluster', 'Cluster Label']].drop_duplicates())

    # Plot clustering results using Plotly with more contrasting colors
    fig = px.scatter(
        df,
        x='net_pnl',
        y='holding_period',
        color=df['Cluster'].astype(str),  # Convert to string for categorical coloring
        hover_data=['Cluster Label'],
        title="Trade Clustering",
        labels={'net_pnl': 'Net PnL', 'holding_period': 'Holding Period', 'color': 'Cluster'},
        color_discrete_sequence=px.colors.qualitative.Set1  # More contrasting colors
    )

    st.plotly_chart(fig, use_container_width=True)

def display_aggrid_pivot_old(df_filtered):


    df_filtered['execution_time_sell'] = pd.to_datetime(df_filtered['execution_time_sell'], errors='coerce')
    df_filtered['TradeDate'] = df_filtered['execution_time_sell'].dt.date
    df_filtered['Result'] = np.where(df_filtered['net_pnl'] > 0, 'Win', 'Lose')
    df_filtered['net_pnl_percentage'] = ((df_filtered['price_sell'] - df_filtered['price_buy']) / df_filtered['price_buy'])
    df_filtered['TradeDate'] = pd.to_datetime(df_filtered['TradeDate'], errors='coerce').dt.date
    df_filtered['stars'] = [random.randint(0, 1000) for _ in range(len(df_filtered))]
    df_filtered['views_history'] = [[random.randint(0, 5000) for _ in range(30)] for _ in range(len(df_filtered))]
    
    df_filtered['holding_period_str'] = df_filtered['holding_period'].apply(lambda x: "{:.1f}".format(x) if pd.notna(x) else "")

    df_range = get_ticker_range(df_filtered)

    df_filtered['atr'] = df_filtered.merge(df_range[['atr']], left_index=True, right_index=True, how='left')['atr']
    df_filtered['atr'] = df_filtered['atr'].apply(lambda x: round(x, 2))  # Round before displaying
    df_filtered['capture'] = df_filtered.apply(
        lambda row: row['net_pnl'] / row['atr'] if row['atr'] != 0 else 0, axis=1
    )
    df_filtered = df_filtered.sort_values(by="TradeDate", ascending=False)

    

    shouldDisplayPivoted = st.checkbox("Pivot data on Trade Date", value=True)
    # shouldDisplayPivoted = False

    gb = GridOptionsBuilder()

    gb.configure_default_column(
        resizable=True,
        filterable=True,
        sortable=True,
        editable=False,
    )

    if shouldDisplayPivoted:
        gb.configure_column(
            field="TradeDate",
            header_name="Trading Date",
            # width=100,
            rowGroup=True,
            valueFormatter="value != undefined ? new Date(value).toLocaleDateString('en-US', {month: 'short', day: 'numeric', year: 'numeric'}) : ''"
        )

        gb.configure_column(
            field="symbol",
            header_name="Asset Symbol",
            flex=1,
            tooltipField="symbol",
            rowGroup=True,
        )

        gb.configure_column(
            field="shares",
            header_name="Shares",
            width=100,
            type=["numericColumn"],
            aggFunc="sum",
            valueFormatter="value.toLocaleString('en-US')",
        )

        gb.configure_column(
            field="net_pnl",
            header_name="Total PnL",
            width=100,
            type=["numericColumn"],
            aggFunc="sum",
            valueFormatter="'$' + value.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})",
        )

        gb.configure_column(
            field="atr",
            header_name="ATR",
            width=100,
            aggFunc="max",
            type=["numericColumn"],
            valueFormatter="'$' +value.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})",
        )

        gb.configure_column(
            field="capture",
            header_name="Capture",
            width=100,
            aggFunc="max",
            type=["numericColumn"],
            valueFormatter="value.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})+'%'",
        )

        gb.configure_column(
            "views_history",
            header_name="Views (30 days)",
            width=200,
            cellRenderer="agSparklineCellRenderer",
            aggFunc="last",
            cellRendererParams={
                "sparklineOptions": {
                    "type": "line",
                    "line": {"stroke": "#3498db", "strokeWidth": 2},
                }
            },
        )



        # gb.configure_column(
        #     field="holding_period",
        #     header_name="Hold",
        #     # width=180,
        #     type=["numericColumn"],
        #     aggFunc="avg",
        #     valueFormatter="value.toLocaleString('en-US', {minimumFractionDigits: 1, maximumFractionDigits: })",
        # )


        # gb.configure_column(
        #     field="stars",
        #     header_name="Average Stars",
        #     width=120,
        #     type=["numericColumn"],
        #     aggFunc="avg",
        #     valueFormatter="value.toLocaleString('en-US', {minimumFractionDigits: 0})",
        # )

    else:
        gb.configure_column(
            field="TradeDate",
            header_name="Trade Date",
            width=120,
            valueFormatter="value != undefined ? new Date(value).toLocaleDateString('en-US', {dateStyle:'medium'}): ''"
        )

        gb.configure_column(
            field="symbol",
            header_name="Symbol",
            flex=1,
            tooltipField="symbol",
        )

        gb.configure_column(
            field="shares",
            header_name="Shares",
            width=80,
            type=["numericColumn"],
            valueFormatter="value.toLocaleString()",
        )

        gb.configure_column(
            field="net_pnl",
            header_name="Net P&L",
            width=100,
            type=["numericColumn"],
            valueFormatter="value.toLocaleString()",
        )

        gb.configure_column(
            field="price_buy",
            header_name="Buy Price",
            width=120,
            type=["numericColumn"],
            valueFormatter="value.toLocaleString()",
        )

        gb.configure_column(
            field="price_sell",
            header_name="Sell Price",
            width=120,
            type=["numericColumn"],
            valueFormatter="value.toLocaleString()",
        )

        gb.configure_column(
            field="holding_period",
            header_name="Holding Period",
            width=120,
            type=["numericColumn"],
            valueFormatter="value.toLocaleString()",
        )

        gb.configure_column(
            field="stars",
            header_name="Stars",
            width=100,
            type=["numericColumn"],
            valueFormatter="value.toLocaleString()",
        )

        gb.configure_column(
            "views_history",
            header_name="Views (30 days)",
            width=200,
            cellRenderer="agSparklineCellRenderer",
            cellRendererParams={
                "sparklineOptions": {
                    "type": "line",
                    "line": {"stroke": "#3498db", "strokeWidth": 2},
                }
            },
        )


    gb.configure_grid_options(
        tooltipShowDelay=0,
        pivotMode=shouldDisplayPivoted,
        suppressAggFuncInHeader=True, # Add this line
        autoSizeColumns=True,  # Add this line
    )

    gb.configure_grid_options(
        autoGroupColumnDef=dict(
            minWidth=150,
            pinned="left",
            cellRendererParams=dict(suppressCount=True),
        )
    )
    go = gb.build()    

    AgGrid(df_filtered, gridOptions=go, height=400)


def display_aggrid_pivot(df_filtered):
    # Load Bootstrap CSS for styling
    st.markdown('<link href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">', unsafe_allow_html=True)

    # Preprocess DataFrame
    df_filtered['execution_time_sell'] = pd.to_datetime(df_filtered['execution_time_sell'], errors='coerce')
    df_filtered['TradeDate'] = df_filtered['execution_time_sell'].dt.date
    df_filtered['Result'] = np.where(df_filtered['net_pnl'] > 0, 'Win', 'Lose')
    df_filtered['net_pnl_percentage'] = ((df_filtered['price_sell'] - df_filtered['price_buy']) / df_filtered['price_buy'])
    df_filtered['stars'] = [random.randint(0, 1000) for _ in range(len(df_filtered))]
    df_filtered['views_history'] = [[random.randint(0, 5000) for _ in range(30)] for _ in range(len(df_filtered))]
    df_filtered['holding_period_str'] = df_filtered['holding_period'].apply(lambda x: "{:.1f}".format(x) if pd.notna(x) else "")

    # Merge ATR values and calculate capture ratio
    df_range = get_ticker_range(df_filtered)
    df_filtered['atr'] = df_filtered.merge(df_range[['atr']], left_index=True, right_index=True, how='left')['atr']
    df_filtered['atr'] = df_filtered['atr'].apply(lambda x: round(x, 2))
    df_filtered['capture'] = df_filtered.apply(lambda row: row['net_pnl'] / row['atr'] if row['atr'] != 0 else 0, axis=1)

    # Calculate cumulative net PnL per trading day and merge min/max values
    df_filtered['cumulative_net_pnl'] = df_filtered.groupby("TradeDate")["net_pnl"].cumsum()
    min_max_pnl = df_filtered.groupby("TradeDate")["cumulative_net_pnl"].agg(["min", "max"]).reset_index()
    min_max_pnl.rename(columns={"min": "min_cum_pnl", "max": "max_cum_pnl"}, inplace=True)
    df_filtered = df_filtered.merge(min_max_pnl, on="TradeDate", how="left")

    # Calculate percent max profit and round values
    df_filtered['percent_max_profit'] = df_filtered.apply(
        lambda row: (row['cumulative_net_pnl'] / (abs(row['min_cum_pnl']) + row['max_cum_pnl'])) * 100 if (abs(row['min_cum_pnl']) + row['max_cum_pnl']) != 0 else 0,
        axis=1
    )
    df_filtered['percent_max_profit'] = df_filtered['percent_max_profit']
    # Add a new 'net_pnl_range' column to your dataframe

    # Add the 'net_pnl_range' column to df_filtered for visualization
    df_filtered['net_pnl_range'] = df_filtered.apply(
        lambda row: row['net_pnl'] if pd.notna(row['net_pnl']) else 0, axis=1
    )
    
    df_filtered = df_filtered.sort_values(by="TradeDate", ascending=False)

    # Grid Options Builder
    gb = GridOptionsBuilder()
    gb.configure_default_column(
        resizable=True,
        filterable=True,
        sortable=True,
        editable=False,
    )

    # Configure columns
    gb.configure_column(
        field="TradeDate",
        header_name="Trading Date",
        rowGroup=True,
        valueFormatter="value != undefined ? new Date(value).toLocaleDateString('en-US', {month: 'short', day: 'numeric', year: 'numeric'}) : ''"
    )

    gb.configure_column(
        field="symbol",
        header_name="Asset Symbol",
        flex=1,
        tooltipField="symbol",
        rowGroup=True,
    )

    gb.configure_column(
        field="net_pnl",
        header_name="Total PnL",
        width=100,
        type=["numericColumn"],
        aggFunc="sum",
        valueFormatter="'$' + value.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})",
    )

    gb.configure_column(
        field="atr",
        header_name="ATR",
        width=100,
        aggFunc="max",
        type=["numericColumn"],
        valueFormatter="'$' +value.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})",
    )

    gb.configure_column(
        field="capture",
        header_name="Capture",
        width=100,
        aggFunc="max",
        type=["numericColumn"],
        valueFormatter="value.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})+'%'",
    )

    gb.configure_column(
        field="min_cum_pnl",
        header_name="Min Cum PnL",
        width=120,
        type=["numericColumn"],
        aggFunc="min",
        valueFormatter="'$' + value.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})",
    )

    gb.configure_column(
        field="max_cum_pnl",
        header_name="Max Cum PnL",
        width=120,
        type=["numericColumn"],
        aggFunc="max",
        valueFormatter="'$' + value.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})",
    )

    gb.configure_column(
        field="percent_max_profit",
        header_name="% Max Profit",
        width=120,
        type=["numericColumn"],
        aggFunc="last",
        valueFormatter="value.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})+'%'",
    )

    # Configure views history as sparkline (line chart)
    gb.configure_column(
        "views_history",
        header_name="Views (30 days)",
        width=200,
        cellRenderer="agSparklineCellRenderer",
        aggFunc="last",
        cellRendererParams={
            "sparklineOptions": {
                "type": "line",
                "line": {"stroke": "#3498db", "strokeWidth": 2},
            }
        },
    )

    # Additional grid options
    gb.configure_grid_options(
        pivotMode=True,
        suppressAggFuncInHeader=True,
        autoSizeColumns=True,
    )

    gb.configure_grid_options(
        autoGroupColumnDef=dict(
            minWidth=150,
            pinned="left"
        )
    )

    # Build and display AgGrid
    go = gb.build()
    AgGrid(df_filtered, gridOptions=go, height=400)
