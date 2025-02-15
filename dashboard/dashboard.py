import streamlit as st
import pandas as pd
import sqlite3
import numpy as np
import os
import plotly.express as px
from streamlit_elements import elements, mui, html, dashboard
import random
import calplot  # Install: pip install calplot
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from streamlit_calendar import calendar
from wordcloud import WordCloud
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set Streamlit page config
st.set_page_config(page_title="Trade Performance Dashboard", layout="wide",initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 70px !important;  /* Adjust this value as needed */
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# Connect to SQLite
# DB_PATH = os.path.join(os.path.dirname(__file__), "real_all_transactions.db")
conn = sqlite3.connect("real_all_transactions.db")

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

# Compute cumulative P&L
df['cumulative_pnl'] = df['net_pnl'].cumsum()
latest_metrics = metrics.iloc[0]



# Sidebar filters
# st.sidebar.header("🔍 Filters")
# selected_symbols = st.sidebar.multiselect("Select Asset Symbols", df['symbol'].unique(), default=df['symbol'].unique())
# date_range = st.sidebar.date_input("Select Date Range", [df['trade_date'].min(), df['trade_date'].max()])

# # Apply filters
# df_filtered = df[(df['symbol'].isin(selected_symbols)) & (df['trade_date'].between(date_range[0], date_range[1]))]
df_filtered=df.copy()

# Sidebar Key Metrics
st.sidebar.header('📊 Key Metrics')
st.sidebar.metric(label='Total P/L', value=f"€{df['cumulative_pnl'].iloc[-1]:.2f}", delta="1.2 °F")
st.sidebar.metric("Total Trades", int(latest_metrics['total_trades']))
st.sidebar.metric("Win Rate (%)", f"{latest_metrics['win_rate']:.2f}")

view = st.sidebar.radio(
    "Select a view:",
    ("Dashboard", "Trades", "Calendar", "Deep Learning Analysis", "Market Sentiment")  # Add more views here
)


# winrate, expectancy, profit factor, avg win hold, avg loss hold. avg loss, avg win, win streak, loss streak, top loss, top win, avg daily vol, avg size

if view == "Dashboard":
    st.title("📊 Trade Performance Dashboard")

    a, b, c, d, e, f = st.columns(6)
    g,h,i,j,k,l = st.columns(6)

    a.metric("Total wins", "21", "-9°F", border=True)
    b.metric("Average win", "$374", "2 mph", border=True)
    c.metric("Profit factor", "77%", "5%", border=True)
    d.metric("Total losses", "6", "-2 inHg", border=True)
    e.metric("Average loss", "$197", "2 mph", border=True)
    f.metric("Average hold", "3 min", "5%", border=True)

    g.metric("Total wins", "21", "-9°F", border=True)
    h.metric("Average win", "$374", "2 mph", border=True)
    i.metric("Profit factor", "77%", "5%", border=True)
    j.metric("Total losses", "6", "-2 inHg", border=True)
    k.metric("Average loss", "$197", "2 mph", border=True)
    l.metric("Average hold", "3 min", "5%", border=True)


    fig_cum_pnl = px.area(
        df_filtered, x="execution_time_sell", y="cumulative_pnl"
    )

    # Customize the chart (optional, but recommended for cleaner look)
    fig_cum_pnl.update_xaxes(showgrid=False, title_text="", showticklabels=False)
    fig_cum_pnl.update_yaxes(showgrid=False, title_text="")

    # Optionally, fill the area with a specific color and opacity
    fig_cum_pnl.update_traces(fillcolor="skyblue", opacity=0.7)  # Example color

    # Set the background color to white for better contrast (optional)
    fig_cum_pnl.update_layout(plot_bgcolor='white')

    st.plotly_chart(fig_cum_pnl, use_container_width=True)


        # Sample data
    data = {
        'weekday': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
        'average_daily_pnl': [150, -50, 200, 75, -100]
    }
    df = pd.DataFrame(data)

    # Sorting weekdays in correct order
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    df['weekday'] = pd.Categorical(df['weekday'], categories=weekday_order, ordered=True)
    df = df.sort_values('weekday')

    # Sample data for PnL per hour
    hours = np.arange(0, 24)
    pnl_per_hour = np.random.randint(-50, 100, size=24)

    # Sample data for performance by asset name
    assets_data = {
    'symbol': ['AAPL', 'TSLA', 'GOOGL', 'AMZN', 'MSFT'],
    'total_trades': [10, 15, 12, 8, 20],
    'pnl': [500, -200, 300, -100, 600],
    'pnl_percent': [5.0, -2.5, 3.0, -1.0, 6.0],
    'weighted': [250, -150, 180, -50, 400]
    }
    assets_df = pd.DataFrame(assets_data)

    # Sample data for performance by asset type
    asset_type_data = {
        'symbol': ['Tech', 'Auto', 'Tech', 'E-commerce', 'Tech'],
        'total_trades': [30, 15, 25, 10, 35],
        'pnl': [1400, -300, 900, -250, 1600],
        'pnl_percent': [4.5, -1.5, 3.2, -2.0, 5.8],
        'weighted': [700, -200, 500, -150, 900]
    }
    asset_type_df = pd.DataFrame(asset_type_data)

    # Streamlit App
    st.title("PnL Analysis")

    # Average PnL per Weekday Bar Chart
    st.subheader("Average PnL per Weekday")
    fig, ax = plt.subplots()
    ax.barh(df['weekday'], df['average_daily_pnl'], color=['green' if x > 0 else 'red' for x in df['average_daily_pnl']])
    ax.set_ylabel('Weekday')
    ax.set_xlabel('Average P&L')
    ax.set_title('Average P&L per Weekday')
    ax.axvline(0, color='black', linewidth=0.8)
    st.pyplot(fig)

    # Average PnL per Hour Bar Chart
    st.subheader("Average PnL per Hour")
    fig, ax = plt.subplots()
    ax.barh(hours, pnl_per_hour, color=['green' if x > 0 else 'red' for x in pnl_per_hour])
    ax.set_ylabel('Hour of the Day')
    ax.set_xlabel('Average P&L')
    ax.set_title('Average P&L per Hour')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_yticks(hours)
    ax.set_yticklabels([str(h) for h in hours])
    st.pyplot(fig)

        # Performance by Asset Name Table
    st.subheader("Performance by Asset Name")
    st.dataframe(assets_df)

    # Performance by Asset Type Table
    st.subheader("Performance by Asset Type")
    st.dataframe(asset_type_df)





elif view == "Trades":
    df = pd.DataFrame(
        {
            "name": ["Roadmap", "Extras", "Issues"],
            "url": ["https://roadmap.streamlit.app", "https://extras.streamlit.app", "https://issues.streamlit.app"],
            "stars": [random.randint(0, 1000) for _ in range(3)],
            "views_history": [[random.randint(0, 5000) for _ in range(30)] for _ in range(3)],
        }
    )
    st.dataframe(
        df,
        column_config={
            "name": "App name",
            "stars": st.column_config.NumberColumn(
                "Github Stars",
                help="Number of stars on GitHub",
                format="%d ⭐",
            ),
            "url": st.column_config.LinkColumn("App URL"),
            "views_history": st.column_config.LineChartColumn(
                "Views (past 30 days)", y_min=0, y_max=5000
            ),
        },
        hide_index=True,
    )


    # 📋 **Trade Data Table**
    df_filtered['TradeDate'] = df_filtered['execution_time_sell'].dt.date
    df_filtered['Result'] = np.where(df_filtered['net_pnl'] > 0, 'Win', 'Lose')  # Handles 0 as Lose
    df_filtered['net_pnl_percentage'] = (df_filtered['net_pnl'] / df_filtered['price_buy']) * 100

    # Define styling function for the "Result" column
    def highlight_result(val):
        color = "green" if val == "Win" else "red"
        return f'background-color: {color}; color: white; font-weight: bold;'

    # Format and style DataFrame
    styled_df = (
        df_filtered[['TradeDate', 'symbol', 'Result', 'subcategory', 'shares', 'price_buy', 'price_sell', 'net_pnl', 'net_pnl_percentage', 'holding_period']]
        .rename(columns={'TradeDate': 'Date', 'subcategory': 'OPT', 'net_pnl': 'Net P&L ($)', 'holding_period': 'Holding (m)', 'net_pnl_percentage': 'Net P&L (%)'})
        .style
        .format({'price_buy': '{:.2f}', 'price_sell': '{:.2f}', 'Net P&L ($)': '{:.2f}', 'Net P&L (%)': '{:.0f}', 'Holding (m)': '{:.0f}'})
        .applymap(lambda x: 'background-color: red' if isinstance(x, (int, float)) and x < 0 else '', subset=['Net P&L ($)'])  # Apply PnL highlighting
        .applymap(highlight_result, subset=['Result'])  # Apply conditional formatting to Result column
        .set_table_styles([
            {'selector': 'th', 'props': [('background-color', '#4CAF50'), ('color', 'white'), ('font-size', '16px'), ('text-align', 'center')]},  # Green header
            {'selector': 'td', 'props': [('font-size', '14px')]}  # Larger text
        ])
        .set_table_attributes("style='display:inline'") 
    )

    # Display in Streamlit
    st.dataframe(styled_df,hide_index=True)


elif view == "Calendar":
    st.title("PnL Calendar Dashboard")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total PNL Year", "70 °F", "1.2 °F")
    col2.metric("Total PNL Month", "9 mph", "-8%")
    col3.metric("Humidity", "86%", "4%")


    def generate_sample_data(days=365):
        today = datetime.now()
        dates = [today - timedelta(days=i) for i in range(days)]
        pnl = np.random.uniform(-500, 500, days)  # Random PnL values
        return pd.DataFrame({'Date': dates, 'PnL': pnl})

    # Load sample data
    df = generate_sample_data()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Streamlit UI

    st.write("This dashboard displays the total PnL per day in a calendar format.")

    # Month navigation
    selected_month = st.slider("Select Month", 1, 12, datetime.now().month)
    selected_year = st.slider("Select Year", df.index.year.min(), df.index.year.max(), datetime.now().year)

    # Filter data by selected month and year
    filtered_df = df[(df.index.month == selected_month) & (df.index.year == selected_year)]

    def create_calendar_heatmap(data):
        if data.empty:
            st.warning("No data available for the selected month and year.")
            return None
        fig = calplot.calplot(data['PnL'], cmap='RdYlGn', colorbar=True)
        return fig[0]  # calplot returns a tuple (fig, axes), we only need fig

    # Display heatmap
    fig = create_calendar_heatmap(filtered_df)
    if fig:
        st.pyplot(fig)

    # Load sample data
    df = generate_sample_data()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Sidebar for navigation
    st.sidebar.header("Filter Options")
    selected_year = st.sidebar.selectbox("Select Year", sorted(df.index.year.unique()), index=0)
    selected_month = st.sidebar.selectbox("Select Month", range(1, 13), index=datetime.now().month - 1, format_func=lambda x: datetime(2000, x, 1).strftime('%B'))

    # Filter data by selected month and year
    filtered_df = df[(df.index.month == selected_month) & (df.index.year == selected_year)]

    # Convert data to events format for streamlit-calendar
    events = [
        {
            "title": f"PnL: {round(row.PnL, 2)}",
            "start": row.Index.strftime("%Y-%m-%d"),
            "backgroundColor": "#FF4B4B" if row.PnL < 0 else "#3DD56D",
            "borderColor": "#FF4B4B" if row.PnL < 0 else "#3DD56D",
        }
        for row in filtered_df.itertuples()
    ]

    # Display interactive calendar
    calendar(events=events, options={"initialView": "dayGridMonth"}, key="pnl_calendar")

    # Show raw data
    if st.checkbox("Show Raw Data"):
        st.dataframe(df.reset_index())


elif view == "Deep Learning Analysis":
    # Generate synthetic trade data
    def generate_trade_data(n=200):
        np.random.seed(42)
        profit_loss = np.random.normal(loc=0, scale=10, size=n)
        duration = np.random.normal(loc=50, scale=20, size=n)
        volatility = np.random.normal(loc=2, scale=1, size=n)
        df = pd.DataFrame({'Profit/Loss (%)': profit_loss, 'Duration (mins)': duration, 'Volatility': volatility})
        return df

    # Perform clustering
    def apply_kmeans(df, n_clusters=2):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df['Cluster'] = kmeans.fit_predict(scaled_data)
        return df, kmeans

    # Streamlit UI
    st.title('Trade Clustering with K-Means')
    st.sidebar.header('Settings')
    n_clusters = st.sidebar.slider('Number of Clusters', min_value=2, max_value=5, value=2)

    df = generate_trade_data()
    df, model = apply_kmeans(df, n_clusters)

    st.dataframe(df.head())

    # Compute cluster characteristics
    cluster_means = df.groupby('Cluster').mean()
    st.write(cluster_means)

    # Assign meaning based on cluster properties
    cluster_labels = {}
    for cluster in df['Cluster'].unique():
        mean_profit = cluster_means.loc[cluster, 'Profit/Loss (%)']
        mean_duration = cluster_means.loc[cluster, 'Duration (mins)']
        mean_volatility = cluster_means.loc[cluster, 'Volatility']
        
        if mean_profit > 5 and mean_duration > 40:
            label = "High-Performing Trades"
        elif mean_profit < -5 and mean_volatility > 2.5:
            label = "High-Risk Unsuccessful Trades"
        elif mean_profit < -5:
            label = "Unsuccessful Trades"
        elif mean_duration > 60:
            label = "Long-Term Moderate Trades"
        else:
            label = "Steady Performers"
        
        cluster_labels[cluster] = label

    df['Cluster Label'] = df['Cluster'].map(cluster_labels)

    st.write("## Cluster Interpretation")
    st.dataframe(df[['Cluster', 'Cluster Label']].drop_duplicates())

    # Plot clustering results
    fig, ax = plt.subplots()
    scatter = ax.scatter(df['Profit/Loss (%)'], df['Duration (mins)'], c=df['Cluster'], cmap='viridis', alpha=0.7)
    ax.set_xlabel('Profit/Loss (%)')
    ax.set_ylabel('Duration (mins)')
    ax.set_title('Trade Clustering')
    plt.colorbar(scatter, label='Cluster')
    st.pyplot(fig)




elif view == "Market Sentiment":
    st.title("Sentiment Analysis with NLP")
    a, b, c = st.columns(3)
    d, e, f = st.columns(3)

    a.metric("This Week", "Trending", "-9°F", border=True)
    b.metric("This Month", "Sideways", "2 mph", border=True)
    c.metric("This Year", "Sideways", "5%", border=True)


    def convert_to_cumulative_returns(values):
        returns = (values[1:] - values[:-1]) / values[:-1]
        cum_returns = np.cumsum(returns, axis=0)
        return np.vstack([[0]*values.shape[1], cum_returns]) * 100

    PORTFOLIOS = {
        'Tech Stocks': ['Apple', 'Microsoft', 'Google', 'Amazon', 'Facebook'],
        'Emerging Markets': ['Tencent', 'Alibaba', 'Samsung', 'Baidu', 'Xiaomi'],
        'Green Energy Assets': ['Tesla', 'NIO', 'Plug Power', 'First Solar', 'Enphase Energy']
    }

    DATE_RANGE = pd.date_range(start="2021-01-01", end="2021-12-31", freq='B')
    ASSET_VALUES = np.random.rand(len(DATE_RANGE), 5) * 1000

    portfolio_selected = "Tech Stocks"
    date_range = DATE_RANGE[-1]

    NEWS_ITEMS = ["Positive news about Apple", "Google faces regulatory challenges", "Amazon grows in Europe", "Facebook under scrutiny", "Microsoft announces new partnership"]
    SENTIMENTS = ['positive', 'neutral', 'negative']


    news_df = pd.DataFrame({
        'Headline': NEWS_ITEMS,
        'Sentiment': np.random.choice(SENTIMENTS, 5)
    })
    st.write(news_df)

    wordcloud_data = {
        'Apple': 50,
        'Google': 30,
        'Regulatory': 25,
        'Europe': 20,
        'Partnership': 15,
        'Growth': 10,
        'Batteries': 45,
        'Large Language Model': 40,
        'Dodge and Cox': 60,
        'Innovation': 35,
        'E-commerce': 28,
        'Blockchain': 33,
        'AI': 50,
        'Sustainability': 24,
        'Data Privacy': 32,
        'Fintech': 27,
        'Cloud Computing': 38,
        'Digital Transformation': 29,
        '5G': 31,
        'Machine Learning': 34
    }

    wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(wordcloud_data)
   
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)