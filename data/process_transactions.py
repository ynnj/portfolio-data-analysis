import sqlite3
import pandas as pd
import os

# Connect to SQLite
DB_PATH = os.path.join(os.path.dirname(__file__), "../data/trades.db")
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Ensure merged_trades table exists
cursor.execute('''
    CREATE TABLE IF NOT EXISTS merged_trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conid INTEGER,
        symbol TEXT,
        price_buy REAL,
        price_sell REAL,
        shares INTEGER,
        execution_time_buy TEXT,
        execution_time_sell TEXT,
        realized_pnl REAL,
        commission_buy REAL,
        commission_sell REAL,
        net_pnl REAL,
        holding_period REAL,
        win_loss TEXT
    )
''')

# Load new trade data that hasn't been processed
query = "SELECT * FROM trades WHERE processed = 0 ORDER BY execution_time"
df = pd.read_sql(query, conn)

if df.empty:
    print("⚠️ No new trades found.")
    conn.close()
    exit()

# Ensure proper data types
df['price'] = pd.to_numeric(df['price'])
df['shares'] = pd.to_numeric(df['shares'])
df['execution_time'] = pd.to_datetime(df['execution_time'])
df['commission'] = pd.to_numeric(df['commission'], errors='coerce').fillna(0)

# Fix sell orders (convert negative shares to positive)
df.loc[df['side'] == 'SELL', 'shares'] = df.loc[df['side'] == 'SELL', 'shares'].abs()

# Split buy and sell orders
buys = df[df['side'] == 'BUY'].copy()
sells = df[df['side'] == 'SELL'].copy()

# Sort transactions by execution time (FIFO)
buys = buys.sort_values(by='execution_time')
sells = sells.sort_values(by='execution_time')

# List to store merged trades
merged_trades = []

# Dictionary to track remaining shares per conid
remaining_buys = {}

# Process buys
for _, buy in buys.iterrows():
    conid = buy['conid']
    
    if conid not in remaining_buys:
        remaining_buys[conid] = []
    
    remaining_buys[conid].append({
        'price': buy['price'],
        'shares': buy['shares'],
        'execution_time': buy['execution_time'],
        'commission': buy['commission']
    })

# Process sells and match with remaining buys
for _, sell in sells.iterrows():
    conid = sell['conid']

    if conid not in remaining_buys or not remaining_buys[conid]:
        continue  # No matching buy found

    sell_shares = sell['shares']
    sell_price = sell['price']
    sell_execution_time = sell['execution_time']
    sell_commission = sell['commission']

    while sell_shares > 0 and remaining_buys[conid]:
        buy_order = remaining_buys[conid][0]  # FIFO: Take the oldest buy
        buy_shares = buy_order['shares']
        
        matched_shares = min(sell_shares, buy_shares)

        # Compute trade statistics
        realized_pnl = (sell_price - buy_order['price']) * matched_shares * 100
        net_pnl = realized_pnl - (buy_order['commission'] + sell_commission)
        holding_period = (sell_execution_time - buy_order['execution_time']).total_seconds() / 60  # Minutes
        win_loss = 'Win' if net_pnl > 0 else 'Loss'

        # Store merged trade
        merged_trades.append({
            'conid': conid,
            'symbol': sell['symbol'],
            'price_buy': buy_order['price'],
            'price_sell': sell_price,
            'shares': matched_shares,
            'execution_time_buy': buy_order['execution_time'],
            'execution_time_sell': sell_execution_time,
            'realized_pnl': realized_pnl,
            'commission_buy': buy_order['commission'],
            'commission_sell': sell_commission,
            'net_pnl': net_pnl,
            'holding_period': holding_period,
            'win_loss': win_loss
        })

        # Update remaining shares
        buy_order['shares'] -= matched_shares
        sell_shares -= matched_shares

        # Remove fully matched buy orders
        if buy_order['shares'] == 0:
            remaining_buys[conid].pop(0)

# Convert to DataFrame and save to database
merged_trades_df = pd.DataFrame(merged_trades)

if not merged_trades_df.empty:
    merged_trades_df.to_sql('merged_trades', conn, if_exists='append', index=False)

# Mark processed trades in the trades table
trade_ids = df['trade_id'].tolist()
if trade_ids:
    cursor.execute(f"UPDATE trades SET processed = 1 WHERE trade_id IN ({','.join(['?']*len(trade_ids))})", trade_ids)

print(f'✅ {len(merged_trades)} trades successfully merged and stored.')
conn.commit()
conn.close()
