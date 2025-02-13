import sqlite3
import pandas as pd
import os
import uuid
import sys
import argparse

# Connect to SQLite
# Argument parsing to select database
parser = argparse.ArgumentParser(description="Process transactions for a specific account.")
parser.add_argument("account_type", choices=["real", "paper"], help="Specify which account to process: real or paper")
args = parser.parse_args()

# Select database based on the argument
DB_NAME = "real_all_transactions.db" if args.account_type == "real" else "paper_all_transactions.db"

conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()

# Ensure trades table exists
cursor.execute('''
    CREATE TABLE IF NOT EXISTS merged_trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conid INTEGER,
        symbol TEXT,
        order_id TEXT,
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
        win_loss TEXT,
        subcategory TEXT
    )
''')

# Load new trade data that hasn't been processed
# query = "SELECT * FROM transactions ORDER BY OrderTime"
query = "SELECT * FROM transactions WHERE processed = 0 ORDER BY OrderTime"
df = pd.read_sql(query, conn)

if df.empty:
    print("⚠️ No new trades found.")
    conn.close()
    exit()

# Ensure proper data types
df['TradePrice'] = pd.to_numeric(df['TradePrice'])
df['Quantity'] = pd.to_numeric(df['Quantity'])
df['OrderTime'] = pd.to_datetime(df['OrderTime'])
df['IBCommission'] = pd.to_numeric(df['IBCommission'], errors='coerce').fillna(0)

# Fix sell orders (convert negative shares to positive)
df.loc[df['BuySell'] == 'SELL', 'Quantity'] = df.loc[df['BuySell'] == 'SELL', 'Quantity'].abs()

# Split buy and sell orders
buys = df[df['BuySell'] == 'BUY'].copy()
sells = df[df['BuySell'] == 'SELL'].copy()

# Sort transactions by execution time (FIFO)
buys = buys.sort_values(by='OrderTime')
sells = sells.sort_values(by='OrderTime')

# List to store merged trades
merged_trades = []

# Dictionary to track remaining shares per conid
remaining_buys = {}

# Store the TransactionIDs that are part of this merged trade
transaction_ids = []
# Track transaction updates
transaction_updates = []

# Process buys
for _, buy in buys.iterrows():
    conid = buy['Conid']  # Ensure column names are correct
    
    if conid not in remaining_buys:
        remaining_buys[conid] = []

    buy_order_id = str(uuid.uuid4())  # Use UUIDs for unique IDs
    remaining_buys[conid].append({
        'order_id': buy_order_id,
        'TradePrice': buy['TradePrice'],
        'Quantity': buy['Quantity'],
        'OrderTime': buy['OrderTime'],
        'IBCommission': buy['IBCommission'],
        'TransactionID': buy['TransactionID'] 
    })
    # Update transaction with order_id and mark as processed
    transaction_updates.append((buy_order_id, 1, buy['TransactionID']))

# Process sells and match with remaining buys
for _, sell in sells.iterrows():
    conid = sell['Conid']  # Ensure column names are correct

    if conid not in remaining_buys or not remaining_buys[conid]:
        continue  # No matching buy found

    sell_shares = sell['Quantity']
    sell_price = sell['TradePrice']
    sell_execution_time = sell['OrderTime']
    sell_commission = sell['IBCommission']
    sell_transaction_id = sell['TransactionID']

    while sell_shares > 0 and remaining_buys[conid]:
        buy_order = remaining_buys[conid][0]  # FIFO: Take the oldest buy
        buy_shares = buy_order['Quantity']
        buy_order_id = buy_order['order_id']  # Retrieve the order ID
        
        matched_shares = min(sell_shares, buy_shares)

        # Compute trade statistics
        realized_pnl = (sell_price - buy_order['TradePrice']) * matched_shares * 100
        net_pnl = realized_pnl + (buy_order['IBCommission'] + sell_commission)
        holding_period = (sell_execution_time - buy_order['OrderTime']).total_seconds() / 60  # Minutes
        win_loss = 'Win' if net_pnl > 0 else 'Loss'

        # Store merged trade
        merged_trades.append({
            'conid': conid,
            'symbol': sell['UnderlyingSymbol'], 
            'order_id': buy_order_id,  
            'price_buy': buy_order['TradePrice'],
            'price_sell': sell_price,
            'shares': matched_shares,
            'execution_time_buy': buy_order['OrderTime'],
            'execution_time_sell': sell_execution_time,
            'realized_pnl': realized_pnl,
            'commission_buy': buy_order['IBCommission'],
            'commission_sell': sell_commission,
            'net_pnl': net_pnl,
            'holding_period': holding_period,
            'win_loss': win_loss,
            'subcategory': sell['SubCategory']
        })

        # Update remaining shares
        buy_order['Quantity'] -= matched_shares
        sell_shares -= matched_shares

        # Remove fully matched buy orders
        if buy_order['Quantity'] == 0:
            remaining_buys[conid].pop(0)

    # Update transaction with processed flag
    transaction_updates.append((buy_order_id, 1, sell_transaction_id))

# Convert to DataFrame and save to database
merged_trades_df = pd.DataFrame(merged_trades)

if not merged_trades_df.empty:
    merged_trades_df.to_sql('merged_trades', conn, if_exists='append', index=False)

# Update transactions table with processed order_ids
cursor.executemany("UPDATE transactions SET processed = ? WHERE TransactionID = ?", 
                   [(order_id, transaction_id) for order_id, _, transaction_id in transaction_updates])

print(f'✅ {len(merged_trades)} trades successfully merged and stored.')
conn.commit()
conn.close()
