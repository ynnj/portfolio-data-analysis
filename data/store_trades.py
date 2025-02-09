from ib_insync import *
import sqlite3

# Connect to IBKR TWS or Gateway
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)  # Paper Trading port

# Retrieve trade executions
executions = ib.reqExecutions()

# Connect to SQLite database
conn = sqlite3.connect("trades.db")
cursor = conn.cursor()

# Create table (if not exists)
cursor.execute('''
    CREATE TABLE IF NOT EXISTS trades (
        trade_id TEXT PRIMARY KEY,
        symbol TEXT,
        asset_type TEXT,
        trade_type TEXT,
        quantity INTEGER,
        execution_price REAL,
        execution_time TIMESTAMP,
        commission REAL
    )
''')

# Process executions and store them
for trade in executions:
    trade_id = trade.execId
    symbol = trade.contract.symbol
    asset_type = trade.contract.secType
    trade_type = trade.side  # Buy/Sell
    quantity = trade.shares
    execution_price = trade.price
    execution_time = trade.time
    commission = trade.commissionReport.commission

    cursor.execute('''
        INSERT OR IGNORE INTO trades 
        (trade_id, symbol, asset_type, trade_type, quantity, execution_price, execution_time, commission)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (trade_id, symbol, asset_type, trade_type, quantity, execution_price, execution_time, commission))

# Commit and close connection
conn.commit()
conn.close()

print("✅ Trade executions stored successfully!")
