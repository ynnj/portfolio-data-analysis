import pandas as pd
import sqlite3
import os

# Define file paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
csv_file = os.path.join(SCRIPT_DIR, "Tradersync-2.csv")
db_file = os.path.join(SCRIPT_DIR, "trades.db")  # Store database in the same directory


# Load the CSV file into a Pandas DataFrame
df = pd.read_csv(csv_file)

# Rename columns to match a structured database schema
df.rename(columns={
    "CurrencyPrimary": "currency",
    "AssetClass": "asset_class",
    "Symbol": "symbol",
    "Description": "description",
    "Conid": "conid",
    "UnderlyingConid": "underlying_conid",
    "UnderlyingSymbol": "underlying_symbol",
    "Multiplier": "multiplier",
    "Strike": "strike",
    "Expiry": "expiry",
    "Put/Call": "right",
    "TradeID": "trade_id",
    "Date/Time": "execution_time",
    "TradeDate": "trade_date",
    "Buy/Sell": "side",
    "Quantity": "shares",
    "Price": "price",
    "Amount": "amount",
    "Proceeds": "proceeds",
    "NetCash": "net_cash",
    "Commission": "commission",
    "BrokerExecutionCommission": "broker_execution_commission",
    "BrokerClearingCommission": "broker_clearing_commission",
    "ThirdPartyExecutionCommission": "third_party_execution_commission",
    "ThirdPartyClearingCommission": "third_party_clearing_commission",
    "ThirdPartyRegulatoryCommission": "third_party_regulatory_commission",
    "OtherCommission": "other_commission",
    "OrderType": "order_type"
}, inplace=True)

# Convert column types
df["execution_time"] = pd.to_datetime(df["execution_time"])
df["trade_date"] = pd.to_datetime(df["trade_date"])
df["price"] = pd.to_numeric(df["price"])
df["shares"] = pd.to_numeric(df["shares"])
df["strike"] = pd.to_numeric(df["strike"], errors='coerce')  # Handle missing values
df["commission"] = pd.to_numeric(df["commission"], errors='coerce')

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

# Define table schema and create the `trades` table if it doesn't exist
cursor.execute('''
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        currency TEXT,
        asset_class TEXT,
        symbol TEXT,
        description TEXT,
        conid INTEGER,
        underlying_conid INTEGER,
        underlying_symbol TEXT,
        multiplier INTEGER,
        strike REAL,
        expiry TEXT,
        right TEXT,
        trade_id TEXT UNIQUE,
        execution_time TEXT,
        trade_date TEXT,
        side TEXT,
        shares INTEGER,
        price REAL,
        amount REAL,
        proceeds REAL,
        net_cash REAL,
        commission REAL,
        broker_execution_commission REAL,
        broker_clearing_commission REAL,
        third_party_execution_commission REAL,
        third_party_clearing_commission REAL,
        third_party_regulatory_commission REAL,
        other_commission REAL,
        order_type TEXT,
        processed INTEGER DEFAULT 0
    )
''')

# Insert data into the database
df.to_sql("trades", conn, if_exists="append", index=False)

# Commit and close connection
conn.commit()
conn.close()

print("✅ CSV file successfully converted to trades.db")
