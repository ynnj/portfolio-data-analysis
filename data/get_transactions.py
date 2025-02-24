import sqlite3
import requests
import pandas as pd
import time
import os
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
import math
import numpy as np
from io import StringIO
import argparse

# Load environment variables
load_dotenv()

# Argument parsing to select database
parser = argparse.ArgumentParser(description="Process transactions for a specific account.")
parser.add_argument("account_type", choices=["real", "paper"], help="Specify which account to process: real or paper")
args = parser.parse_args()

# Select database based on the argument
DB_NAME = "real_all_transactions.db" if args.account_type == "real" else "paper_all_transactions.db"

if args.account_type=="paper":
    IBKR_TOKEN = os.getenv("IBKR_TOKEN_PAPER")
    FLEX_QUERY_ID = os.getenv("FLEX_QUERY_ID_PAPER")
else:
    IBKR_TOKEN = os.getenv("IBKR_TOKEN_REAL")
    FLEX_QUERY_ID = os.getenv("FLEX_QUERY_ID_REAL")

SEND_REQUEST_URL = "https://ndcdyn.interactivebrokers.com/AccountManagement/FlexWebService/SendRequest"
GET_STATEMENT_URL = "https://ndcdyn.interactivebrokers.com/AccountManagement/FlexWebService/GetStatement"


def create_connection(db_name=DB_NAME):
    conn = sqlite3.connect(DB_NAME)
    return conn

def create_transactions_table():
    conn = create_connection()
    cursor = conn.cursor()

    # Check if the table exists. Only create it if it doesn't.
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='transactions'")
    table_exists = cursor.fetchone() is not None

    if not table_exists:  # Create only if it doesn't exist already
        cursor.execute("""
            CREATE TABLE transactions (
                ClientAccountID TEXT,
                AccountAlias TEXT,
                Model TEXT,
                CurrencyPrimary TEXT,
                FXRateToBase REAL,
                AssetClass TEXT,
                SubCategory TEXT,
                Symbol TEXT,
                TransactionID INTEGER PRIMARY KEY,  -- Keep TransactionID as PRIMARY KEY
                Description TEXT,
                Conid INTEGER,
                SecurityID TEXT,
                SecurityIDType TEXT,
                CUSIP TEXT,
                ISIN TEXT,
                FIGI TEXT,
                ListingExchange TEXT,
                UnderlyingConid INTEGER,
                UnderlyingSymbol TEXT,
                UnderlyingSecurityID TEXT,
                UnderlyingListingExchange TEXT,
                Issuer TEXT,
                IssuerCountryCode TEXT,
                TradeID INTEGER,
                Multiplier INTEGER,
                RelatedTradeID INTEGER,
                Strike REAL,
                ReportDate INTEGER,
                Expiry INTEGER,
                DateTime TEXT,
                PutCall TEXT,
                TradeDate INTEGER,
                PrincipalAdjustFactor REAL,
                SettleDateTarget INTEGER,
                TransactionType TEXT,
                Exchange TEXT,
                Quantity INTEGER,
                TradePrice REAL,
                TradeMoney REAL,
                Proceeds REAL,
                Taxes REAL,
                IBCommission REAL,
                IBCommissionCurrency TEXT,
                NetCash REAL,
                ClosePrice REAL,
                OpenCloseIndicator TEXT,
                NotesCodes TEXT,
                CostBasis REAL,
                FifoPnlRealized REAL,
                MtmPnl REAL,
                OrigTradePrice REAL,
                OrigTradeDate TEXT,
                OrigTradeID INTEGER,
                OrigOrderID INTEGER,
                OrigTransactionID INTEGER,
                BuySell TEXT,
                ClearingFirmID TEXT,
                IBOrderID INTEGER,
                IBExecID TEXT,
                RelatedTransactionID TEXT,
                RTN TEXT,
                BrokerageOrderID TEXT,
                OrderReference TEXT,
                VolatilityOrderLink TEXT,
                ExchOrderID TEXT,
                ExtExecID TEXT,
                OrderTime TEXT,
                OpenDateTime TEXT,
                HoldingPeriodDateTime TEXT,
                WhenRealized TEXT,
                WhenReopened TEXT,
                LevelOfDetail TEXT,
                ChangeInPrice REAL,
                ChangeInQuantity INTEGER,
                OrderType TEXT,
                TraderID TEXT,
                IsAPIOrder TEXT,
                AccruedInterest REAL,
                InitialInvestment REAL,
                SerialNumber TEXT,
                DeliveryType TEXT,
                CommodityType TEXT,
                Fineness REAL,
                Weight REAL,
                Processed INTEGER
            )
        """)
        conn.commit()
        print("Transactions table created.")
    else:
        print("Transactions table already exists.")

    conn.close()

def get_flex_query_report(token, query_id):
    """
    Fetches a Flex Query Report reference code from IBKR Client Portal API.
    """
    params = {
        "t": token,
        "q": query_id,
        "v": 3
    }
    
    response = requests.get(SEND_REQUEST_URL, params=params)
    
    if response.status_code == 200:
        try:
            root = ET.fromstring(response.text)  # Parse XML response
            status = root.find("Status").text
            if status != "Success":
                raise Exception(f"IBKR API returned an error: {status}")
            
            reference_code = root.find("ReferenceCode").text
            return reference_code
        except ET.ParseError:
            raise Exception("Failed to parse XML response")
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")

def download_flex_report(token, reference_code):
    """
    Downloads the trade report using the reference code.
    """
    params = {
        "t": token,
        "q": reference_code,
        "v": 3
    }
    
    while True:
        response = requests.get(GET_STATEMENT_URL, params=params)
        if response.status_code == 200:
            return response.text
        elif response.status_code == 202:
            print("Report is still processing. Retrying in 5 seconds...")
            time.sleep(5)
        else:
            raise Exception(f"Error: {response.status_code}, {response.text}")

def clean_transaction_data(transaction):
    for key, value in transaction.items():
        if isinstance(value, float) and (math.isnan(value) or np.isnan(value)):
            transaction[key] = None  # Replace NaN with None for DB insertion
    return transaction

def process_csv_data(csv_data):
    """
    Parses and processes CSV data into a DataFrame.
    """
    
    df = pd.read_csv(StringIO(csv_data))
    save_to_csv(df)
    # Replace slashes (/) with underscores (_) in column names
    df.columns = [col.replace('/', '_') for col in df.columns]
    df.columns = [col.replace('Put_Call', 'PutCall') for col in df.columns]
    df.columns = [col.replace('Open_CloseIndicator', 'OpenCloseIndicator') for col in df.columns]
    df.columns = [col.replace('Notes_Codes', 'NotesCodes') for col in df.columns]
    df.columns = [col.replace('Buy_Sell', 'BuySell') for col in df.columns]
    
    # Convert DataFrame to list of dictionaries
    transactions = df.to_dict(orient="records")
    return transactions

def insert_transaction(transaction):
    conn = create_connection()  # Your connection function here
    cursor = conn.cursor()

    # Clean transaction data to handle NaN values
    print("Before Cleaning:", transaction)
    transaction = clean_transaction_data(transaction)
    print("After Cleaning:", transaction)

    # Check if TransactionID is None
    if 'TransactionID' not in transaction or transaction['TransactionID'] is None:
        print("TransactionID is missing or None, skipping this transaction.")
        return

    try:
        # Check if the TransactionID already exists
        cursor.execute("SELECT COUNT(*) FROM transactions WHERE TransactionID = ?", (transaction['TransactionID'],))
        if cursor.fetchone()[0] == 0:
            # Attempt to insert the transaction into the database
            cursor.execute("""
                INSERT INTO transactions (
                    ClientAccountID, AccountAlias, Model, CurrencyPrimary, FXRateToBase, 
                    AssetClass, SubCategory, Symbol, TransactionID, Description, Conid,SecurityID,SecurityIDType,CUSIP, ISIN, FIGI,ListingExchange, UnderlyingConid, 
                    UnderlyingSymbol, UnderlyingSecurityID, UnderlyingListingExchange, Issuer, 
                    IssuerCountryCode, TradeID, Multiplier, RelatedTradeID, Strike,ReportDate,Expiry,DateTime,PutCall,TradeDate,PrincipalAdjustFactor,SettleDateTarget,TransactionType,Exchange,Quantity,TradePrice,TradeMoney,Proceeds,Taxes,IBCommission, IBCommissionCurrency,NetCash,ClosePrice,OpenCloseIndicator,NotesCodes,CostBasis,FifoPnlRealized,MtmPnl,OrigTradePrice,OrigTradeDate,OrigTradeID,OrigOrderID,OrigTransactionID,BuySell,
                    ClearingFirmID,IBOrderID,IBExecID,RelatedTransactionID,RTN,BrokerageOrderID,OrderReference,VolatilityOrderLink,ExchOrderID,ExtExecID,OrderTime,OpenDateTime,     HoldingPeriodDateTime,WhenRealized,WhenReopened,LevelOfDetail,ChangeInPrice,ChangeInQuantity,OrderType,TraderID,IsAPIOrder,AccruedInterest,InitialInvestment,SerialNumber,DeliveryType,CommodityType,Fineness,Weight,Processed
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
                )
            """, (
                transaction['ClientAccountID'], transaction['AccountAlias'], transaction['Model'], 
                transaction['CurrencyPrimary'], transaction['FXRateToBase'], transaction['AssetClass'], 
                transaction['SubCategory'], transaction['Symbol'], transaction['TransactionID'],transaction['Description'], 
                transaction['Conid'], transaction['SecurityID'], transaction['SecurityIDType'], 
                transaction['CUSIP'], transaction['ISIN'], transaction['FIGI'], transaction['ListingExchange'], 
                transaction['UnderlyingConid'], transaction['UnderlyingSymbol'], transaction['UnderlyingSecurityID'], 
                transaction['UnderlyingListingExchange'], transaction['Issuer'], transaction['IssuerCountryCode'], 
                transaction['TradeID'], transaction['Multiplier'], transaction['RelatedTradeID'], transaction['Strike'],transaction['ReportDate'],transaction['Expiry'],transaction['DateTime'],transaction['PutCall'],transaction['TradeDate'], transaction['PrincipalAdjustFactor'], transaction['SettleDateTarget'], 
                transaction['TransactionType'], transaction['Exchange'], transaction['Quantity'], transaction['TradePrice'], 
                transaction['TradeMoney'], transaction['Proceeds'], transaction['Taxes'], transaction['IBCommission'],transaction['IBCommissionCurrency'], transaction['NetCash'], transaction['ClosePrice'], 
                transaction['OpenCloseIndicator'], transaction['NotesCodes'],transaction['CostBasis'], 
                transaction['FifoPnlRealized'], transaction['MtmPnl'], transaction['OrigTradePrice'], 
                transaction['OrigTradeDate'], transaction['OrigTradeID'], transaction['OrigOrderID'], 
                transaction['OrigTransactionID'], transaction['BuySell'],transaction['ClearingFirmID'], 
                transaction['IBOrderID'], transaction['IBExecID'], 
                transaction['RelatedTransactionID'], transaction['RTN'], transaction['BrokerageOrderID'], 
                transaction['OrderReference'], transaction['VolatilityOrderLink'], transaction['ExchOrderID'], 
                transaction['ExtExecID'], transaction['OrderTime'], transaction['OpenDateTime'],transaction['HoldingPeriodDateTime'], transaction['WhenRealized'], transaction['WhenReopened'], 
                transaction['LevelOfDetail'], transaction['ChangeInPrice'], transaction['ChangeInQuantity'], 
                transaction['OrderType'], transaction['TraderID'], transaction['IsAPIOrder'], 
                transaction['AccruedInterest'], transaction['InitialInvestment'], transaction['SerialNumber'], 
                transaction['DeliveryType'], transaction['CommodityType'], transaction['Fineness'], 
                transaction['Weight'],0
            ))
            conn.commit()
            print(f"Transaction {transaction['TransactionID']} inserted successfully.")
        else:
            print(f"Duplicate found for TransactionID: {transaction['TransactionID']}. Skipping insertion.")
    
    except sqlite3.DatabaseError as e:
        # Catch database-specific errors
        print(f"Database error: {e}")
    
    except Exception as e:
        # Catch other errors that might occur during insertion
        print(f"An error occurred: {e}")

    finally:
        conn.close()

def save_to_csv(df, filename="trades.csv"):
    try:
        df.to_csv(filename, index=False, encoding='utf-8')  # Explicit encoding
        print(f"Data saved to {filename}")
    except Exception as e:
        print(f"Error saving to CSV: {e}")

def process_transactions_from_csv(csv_filepath):
    """Processes transactions from a CSV file."""
    try:
        with open(csv_filepath, 'r', encoding='utf-8') as file:  # Handle encoding
            csv_data = file.read() # read the csv file as string
        transactions = process_csv_data(csv_data) # use the same function
        return transactions
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_filepath}")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: CSV file is empty: {csv_filepath}")
        return None
    except Exception as e:
        print(f"Error processing CSV: {e}")
        return None

if __name__ == "__main__":
    create_transactions_table()
    try:
        reference_code = get_flex_query_report(IBKR_TOKEN, FLEX_QUERY_ID)
        csv_data = download_flex_report(IBKR_TOKEN, reference_code)  # This is already CSV data

        transactions = process_csv_data(csv_data)  # Reuse process_csv_data

        for transaction in transactions:
            insert_transaction(transaction)

    except Exception as e:
        print("Error using IBKR API:", e)