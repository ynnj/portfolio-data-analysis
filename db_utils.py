import os
import sqlite3
import pandas as pd
import tempfile
import boto3
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

APP_ENV = os.getenv("APP_ENV", "development")  # Default to 'development'

if APP_ENV == "production":
    try:
        import streamlit as st  # Import Streamlit only when needed
        AWS_ACCESS_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
        AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
        AWS_BUCKET_NAME = st.secrets["AWS_BUCKET_NAME"]
        DB_FILE_NAME_IN_S3 = st.secrets["DATABASE_NAME_PAPER"]
    except Exception as e:
        print(f"⚠️ Error loading Streamlit secrets: {e}")
        exit(1)
elif APP_ENV == "development":
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
    DB_FILE_NAME_IN_S3 = os.getenv("DATABASE_NAME_PAPER")
else:
    print(f"❌ Invalid APP_ENV: {APP_ENV}. Must be 'production' or 'development'.")
    exit(1)

def download_db_from_s3(bucket_name, s3_file_name):
    """Downloads a file from S3 to a temporary file."""
    if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_BUCKET_NAME]):
        print("❌ Missing AWS credentials or bucket name.")
        return None

    try:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp_file:
            s3.download_file(bucket_name, s3_file_name, tmp_file.name)
            return tmp_file.name

    except Exception as e:
        print(f"❌ Error downloading database: {e}")
        return None

def get_db_connection():
    """Gets a database connection (downloads from S3 if needed)."""
    db_file_path = download_db_from_s3(AWS_BUCKET_NAME, DB_FILE_NAME_IN_S3)
    if db_file_path is None:
        return None

    try:
        conn = sqlite3.connect(db_file_path)
        return conn
    except sqlite3.Error as e:
        print(f"❌ Database connection error: {e}")
        return None

def fetch_data(query):
    """Fetches data from the database."""
    conn = get_db_connection()
    if conn is None:
        return None

    try:
        df = pd.read_sql_query(query, conn)
        return df
    except sqlite3.Error as e:
        print(f"❌ Database query error: {e}")
        return None
    finally:
        if 'conn' in locals() and conn:
            conn.close()

def execute_query(query):
    """Executes a SQL query (INSERT, UPDATE, DELETE)."""
    conn = get_db_connection()
    if conn is None:
        return False

    try:
        cursor = conn.cursor()
        cursor.execute(query)
        conn.commit()
        return True
    except sqlite3.Error as e:
        print(f"❌ Database query error: {e}")
        conn.rollback()
        return False
    finally:
        if 'conn' in locals() and conn:
            conn.close()

def get_metrics_df():
    """Retrieves the 'metrics' table as a DataFrame."""
    metrics_query = "SELECT * FROM trade_metrics;"  # Or specify the columns you need
    metrics_df = fetch_data(metrics_query)  # Directly call fetch_data from the current script
    return metrics_df

def get_merged_trades_df():
    """Retrieves the 'merged_trades' table as a DataFrame."""
    merged_trades_query = "SELECT * FROM merged_trades;"  # Or specify the columns you need
    merged_trades_df = fetch_data(merged_trades_query)  # Directly call fetch_data from the current script
    return merged_trades_df


# === TEST SCRIPT ===
if __name__ == "__main__":
    print("🔍 Testing database connection...")
    conn = get_db_connection()
    if conn:
        print("✅ Connection successful!")
        conn.close()
    else:
        print("❌ Connection failed!")

    print("\n🔍 Testing data fetch...")
    sample_query = "SELECT name FROM sqlite_master WHERE type='table';"
    tables = fetch_data(sample_query)
    if tables is not None:
        print(f"✅ Fetch successful! Tables: {tables.to_dict(orient='records')}")
    else:
        print("❌ Fetch failed!")

    print("\n🔍 Testing query execution (Creating test table)...")
    test_query = "CREATE TABLE IF NOT EXISTS test_table (id INTEGER PRIMARY KEY, name TEXT);"
    if execute_query(test_query):
        print("✅ Query execution successful!")
    else:
        print("❌ Query execution failed!")
