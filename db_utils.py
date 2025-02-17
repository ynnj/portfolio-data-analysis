# db_utils.py
import streamlit as st
import pandas as pd
import sqlite3
import os
import boto3
import tempfile

# 1. Configure S3 (using Streamlit Secrets) test
AWS_ACCESS_KEY_ID = st.secrets.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = st.secrets.get("AWS_SECRET_ACCESS_KEY")
BUCKET_NAME = st.secrets.get("AWS_BUCKET_NAME")
DB_FILE_NAME_IN_S3 = "paper_all_transactions.db"  # Replace with your actual file name

# 2. Function to download from S3
def download_db_from_s3(bucket_name, s3_file_name):
    """Downloads a file from S3 to a temporary file."""
    if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, BUCKET_NAME]): #check if all secrets are available
        st.error("Missing AWS credentials or bucket name in Streamlit secrets.")
        return None
    try:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp_file:
            s3.download_file(bucket_name, s3_file_name, tmp_file.name)
            tmp_file_path = tmp_file.name
        return tmp_file_path

    except Exception as e:
        st.error(f"Error downloading database: {e}")
        return None

# 3. Functions to interact with the database
def get_db_connection():
    """Gets a database connection (downloads from S3 if needed)."""
    db_file_path = download_db_from_s3(BUCKET_NAME, DB_FILE_NAME_IN_S3)
    if db_file_path is None:
        return None

    try:
        conn = sqlite3.connect(db_file_path)
        return conn
    except sqlite3.Error as e:
        st.error(f"Database connection error: {e}")
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
        st.error(f"Database query error: {e}")
        return None
    finally:
        if 'conn' in locals() and conn:
            conn.close()
        if 'db_file_path' in locals() and os.path.exists(db_file_path): #check if file exists before remove
            os.remove(db_file_path)


def execute_query(query):  # Example function to execute other queries (INSERT, UPDATE, etc.)
    """Executes a SQL query (INSERT, UPDATE, DELETE)."""
    conn = get_db_connection()
    if conn is None:
        return False  # Or raise an exception

    try:
        cursor = conn.cursor()
        cursor.execute(query)
        conn.commit()  # Important for write operations
        return True
    except sqlite3.Error as e:
        st.error(f"Database query error: {e}")
        conn.rollback()  # Rollback on error
        return False
    finally:
        if 'conn' in locals() and conn:
            conn.close()
        if 'db_file_path' in locals() and os.path.exists(db_file_path): #check if file exists before remove
            os.remove(db_file_path)

# ... (Add other database functions as needed)