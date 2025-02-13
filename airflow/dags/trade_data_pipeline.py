from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator  # For running scripts
from datetime import datetime
import os

# Define paths (adjust these to your actual paths)
PROJECT_ROOT = "/Users/jennyhu/Documents/Projects/portfolio-data-analysis"  # Replace with your project root
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
GET_TRANSACTIONS_SCRIPT = os.path.join(DATA_DIR, "get_transactions2.py")
PROCESS_TRANSACTIONS_SCRIPT = os.path.join(DATA_DIR, "process_transactions2.py")
PYTHON_EXECUTABLE = "/full/path/to/your/python3"  # Replace with the full path to your Python 3

with DAG(
    dag_id="trade_data_pipeline",
    start_date=datetime(2024, 1, 1),  # Adjust start date as needed
    schedule_interval="0 11 * * *",  # 11:00 AM daily
    catchup=False,  # Don't backfill
    default_args={
        "owner": "your_name",  # Replace with your name
        "retries": 1,  # Number of retries if a task fails
    },
) as dag:
    get_transactions = BashOperator(
        task_id="get_transactions",
        bash_command=f"{PYTHON_EXECUTABLE} {GET_TRANSACTIONS_SCRIPT}",
        # You can also use env to set environment variables if needed
        # env={"YOUR_API_KEY": "your_api_key_value"},
    )

    process_transactions = BashOperator(
        task_id="process_transactions",
        bash_command=f"{PYTHON_EXECUTABLE} {PROCESS_TRANSACTIONS_SCRIPT}",
    )

    get_transactions >> process_transactions  # Define task dependency