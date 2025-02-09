import requests
import pandas as pd
import time

# IBKR Client Portal API base URL (update if needed)
IBKR_BASE_URL = "https://localhost:5000/v1/api"

# Disable SSL warnings (since IBKR uses self-signed certificates)
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def authenticate():
    """Ping IBKR API to check if the connection is active."""
    url = f"{IBKR_BASE_URL}/iserver/auth/status"
    response = requests.get(url, verify=False)  # Disable SSL verification
    return response.json()

def get_accounts():
    """Retrieve IBKR account IDs."""
    url = f"{IBKR_BASE_URL}/iserver/accounts"
    response = requests.get(url, verify=False)
    return response.json()

def get_trade_history(account_id):
    """Fetch trade history for a given IBKR account."""
    url = f"{IBKR_BASE_URL}/iserver/account/{account_id}/trades"
    response = requests.get(url, verify=False)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

def save_trades_to_csv(trades, filename="ibkr_trades.csv"):
    """Save trades to a CSV file."""
    if trades:
        df = pd.DataFrame(trades)
        df.to_csv(filename, index=False)
        print(f"✅ Trades saved to {filename}")
    else:
        print("⚠️ No trades found.")

def main():
    """Main function to authenticate and fetch IBKR trade history."""
    print("🔄 Checking IBKR API status...")
    auth_status = authenticate()
    print("✅ Authentication:", auth_status)

    print("\n🔄 Retrieving IBKR accounts...")
    accounts = get_accounts()
    if accounts and "accounts" in accounts:
        account_id = accounts["accounts"][0]  # Use the first account
        print(f"✅ Using Account ID: {account_id}")

        print("\n🔄 Fetching trade history...")
        trades = get_trade_history(account_id)
        
        if trades:
            print(f"✅ Retrieved {len(trades)} trades.")
            save_trades_to_csv(trades)
        else:
            print("⚠️ No trade history found.")
    else:
        print("⚠️ Failed to retrieve accounts.")

if __name__ == "__main__":
    main()
