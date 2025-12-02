'''
Download historical stock data using yFinance and save it as CSV files.
'''

import yfinance as yf
import pandas as pd
import os
import argparse
from datetime import datetime

# Function to download stock data
def download_stock_data(ticker, start_date, end_date, interval, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Download stock data
    stock_data = yf.download(ticker, start=start_date, end=end_date, interval=interval)

    # Flatten MultiIndex columns if present
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = stock_data.columns.get_level_values(0)

    # Save to CSV
    output_file = os.path.join(output_dir, f"{ticker}_{start_date}_{end_date}_{interval}.csv")
    stock_data.to_csv(output_file)
    print(f"Data for {ticker} saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download historical stock data using yFinance.")
    parser.add_argument("--tickers", nargs='+', required=True, help="List of stock tickers to download. Example: AAPL MSFT GOOGL")
    parser.add_argument("--start_date", type=str, required=True, help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end_date", type=str, required=True, help="End date in YYYY-MM-DD format.")
    parser.add_argument("--interval", type=str, default="1d", help="Data interval (e.g., 1d, 1h, 15m).")
    parser.add_argument("--output_dir", type=str, default="stock_data", help="Directory to save CSV files.")

    args = parser.parse_args()

    for ticker in args.tickers:
        download_stock_data(ticker, args.start_date, args.end_date, args.interval, args.output_dir)