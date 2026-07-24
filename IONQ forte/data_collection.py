import yfinance as yf
import numpy as np
import pandas as pd

def fetch_data(tickers, start_date, end_date):
    print(f"Fetching data for {len(tickers)} tickers...")
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    
    # Drop columns with all NaNs just in case
    data = data.dropna(axis=1, how='all')
    # Forward fill and backward fill for missing data
    data = data.ffill().bfill()
    
    # Compute log returns
    log_returns = np.log(data / data.shift(1)).dropna()
    
    # Expected returns (annualized by multiplying by 252)
    mu = log_returns.mean().values * 252
    
    # Covariance matrix (annualized)
    C = log_returns.cov().values * 252
    
    # Correlation matrix
    C_corr = log_returns.corr().values
    
    return mu, C, C_corr, list(data.columns), log_returns

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "JNJ", "V", 
               "PG", "UNH", "HD", "MA", "DIS", "PYPL", "VZ", "ADBE", "NFLX", "INTC", 
               "CMCSA", "PFE", "T", "PEP", "CSCO", "XOM", "KO", "ABT", "MRK", "CVX"]
    mu, C, C_corr, actual_tickers = fetch_data(tickers, "2020-01-02", "2023-12-31")
    print(f"Fetched {len(actual_tickers)} assets.")
    print("mu shape:", mu.shape)
    print("C shape:", C.shape)
    print("C_corr shape:", C_corr.shape)
