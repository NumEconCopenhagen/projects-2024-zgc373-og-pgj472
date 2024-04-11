import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from matplotlib_venn import venn2
import yfinance as yf
from IPython.display import display
from IPython.display import HTML
from datetime import datetime

class StockData:
    def __init__(self, tickers, start_year, end_year):
        self.tickers = tickers
        self.start_date = datetime(start_year, 1, 1)
        self.end_date = datetime(end_year, 4, 4)
        self.data = {}

        self.download_data()
        self.calculate_daily_returns()
        self.calculate_expected_returns()

        for ticker, data in self.data.items():
            if data is not None:
                display(data.describe())
    
    def download_data(self):
        for ticker in self.tickers:
            self.data[ticker] = yf.download(ticker, start=self.start_date, end=self.end_date)

    def calculate_daily_returns(self):
        for ticker, data in self.data.items():
            if data is None:
                print(f"No data available for {ticker}. Download data first.")
                continue

            # Calculate daily returns
            data['Daily Return'] = data['Adj Close'].pct_change()

    def calculate_expected_returns(self):
        expected_returns = {}
        for ticker, data in self.data.items():
            if ticker in self.tickers:  # Check if the ticker is in the list of valid tickers
                if data is not None:
                expected_returns[ticker] = data['Daily Return'].mean()
        
        return expected_returns

    def two_stocks_(self):
        # Create a StockData object with the ticker symbols and specified time interval
        # We use Apple and Chevron Corporation
        # Note: We do not need to pass start_year and end_year here since they are already defined in the __init__ method
        self.tickers = ['AAPL', 'CVX']
        self.download_data()
        self.calculate_daily_returns()

    def four_stocks_(self):
        # Create a StockData object with the ticker symbols and specified time interval
        # We use Apple, Chevron Corporation, Coca-cola and McDonald's
        # Note: We do not need to pass start_year and end_year here since they are already defined in the __init__ method
        self.tickers = ['AAPL', 'CVX', 'KO', 'MCD']
        self.download_data()
        self.calculate_daily_returns()

    def six_stocks_(self):
        # Create a StockData object with the ticker symbols and specified time interval
        # We use Apple, Chevron Corporation, Coca-cola, McDonald's, Bank of America and Nike
        # Note: We do not need to pass start_year and end_year here since they are already defined in the __init__ method
        self.tickers = ['AAPL', 'CVX', 'KO', 'MCD', 'BAC', 'NKE']
        self.download_data()
        self.calculate_daily_returns()
       
        # Extract daily returns for each stock
        daily_returns = {}
        for ticker, data in self.data.items():
            if data is not None:
                daily_returns[ticker] = data['Daily Return']

        # Create a DataFrame containing the daily returns for the six stocks
        df = pd.DataFrame(daily_returns)
        return df

    def expected_return_(self):
        # Calculate expected returns
        expected_returns = self.calculate_expected_returns()

        # Convert expected returns to percentages
        expected_returns_percentage = {ticker: expected_return * 100 for ticker, expected_return in expected_returns.items()}

        # Create a DataFrame for expected returns
        df_expected_returns = pd.DataFrame(expected_returns_percentage.items(), columns=['Ticker', 'Expected Return (%)'])

        # Set the Ticker column as index
        df_expected_returns.set_index('Ticker', inplace=True)
        
        # Return the DataFrame containing expected returns
        return df_expected_returns
        
