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
        # Mapping of ticker symbols to company names
        company_names = {
        "AAPL": "Apple Inc.",
        "CVX": "Chevron Corporation",
        "KO": "The Coca-Cola Company",
        "MCD": "McDonald's Corporation",
        "BAC": "Bank of America Corporation",
        "NKE": "Nike, Inc."
        }
        
        # Calculate expected returns
        expected_returns = self.calculate_expected_returns()

        # Convert expected returns to percentages
        expected_returns_percentage = {ticker: expected_return * 100 for ticker, expected_return in expected_returns.items()}

        # Create a DataFrame for expected returns
        df_expected_returns = pd.DataFrame(expected_returns_percentage.items(), columns=['Ticker', 'Expected Return (%)'])

        # Set the Ticker column as index
        df_expected_returns.set_index('Ticker', inplace=True)
        
        # Filter out unwanted ticker symbols
        expected_returns_df_filtered = df_expected_returns[~df_expected_returns.index.isin(["A", "P", "L"])]

        # Replace index labels (ticker symbols) with company names
        expected_returns_df_filtered.index = expected_returns_df_filtered.index.map(company_names)

        # Return the DataFrame containing expected returns
        return expected_returns_df_filtered

    def calculate_covariance(self):
        # Extract daily returns for the selected stocks
        daily_returns = {}
        for ticker, data in self.data.items():
            if ticker in self.tickers and data is not None:
                daily_returns[ticker] = data['Daily Return']

        # Create a DataFrame containing the daily returns for the selected stocks
        df = pd.DataFrame(daily_returns)

        # Calculate covariance matrix
        covariance_matrix = df.cov()

        return covariance_matrix

    def two_stocks_covariance(self):
        # Define the two stocks
        two_stocks = ['AAPL', 'CVX']
        covariance_matrix = self.calculate_covariance()
        return covariance_matrix.loc[two_stocks, two_stocks]

    def four_stocks_covariance(self):
        # Define the four stocks
        four_stocks = ['AAPL', 'CVX', 'KO', 'MCD']
        covariance_matrix = self.calculate_covariance()
        return covariance_matrix.loc[four_stocks, four_stocks]

    def six_stocks_covariance(self):
        # Define the six stocks
        six_stocks = ['AAPL', 'CVX', 'KO', 'MCD', 'BAC', 'NKE']
        covariance_matrix = self.calculate_covariance()
        return covariance_matrix.loc[six_stocks, six_stocks]

    def calculate_invers_covariance(self, number):
        if number == "two":
            # Calculate covariance matrix
            covariance_matrix = self.two_stocks_covariance()
        elif number == "four":
            # Calculate covariance matrix
            covariance_matrix = self.four_stocks_covariance()
        elif number == "six":
            # Calculate covariance matrix
            covariance_matrix = self.six_stocks_covariance()
    
        # Calculate inverse covariance matrix
        invers_covariance_matrix = np.linalg.inv(covariance_matrix)

        invers_covariance_matrix = pd.DataFrame(invers_covariance_matrix)
    
        return invers_covariance_matrix


