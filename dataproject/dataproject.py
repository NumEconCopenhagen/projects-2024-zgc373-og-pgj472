
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
        self.tickers = ['AAPL', 'CVX', 'KO', 'JNJ']
        self.download_data()
        self.calculate_daily_returns()

    def six_stocks_(self):
        # Create a StockData object with the ticker symbols and specified time interval
        # We use Apple, Chevron Corporation, Coca-cola, McDonald's, Bank of America and Nike
        # Note: We do not need to pass start_year and end_year here since they are already defined in the __init__ method
        self.tickers = ['AAPL', 'CVX', 'KO', 'JNJ', 'BAC', 'NKE']
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
        "JNJ": "Johnson & Johnson",
        "BAC": "Bank of America Corporation",
        "NKE": "Nike, Inc."
        }
        
        expected_returns = self.calculate_expected_returns()
        expected_returns_percentage = {ticker: expected_return * 100 for ticker, expected_return in expected_returns.items()}
        df_expected_returns = pd.DataFrame(expected_returns_percentage.items(), columns=['Ticker', 'Expected Return (%)'])
        df_expected_returns.set_index('Ticker', inplace=True)
        
        expected_returns_df_filtered = df_expected_returns[~df_expected_returns.index.isin(["A", "P", "L"])]
        expected_returns_df_filtered.index = expected_returns_df_filtered.index.map(company_names)

        self.expected_returns_df_filtered = expected_returns_df_filtered

        return expected_returns_df_filtered


    def calculate_covariance(self, number):
        # Extract daily returns for the selected stocks
        daily_returns = {}
        for ticker, data in self.data.items():
            if ticker in self.tickers and data is not None:
                daily_returns[ticker] = data['Daily Return']

        # Create a DataFrame containing the daily returns for the selected stocks
        df = pd.DataFrame(daily_returns)

        # Calculate covariance matrix
        covariance_matrix = df.cov()

        if number == "two":
            # Define the two stocks
            two_stocks = ['AAPL', 'CVX']
            covariance_matrix = covariance_matrix.loc[two_stocks, two_stocks]
        elif number == "four":
            # Define the four stocks
            four_stocks = ['AAPL', 'CVX', 'KO', 'JNJ']
            covariance_matrix = covariance_matrix.loc[four_stocks, four_stocks]
        elif number == "six":
            # Define the six stocks
            six_stocks = ['AAPL', 'CVX', 'KO', 'JNJ', 'BAC', 'NKE']
            covariance_matrix = covariance_matrix.loc[six_stocks, six_stocks]

        return covariance_matrix

    def calculate_invers_covariance(self, number):
        print(f"Number: {number}")  # Add this line

        if number == "two":
            # Define the two stocks
            stocks = ['AAPL', 'CVX']
            # Calculate covariance matrix
            covariance_matrix = self.calculate_covariance("two")
        elif number == "four":
            # Define the four stocks
            stocks = ['AAPL', 'CVX', 'KO', 'JNJ']
            # Calculate covariance matrix
            covariance_matrix = self.calculate_covariance("four")
        elif number == "six":
            # Define the six stocks
            stocks = ['AAPL', 'CVX', 'KO', 'JNJ', 'BAC', 'NKE']
            # Calculate covariance matrix
            covariance_matrix = self.calculate_covariance("six")

        # Calculate inverse covariance matrix
        invers_covariance_matrix = np.linalg.inv(covariance_matrix)

        # Convert the inverse covariance matrix into a DataFrame with the stock names as the column and row names
        invers_covariance_matrix = pd.DataFrame(invers_covariance_matrix, columns=stocks, index=stocks)

        return invers_covariance_matrix

    def one_vector(self, number):
        if number == "two":
            # Define the two stocks
            stocks = ['AAPL', 'CVX']
        elif number == "four":
            # Define the four stocks
            stocks = ['AAPL', 'CVX', 'KO', 'JNJ']
        elif number == "six":
            # Define the six stocks
            stocks = ['AAPL', 'CVX', 'KO', 'JNJ', 'BAC', 'NKE']

        # Create a 1-vector with the same length as the number of stocks
        one_vector = np.ones((len(stocks), 1))

        return one_vector

    def calculate_z_vector(self, number):
        # Get the inverse covariance matrix
        invers_covariance_matrix = self.calculate_invers_covariance(number)

        # Get the 1-vector
        one_vector = self.one_vector(number)

        # Calculate the z-vector
        z_vector = np.dot(invers_covariance_matrix, one_vector)

        return z_vector

    def normalize_z_vector(self, number):
        # Get the z-vector
        z_vector = self.calculate_z_vector(number)

        # Calculate the sum of the z-vector
        z_sum = np.sum(z_vector)

        # Normalize the z-vector
        normalized_z_vector = z_vector / z_sum

        return normalized_z_vector
    
    def check_results(self, number):
        # Get the covariance matrix
        covariance_matrix = self.calculate_covariance(number)

        # Get the z-vector
        z_vector = self.calculate_z_vector(number)

        # Multiply the covariance matrix by the z-vector
        result = np.dot(covariance_matrix, z_vector)

        return result
    
    def calculate_portfolio_expected_return(self, number):
        # Get the normalized z-vector
        normalized_z_vector = self.normalize_z_vector(number)

        # Define the stocks based on the number
        if number == "two":
            stocks = ['AAPL', 'CVX']
        elif number == "four":
            stocks = ['AAPL', 'CVX', 'KO', 'JNJ']
        elif number == "six":
            stocks = ['AAPL', 'CVX', 'KO', 'JNJ', 'BAC', 'NKE']

        # Filter the DataFrame to only include the relevant stocks
        expected_returns_df_filtered = self.expected_returns_df_filtered[self.expected_returns_df_filtered['Expected Return (%)'].isin(stocks)]

        # Get the expected return for each stock from the DataFrame
        # Convert the percentages back to proportions by dividing by 100
        expected_returns = expected_returns_df_filtered['Expected Return (%)'] / 100

        # Calculate the expected return for the portfolio
        portfolio_expected_return = np.dot(expected_returns, normalized_z_vector)

        return portfolio_expected_return


    # Values is equal to expected return from the 'expected_returns_df_filtered'
    values_two_stock = [0.118517568275349, 0.053565249783654]
    values_four_stock = [0.118517568275349, 0.053565249783654, 0.039587017095891, 0.033614181941397]
    values_six_stock = [0.118517568275349, 0.053565249783654, 0.039587017095891, 0.033614181941397, 0.057441814728990, 0.053610256468914]   
    
    # Convert the list to a numpy array
    two_stock_vector = np.array(values_two_stock)
    four_stock_vector = np.array(values_four_stock)
    six_stock_vector = np.array(values_six_stock)

    #Normalized_z_vector values
    values_z_vector_two = [0.540426730926023, 0.459573269073977]
    values_z_vector_four = [0.074470299963800, 0.041449232247399, 0.433757794565312, 0.450322673223489]
    values_z_vector_six = [0.060682514783692, 0.046975603420424, 0.423018817709440, 0.445371017675944, -0.030914761366453, 0.054866807776954]   
    
    # Convert the list to a numpy array
    z_vector_two = np.array(values_z_vector_two)
    z_vector_four = np.array(values_z_vector_four)
    z_vector_six = np.array(values_z_vector_six)

    #Calculating the expected return for the whole portfolio for each set of stocks.
    portfolio_expected_return_two = np.dot(two_stock_vector, z_vector_two)
    portfolio_expected_return_four = np.dot(four_stock_vector, z_vector_four)
    portfolio_expected_return_six = np.dot(six_stock_vector, z_vector_six)

    def calculate_portfolio_variances(self):
        # Calculate the portfolio variance for each set of stocks
        self.portfolio_variance_two = np.dot(self.z_vector_two.T, np.dot(self.calculate_covariance("two"), self.z_vector_two))
        self.portfolio_variance_four = np.dot(self.z_vector_four.T, np.dot(self.calculate_covariance("four"), self.z_vector_four))
        self.portfolio_variance_six = np.dot(self.z_vector_six.T, np.dot(self.calculate_covariance("six"), self.z_vector_six))

    def calculate_portfolio_std_devs(self):
        # Calculate the portfolio standard deviation for each set of stocks
        self.portfolio_std_dev_two = np.sqrt(self.portfolio_variance_two)
        self.portfolio_std_dev_four = np.sqrt(self.portfolio_variance_four)
        self.portfolio_std_dev_six = np.sqrt(self.portfolio_variance_six)

