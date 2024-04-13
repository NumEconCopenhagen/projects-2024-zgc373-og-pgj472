# imports and display functions for the data project
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from matplotlib_venn import venn2
import yfinance as yf
from IPython.display import display
from IPython.display import HTML
from datetime import datetime

# We start by creating a class called StockData which will be used to create our functions
class StockData:
    # We define the initializer method for the class
    def __init__(self, tickers, start_year, end_year):
        # We store tickers, so it can be accessed in other methods later from the class
        self.tickers = tickers
        # We define start dates and end dates as the 1st of januar until 4th of april
        self.start_date = datetime(start_year, 1, 1)
        self.end_date = datetime(end_year, 4, 4)
        self.data = {}

        self.download_data()
        self.calculate_daily_returns()
        self.calculate_expected_returns()

        # We are looking for data in the ticker, to find data for different stocks
        for ticker, data in self.data.items():
            if data is not None:
                display(data.describe())
    
    # Here we download the data for the tickers (stocks) given the start and end dates.
    def download_data(self):
        for ticker in self.tickers:
            self.data[ticker] = yf.download(ticker, start=self.start_date, end=self.end_date)

    # We set up a function to calculate the daily returns for the different stocks
    def calculate_daily_returns(self):
        for ticker, data in self.data.items():
            if data is None:
                print(f"No data available for {ticker}. Download data first.")
                continue

            # Calculating daily returns from the closing price of stocks
            data['Daily Return'] = data['Adj Close'].pct_change()

    # We set up a function to calculate the expected returns for the different stocks
    def calculate_expected_returns(self):
        expected_returns = {}
        for ticker, data in self.data.items():
            if ticker in self.tickers:  # Check if the ticker is in the list of valid tickers
                if data is not None:
                    expected_returns[ticker] = data['Daily Return'].mean()
        
        return expected_returns
    
    # We are creating a StockData object using the ticker symbols and in frame of the time interval
    # Note: We do not need to pass start_year and end_year here since they are already defined in the __init__ method.
    # Note: This is the case for all three of the definitions below
    def two_stocks_(self):
        # We will use Apple and Chevron Corporation for the two_stock object
        self.tickers = ['AAPL', 'CVX']
        self.download_data()
        self.calculate_daily_returns()

    def four_stocks_(self):
        # We will use Apple, Chevron Corporation, Coca-cola and Johnson & Johnson for the four_stock object
        self.tickers = ['AAPL', 'CVX', 'KO', 'JNJ']
        self.download_data()
        self.calculate_daily_returns()

    def six_stocks_(self):
        # We will use Apple, Chevron Corporation, Coca-cola, Johnson & Johnson, Bank of America and Nike
        self.tickers = ['AAPL', 'CVX', 'KO', 'JNJ', 'BAC', 'NKE']
        self.download_data()
        self.calculate_daily_returns()
       
        # We are extracting the daily returns for each stock showed in the graph
        daily_returns = {}
        for ticker, data in self.data.items():
            if data is not None:
                daily_returns[ticker] = data['Daily Return']

        # We create a DataFrame containing the daily returns for each of the stocks
        df = pd.DataFrame(daily_returns)
        return df

    # We are creating a function to display the expected return for the different stocks
    def expected_return_(self):
        # We map the ticker symbols to company names
        company_names = {
        "AAPL": "Apple Inc.",
        "CVX": "Chevron Corporation",
        "KO": "The Coca-Cola Company",
        "JNJ": "Johnson & Johnson",
        "BAC": "Bank of America Corporation",
        "NKE": "Nike, Inc."
        }
        
        # We call the calculate_expected_returns function to get the expected return for each stock
        expected_returns = self.calculate_expected_returns() 

        # We convert the expected returns to percentages and store these in new dictionary
        expected_returns_percentage = {ticker: expected_return * 100 for ticker, expected_return in expected_returns.items()}

        # Here we convert the expected_returns_percentage dictionary to a DataFrame
        df_expected_returns = pd.DataFrame(expected_returns_percentage.items(), columns=['Ticker', 'Expected Return (%)'])

        # We are setting the 'Ticker' column as the index
        df_expected_returns.set_index('Ticker', inplace=True)
        
        # We are filtering out rows not consisting of stocks
        expected_returns_df_filtered = df_expected_returns[~df_expected_returns.index.isin(["A", "P", "L"])]

        # We replace the index with corresponding company names
        expected_returns_df_filtered.index = expected_returns_df_filtered.index.map(company_names)

        # We are storing the filtered DataFrame in the expected_returns_df_filtered attribute
        self.expected_returns_df_filtered = expected_returns_df_filtered

        return expected_returns_df_filtered

    # We are here creating a function to display the covariance matrix for the different stocks
    def calculate_covariance(self, number):
        # We extract daily returns for the selected stocks
        daily_returns = {}
        for ticker, data in self.data.items():
            if ticker in self.tickers and data is not None:
                daily_returns[ticker] = data['Daily Return']

        # We are creating a DataFrame containing the daily returns for the selected stocks
        df = pd.DataFrame(daily_returns)

        # This is to calculate the covariance matrix
        covariance_matrix = df.cov()

        # We will filter the covariance matrix, so we can distinguish between the different stock portfolios
        if number == "two":
            # Using the two stocks
            two_stocks = ['AAPL', 'CVX']
            covariance_matrix = covariance_matrix.loc[two_stocks, two_stocks]
        elif number == "four":
            # Using the four stocks
            four_stocks = ['AAPL', 'CVX', 'KO', 'JNJ']
            covariance_matrix = covariance_matrix.loc[four_stocks, four_stocks]
        elif number == "six":
            # Using the six stocks
            six_stocks = ['AAPL', 'CVX', 'KO', 'JNJ', 'BAC', 'NKE']
            covariance_matrix = covariance_matrix.loc[six_stocks, six_stocks]

        return covariance_matrix

    # We are defining the inverse of the covariance matrix
    def calculate_invers_covariance(self, number):
        print(f"Number: {number}")

        # We filter again to distinguish between the different stock portfolios
        if number == "two":
            # Using the two stocks
            stocks = ['AAPL', 'CVX']
            # Calculating the covariance matrix
            covariance_matrix = self.calculate_covariance("two")
        elif number == "four":
            # Using the four stocks
            stocks = ['AAPL', 'CVX', 'KO', 'JNJ']
            # Calculating the covariance matrix
            covariance_matrix = self.calculate_covariance("four")
        elif number == "six":
            # Using the six stocks
            stocks = ['AAPL', 'CVX', 'KO', 'JNJ', 'BAC', 'NKE']
            # Calculating the covariance matrix
            covariance_matrix = self.calculate_covariance("six")

        # We calculate the inverse covariance matrix using the numpy function np.linalg.inv
        invers_covariance_matrix = np.linalg.inv(covariance_matrix)

        # We are converting the inverse covariance matrix into a DataFrame with the stock names as the column and row names
        invers_covariance_matrix = pd.DataFrame(invers_covariance_matrix, columns=stocks, index=stocks)

        return invers_covariance_matrix

    # We are defining the 1-vector, which is used in calculations
    def one_vector(self, number):
        if number == "two":
            # Using the two stocks
            stocks = ['AAPL', 'CVX']
        elif number == "four":
            # Using the four stocks
            stocks = ['AAPL', 'CVX', 'KO', 'JNJ']
        elif number == "six":
            # Using the six stocks
            stocks = ['AAPL', 'CVX', 'KO', 'JNJ', 'BAC', 'NKE']

        # Here we then calculate the 1-vector with the same length as the number of stocks
        one_vector = np.ones((len(stocks), 1))

        return one_vector

    # Defining a z_vector function
    def calculate_z_vector(self, number):
        # We need the inverse covariance matrix
        invers_covariance_matrix = self.calculate_invers_covariance(number)

        # We need the 1-vector
        one_vector = self.one_vector(number)

        # We calculate the z_vector using the inverse covariance matrix and the 1-vector
        z_vector = np.dot(invers_covariance_matrix, one_vector)

        return z_vector

    # We define a normalization of the z-vector
    def normalize_z_vector(self, number):
        # We need to get the z-vector
        z_vector = self.calculate_z_vector(number)

        # We are calculating the sum of the z-vector
        z_sum = np.sum(z_vector)

        # We calculate the normalization of the z-vector
        normalized_z_vector = z_vector / z_sum

        return normalized_z_vector
    
    # We define a function, which can help us check that our results are correctly calculated
    def check_results(self, number):
        # We need to get the covariance matrix
        covariance_matrix = self.calculate_covariance(number)

        # We need to get the z-vector
        z_vector = self.calculate_z_vector(number)

        # To check our results we need to multiply the covariance matrix by the z-vector
        result = np.dot(covariance_matrix, z_vector)

        return result
    
    # We define a function to calculate the expected return for the whole portfolio for each set of stocks
    def calculate_portfolio_expected_return(self, number):
        # We need to get the normalized z-vector
        normalized_z_vector = self.normalize_z_vector(number)

        # We define the stocks based on the number to distinguish between the different stock portfolios
        if number == "two":
            stocks = ['AAPL', 'CVX']
        elif number == "four":
            stocks = ['AAPL', 'CVX', 'KO', 'JNJ']
        elif number == "six":
            stocks = ['AAPL', 'CVX', 'KO', 'JNJ', 'BAC', 'NKE']

        # We are filtering the DataFrame to only include the relevant stocks
        expected_returns_df_filtered = self.expected_returns_df_filtered[self.expected_returns_df_filtered['Expected Return (%)'].isin(stocks)]

        # We will get the expected return for each stock and convert the percentages back to proportions
        expected_returns = expected_returns_df_filtered['Expected Return (%)'] / 100

        # We calculate the expected return for the portfolio
        portfolio_expected_return = np.dot(expected_returns, normalized_z_vector)

        return portfolio_expected_return

    # Values is equal to expected return from the 'expected_returns_df_filtered'
    values_two_stock = [0.118517568275349, 0.053565249783654]
    values_four_stock = [0.118517568275349, 0.053565249783654, 0.039587017095891, 0.033614181941397]
    values_six_stock = [0.118517568275349, 0.053565249783654, 0.039587017095891, 0.033614181941397, 0.057441814728990, 0.053610256468914]   
    
    # We will convert the list to a numpy array, so we can use it in calculations
    two_stock_vector = np.array(values_two_stock)
    four_stock_vector = np.array(values_four_stock)
    six_stock_vector = np.array(values_six_stock)

    # We are here using the normalized_z_vector values (how much of each stock we should have in our portfolios)
    values_z_vector_two = [0.540426730926023, 0.459573269073977]
    values_z_vector_four = [0.074470299963800, 0.041449232247399, 0.433757794565312, 0.450322673223489]
    values_z_vector_six = [0.060682514783692, 0.046975603420424, 0.423018817709440, 0.445371017675944, -0.030914761366453, 0.054866807776954]   
    
    # We convert the list to a numpy array, so we can use it in calculations
    z_vector_two = np.array(values_z_vector_two)
    z_vector_four = np.array(values_z_vector_four)
    z_vector_six = np.array(values_z_vector_six)

    # We are calculating the expected return for the whole portfolio for each set of stocks.
    portfolio_expected_return_two = np.dot(two_stock_vector, z_vector_two)
    portfolio_expected_return_four = np.dot(four_stock_vector, z_vector_four)
    portfolio_expected_return_six = np.dot(six_stock_vector, z_vector_six)

    # We define a function to calculate the portfolio variance for each set
    def calculate_portfolio_variances(self):
        # We calculate the portfolio variance for each set of stocks
        self.portfolio_variance_two = np.dot(self.z_vector_two.T, np.dot(self.calculate_covariance("two"), self.z_vector_two))
        self.portfolio_variance_four = np.dot(self.z_vector_four.T, np.dot(self.calculate_covariance("four"), self.z_vector_four))
        self.portfolio_variance_six = np.dot(self.z_vector_six.T, np.dot(self.calculate_covariance("six"), self.z_vector_six))

    # We define a function to calculate the portfolio standard deviation for each set
    def calculate_portfolio_std_devs(self):
        # We calculate the portfolio standard deviation for each set of stocks
        self.portfolio_std_dev_two = np.sqrt(self.portfolio_variance_two)
        self.portfolio_std_dev_four = np.sqrt(self.portfolio_variance_four)
        self.portfolio_std_dev_six = np.sqrt(self.portfolio_variance_six)
