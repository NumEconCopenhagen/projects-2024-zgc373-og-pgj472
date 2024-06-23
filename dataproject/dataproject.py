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

            # We calculate daily returns from the closing price of stocks
            data['Daily Return'] = data['Adj Close'].pct_change()

    # We set up a function to calculate the expected returns for the different stocks
    def calculate_expected_returns(self):
        expected_returns = {}
        for ticker, data in self.data.items():
            if ticker in self.tickers:
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
            #We calculate the covariance matrix
            covariance_matrix = self.calculate_covariance("two")
        elif number == "four":
            # Using the four stocks
            stocks = ['AAPL', 'CVX', 'KO', 'JNJ']
            #We calculate the covariance matrix
            covariance_matrix = self.calculate_covariance("four")
        elif number == "six":
            # Using the six stocks
            stocks = ['AAPL', 'CVX', 'KO', 'JNJ', 'BAC', 'NKE']
            #We calculate the covariance matrix
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

    # We create a new function to calculate the expected return for the whole portfolio
    def create_pert(self, expected_returns_df_filtered, df_two, df_four, df_six):
        self.values_two_stock = expected_returns_df_filtered[:2]
        self.values_four_stock = expected_returns_df_filtered[:4]
        self.values_six_stock = expected_returns_df_filtered

        # We create the stock vectors and transpose them
        self.two_stock_vector = np.array(self.values_two_stock).T[0]
        self.four_stock_vector = np.array(self.values_four_stock).T[0]
        self.six_stock_vector = np.array(self.values_six_stock).T[0]

        # We create the z-vectors and transpose them
        self.z_vector_two = np.array(df_two).T[0]
        self.z_vector_four = np.array(df_four).T[0]
        self.z_vector_six = np.array(df_six).T[0]
        
        # We calculate the portfolio expected return for each set of stocks
        self.portfolio_expected_return_two = np.dot(self.two_stock_vector, self.z_vector_two)
        self.portfolio_expected_return_four = np.dot(self.four_stock_vector, self.z_vector_four)
        self.portfolio_expected_return_six = np.dot(self.six_stock_vector, self.z_vector_six)

    # We define a function to calculate the portfolio variance for each set
    def calculate_portfolio_variances(self):
        self.portfolio_variance_two = np.dot(self.z_vector_two.T, np.dot(self.calculate_covariance("two"), self.z_vector_two))
        self.portfolio_variance_four = np.dot(self.z_vector_four.T, np.dot(self.calculate_covariance("four"), self.z_vector_four))
        self.portfolio_variance_six = np.dot(self.z_vector_six.T, np.dot(self.calculate_covariance("six"), self.z_vector_six))

    # We define a function to calculate the portfolio standard deviation for each set
    def calculate_portfolio_std_devs(self):
        self.portfolio_std_dev_two = np.sqrt(self.portfolio_variance_two)
        self.portfolio_std_dev_four = np.sqrt(self.portfolio_variance_four)
        self.portfolio_std_dev_six = np.sqrt(self.portfolio_variance_six)

    def plot_two_stock(self):
        # We create an array of weights
        weights = np.arange(0, 1.0, 0.01)

        # We initialize lists to store the portfolio variances, expected returns and the volatility
        portfolio_variances = []
        portfolio_expected_returns = []
        portfolio_volatilities = []

        # We loop over the weights
        for w in weights:
            weight_vector = np.array([w, 1 - w])

            # We calculate the portfolio variance
            portfolio_variance = np.dot(weight_vector.T, np.dot(self.calculate_covariance("two"), weight_vector))
            portfolio_variances.append(portfolio_variance)

            # We calculate the portfolio volatility
            portfolio_volatility = np.sqrt(portfolio_variance)
            portfolio_volatilities.append(portfolio_volatility)

            # We calculate the portfolio expected return
            portfolio_expected_return = np.dot(weight_vector, self.two_stock_vector)
            portfolio_expected_returns.append(portfolio_expected_return)

        # We convert the lists of volatility and expected returns to numpy arrays
        portfolio_volatilities = np.array(portfolio_volatilities)
        portfolio_expected_returns = np.array(portfolio_expected_returns)

        # We are adding our pre-calculated expected return and variance (red-dot)
        precalculated_std_dev_two = self.portfolio_std_dev_two
        precalculated_expected_return_two = self.portfolio_expected_return_two

        # We plot the pre-calculated expected return and volatility and the new calculated expected return and volatility
        plt.plot(portfolio_volatilities, portfolio_expected_returns, label='Portfolio')
        plt.plot(precalculated_std_dev_two, precalculated_expected_return_two, 'ro', label='Two stocks')
        plt.xlabel('Portfolio volatility (standard deviation)')
        plt.ylabel('Portfolio Expected Return')
        plt.title('Portfolio Variance vs Expected Return')
        plt.legend()
        plt.show()

    def plot_four_stock(self):
        # We create an array of weights
        weights_four = np.arange(0, 3.0, 0.01)

        # We initialize lists to store the portfolio variances, expected returns and the volatility
        portfolio_variances = []
        portfolio_expected_returns = []
        portfolio_volatilities = []

        # We loop over the weights
        for w in weights_four:
            # We create the weight vector, so each stock is equally weighted
            w1 = w
            w2 = (1 - w) / 3
            w3 = (1 - w) / 3
            w4 = (1 - w) / 3
        
            weight_vector_four = np.array([w1, w2, w3, w4])

            # We calculate the portfolio variance
            portfolio_variance = np.dot(weight_vector_four.T, np.dot(self.calculate_covariance("four"), weight_vector_four))
            portfolio_variances.append(portfolio_variance)

            # We calculate the portfolio volatility
            portfolio_volatility = np.sqrt(portfolio_variance)
            portfolio_volatilities.append(portfolio_volatility)

            # We calculate the portfolio expected return
            portfolio_expected_return = np.dot(weight_vector_four, self.four_stock_vector)
            portfolio_expected_returns.append(portfolio_expected_return)

        # We convert the lists of volatility and expected returns to numpy arrays
        portfolio_volatilities = np.array(portfolio_volatilities)
        portfolio_expected_returns = np.array(portfolio_expected_returns)

        # We add our pre-calculated expected return and volatility
        precalculated_std_dev_four = self.portfolio_std_dev_four
        precalculated_expected_return_four = self.portfolio_expected_return_four
        plt.plot(portfolio_volatilities, portfolio_expected_returns, label='Portfolio')
        plt.plot(precalculated_std_dev_four, precalculated_expected_return_four, 'ro', label='Four stocks')
        plt.xlabel('Portfolio volatility (standard deviation)')
        plt.ylabel('Portfolio Expected Return')
        plt.title('Portfolio Variance vs Expected Return')
        plt.legend()
        plt.show()

    def plot_six_stock(self):
        # We create an array of weights
        weights_six = np.arange(0, 5.0, 0.01)

        # We initialize lists to store the portfolio variances, expected returns and the volatility
        portfolio_variances = []
        portfolio_expected_returns = []
        portfolio_volatilities = []

        # We loop over the weights
        for w in weights_six:
            # We create the weight vector so each stock is equally weighted
            w1 = w
            w2 = (1 - w) / 5
            w3 = (1 - w) / 5
            w4 = (1 - w) / 5
            w5 = (1 - w) / 5
            w6 = (1 - w) / 5

            weight_vector_six = np.array([w1, w2, w3, w4, w5, w6])

            # We calculate the portfolio variance
            portfolio_variance = np.dot(weight_vector_six.T, np.dot(self.calculate_covariance("six"), weight_vector_six))
            portfolio_variances.append(portfolio_variance)

            # We calculate the portfolio volatility
            portfolio_volatility = np.sqrt(portfolio_variance)
            portfolio_volatilities.append(portfolio_volatility)

            # We calculate the portfolio expected return
            portfolio_expected_return = np.dot(weight_vector_six, self.six_stock_vector)
            portfolio_expected_returns.append(portfolio_expected_return)

        # We convert the lists of volatility and expected return to numpy arrays
        portfolio_volatilities = np.array(portfolio_volatilities)
        portfolio_expected_returns = np.array(portfolio_expected_returns)

        # We add our pre-calculated expected return and volatility
        precalculated_std_dev_six = self.portfolio_std_dev_six
        precalculated_expected_return_six = self.portfolio_expected_return_six
        plt.plot(portfolio_volatilities, portfolio_expected_returns, label='Portfolio')
        plt.plot(precalculated_std_dev_six, precalculated_expected_return_six, 'ro', label='Six stocks')
        plt.xlabel('Portfolio volatility (standard deviation)')
        plt.ylabel('Portfolio Expected Return')
        plt.title('Portfolio Variance vs Expected Return')
        plt.legend()
        plt.show()