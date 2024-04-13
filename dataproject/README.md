# Data analysis project

Our project is titled **"Free-lunch" of Diversification** and is about:
You may have heard of the saying “free lunch of diversification” by nobel prize winner Harry Markowitz within the field of portfolio theory. The idea of the "free-lunch" is that through diversification, investors have the opportunity to reduce risk without sacrificing expected return. 

This concept is illustrated by the Efficient Frontier, which shows the set of optimal portfolios that offer the highest expected return for a given level of risk, or the lowest level of risk for a given expected return.
According to his theory, across a mix of assets with uncorrelated or negatively correlated returns, investors can potentially achieve higher returns for a given level of risk, or lower risk for a given level of return. We have randomly chosen six different and well known companies that we believe could be relatively uncorrelated with each other. The companies are:
- Apple (AAPL) from the tech-industry 
- Chevron Corporation (CVX) from the energy- and oil-industry
- Coca-cola  (KO) in the beverage industry 
- Johnson & Johnson (JNJ) pharmaceutical-industry, 
- Bank of America (BAC) from the financial service industry
- Nike inc (NKE) textile industry.

However, it's essential to note that while diversification can mitigate certain types of risk, such as firm-specific or industry-specific risk, it cannot eliminate all forms of risk, such as systematic risk. Nonetheless, Markowitz's insights have had a profound impact on modern portfolio management and have influenced how investors think about constructing and managing their investment portfolios.

In this project we therefore are seeking to show this concept by recreating the Efficient Frontier for the companies chosen. We illustrate by showing the difference in expected return and volatility (standard deviation) for the case of only two stocks, then four stocks, and finally six stocks in a portfolio.

Our method and calculations are taken from chapter 11 “Optimal Portfolio Choice and the Capital Asset Pricing Model” in Corporate Finance by Jonathan Berk and Peter DeMarzo along with “Lecture Notes on Asset Pricing” by Peter Norman Sørensen (Department of Economics, UCPA). 

We have imported and used stock data for the closing price of the six companies in the period between January 1, 2017, to April 4, 2024. To recreate and test the theory we show three different portfolios consisting of two (AAPL and CVX), four (AAPL, CVX, KO and JNJ) and six (AAPL, CVX, KO, JNJ, BAC and NKE) stocks.


The **results** of the project can be seen from running [dataproject.ipynb](dataproject.ipynb).


**Dependencies:** Apart from a standard Anaconda Python 3 installation, the project requires the following installations:

``pip install matplotlib-venn``
``pip install yfinance``