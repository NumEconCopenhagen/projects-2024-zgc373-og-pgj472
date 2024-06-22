from types import SimpleNamespace
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy import optimize
from scipy.optimize import minimize
import pandas as pd
import numpy as np

class ProductionEconomyClass:
    #Defining parameters:
    def __init__(self):
        par = self.par = SimpleNamespace()
        
        # firms
        par.A = 1.0
        par.gamma = 0.5

        # households
        par.alpha = 0.3
        par.nu = 1.0
        par.epsilon = 2.0

        # government
        par.tau = 0.0
        par.T = 0.0

        # Question 1
        par.w = 1.0

        # Question 3
        par.kappa = 0.1


    #Creating the linspaces given: 
        self.p1_range = np.linspace(0.1, 2.0, 10)
        self.p2_range = np.linspace(0.1, 2.0, 10)


    #We start by defining all the functions given so we can use them later on: 
    #We define the optimal firm behaviour functions: 
    def l_j_star(self, p_j):
        par = self.par
        return (p_j * par.A * par.gamma / par.w) ** (1 / (1 - par.gamma))
    
    def y_j_star(self, p_j):
        par = self.par
        return par.A * self.l_j_star(p_j) ** par.gamma

    #We define the implied profit function:
    def pi_j_star(self, p_j):
        par = self.par
        return (1 - par.gamma) / par.gamma * par.w * self.l_j_star(p_j)
    
    #We define the optimal consumption functions:
    def consumption(self, l, p1, p2):
        par = self.par
        pi1 = self.pi_j_star(p1)
        pi2 = self.pi_j_star(p2)
        c1 = par.alpha * (par.w * l + par.T + pi1 + pi2) / p1
        c2 = (1-par.alpha) * (par.w * l + par.T + pi1 + pi2) / p2 + par.tau
        return c1, c2
    
    #We define the single consumer utility function:
    def utility(self, l, p1, p2):
        par = self.par
        #Including the given optimal consumption:
        c1, c2 = self.consumption(l, p1, p2)
        
        #We then calculate the utility. Because we use the given optimal consumption functions we are implicitly satisfying the budget constraint.
        utility = np.log(c1 ** par.alpha * c2 ** (1 - par.alpha)) - par.nu * (l ** (1 + par.epsilon)) / (1 + par.epsilon)
        
        return utility

    #Before we can check for market clearing we need to find the optimal labor supply for each consumer. We use the given optimal behavior function:
    def find_optimal_labor(self, p1, p2):
        result = minimize(lambda l: -self.utility(l, p1, p2), 1, bounds=[(0.1, None)])
        return result.x[0]
    
    #Now we can check for the market clearing price: 
    def check_market_clearing(self):
        market_clearing_prices = []
        for p1 in self.p1_range:
            for p2 in self.p2_range:
                optimal_labor = self.find_optimal_labor(p1, p2)
                c1_demand, c2_demand = self.consumption(optimal_labor, p1, p2)
                y1_supply = self.y_j_star(p1)
                y2_supply = self.y_j_star(p2)
                
                if np.isclose(c1_demand, y1_supply) and np.isclose(c2_demand, y2_supply):
                    market_clearing_prices.append((p1, p2))
        
        if market_clearing_prices:
            for p1, p2 in market_clearing_prices:
                print(f"Market clears for p1 = {p1:.2f}, p2 = {p2:.2f}")
        else:
            print("Market does not clear for any price.")





