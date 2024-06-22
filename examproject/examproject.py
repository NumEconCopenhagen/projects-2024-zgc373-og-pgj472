from types import SimpleNamespace
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy import optimize
from scipy.optimize import minimize
import pandas as pd
import numpy as np


class productioneconomy:
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
    def l_j_star(self, w, p_j, A, gamma):
        par = self.par
        return (p_j * par.A * par.gamma / par.w) ** (1 / (1 - par.gamma))
    
    def y_j_star(self, w, p_j, A, gamma):
        par = self.par
        return par.A * self.l_j_star(w, p_j, A, gamma) ** par.gamma

    #We define the implied profit function:
    def pi_j_star(self, w, p_j, A, gamma):
        par = self.par
        return (1 - par.gamma) / par.gamma * par.w * self.l_j_star(w, p_j, A, gamma)
    
    #We define the optimal consumption functions:
    def consumption(self, w, l, p1, p2, alpha, tau, T, A, gamma):
        par = self.par
        pi1 = self.pi_j_star(w, p1, A, gamma)
        pi2 = self.pi_j_star(w, p2, A, gamma)
        c1 = par.alpha * (par.w * l + par.T + pi1 + pi2) / p1
        c2 = (1-par.alpha) * (par.w * l + par.T + pi1 + pi2) / p2 + par.tau
        return c1, c2
    
    #We define the single consumer utility function:
    def utility(self, w, l, p1, p2):
        par = self.par
        #Including the given optimal consumption:
        c1, c2 = self.consumption(w, l, p1, p2, par.alpha, par.tau, par.T, par.A, par.gamma)
        
        #We then calculate the utility. Because we use the given optimal consumption functions we are implicitly satisfying the budget constraint.
        utility = np.log(c1 ** par.alpha * c2 ** (1 - par.alpha)) - par.nu * (l ** (1 + par.epsilon)) / (1 + par.epsilon)
        
        return utility

    #Before we can check for market clearing we need to find the optimal labor supply for each consumer. We use the given optimal behavior function:
    def find_optimal_labor(self, w, p1, p2):
        par = self.par
        result = minimize(lambda ell: -self.utility(w, ell, p1, p2), 1, bounds=[(0.1, None)])
        return result.x[0]

    #Now we can check for the market clearing price: 
    def check_market_clearing(self):
            for p1 in self.p1_range:
                for p2 in self.p2_range:





