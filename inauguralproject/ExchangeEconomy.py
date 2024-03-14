from types import SimpleNamespace
import numpy as np

class ExchangeEconomyClass:
    #Define parameters:
    def __init__(self):

        par = self.par = SimpleNamespace()

        # a. preferences
        par.alpha = 1/3
        par.beta = 2/3

        # b. endowments
        par.w1A = 0.8
        par.w2A = 0.3

    #Define utility functions: 
    def utility_A(self,x1A,x2A):
        return(x1A**self.par.alpha)*x2A**(1-self.par.alpha)
    
    def utility_B(self,x1B,x2B):
        return(x1B**(self.par.beta)*x2B**(1-self.par.beta))

    #Define demand functions:
    def demand_A(self,p1):
        return(self.alpha((p1*w1A+p2*w2A)/p1))

    def demand_B(self,p1):
        return(self.beta((p1*w1B+p2*w2B)/p1))

    #Finding market clearing:
    def check_market_clearing(self,p1):

        par = self.par

        x1A,x2A = self.demand_A(p1)
        x1B,x2B = self.demand_B(p1)

        eps1 = x1A-par.w1A + x1B-(1-par.w1A)
        eps2 = x2A-par.w2A + x2B-(1-par.w2A)

        return eps1,eps2