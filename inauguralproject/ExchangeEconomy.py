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
        par.w1B = 0.2
        par.w2B = 0.7

    #Define utility functions: 
    def utility_A(self,x1A,x2A):
        return(x1A**self.par.alpha)*x2A**(1-self.par.alpha)
    
    def utility_B(self,x1B,x2B):
        return(x1B**(self.par.beta)*x2B**(1-self.par.beta))

    #Define demand functions:
    def demand_A(self,p1,p2):
        par = self.par
        x1A = par.alpha*((p1*par.w1A+p2*par.w2A)/p1)
        x2A = (1-par.alpha)*((p1*par.w1A+p2*par.w2A)/p2)
        return x1A, x2A

    def demand_B(self,p1,p2):
        par = self.par
        x1B = par.beta*((p1*(1-par.w1A)+p2*(1-par.w2A))/p1)
        x2B = (1-par.beta)*((p1*(1-par.w1A)+p2*(1-par.w2A))/p2)
        return x1B, x2B

    #Finding market clearing:
    def check_market_clearing(self,p1):

        par = self.par

        x1A,x2A = self.demand_A(p1, 1)
        x1B,x2B = self.demand_B(p1, 1)

        eps1 = x1A-par.w1A + x1B-(1-par.w1A)
        eps2 = x2A-par.w2A + x2B-(1-par.w2A)

        return eps1,eps2