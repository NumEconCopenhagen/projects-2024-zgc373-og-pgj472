# imports for the model project:
from scipy import optimize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace

# We start by creating a class called cournot which will be used to create our functions
class cournot:

    # Define parameters:
    def __init__(self):

        par = self.par = SimpleNamespace()

        # parameters:
        par.a = 30
        par.b = 2
        par.MC = 5

    # We define the cost function:
    def cost_function(self, qi):
        return self.par.MC * qi

    # We define the profit function for firm 1:
    def profit_function1(self, q1, q2):
        Q = q1 + q2
        p = self.par.a - self.par.b * Q
        return p * q1 - self.cost_function(q1)

    # We define the profit function for firm 2:
    def profit_function2(self, q1, q2):
        Q = q1 + q2
        p = self.par.a - self.par.b * Q
        return p * q2 - self.cost_function(q2)

    # We define the first order condition for firm 1:
    def foc1(self, q1, q2):
        return self.par.a - 2 * self.par.b * q1 - self.par.b * q2 - self.par.MC

    # We define the first order condition for firm 2:
    def foc2(self, q1, q2):
        return self.par.a - self.par.b * q1 - 2 * self.par.b * q2 - self.par.MC

    # We define the best response function for firm 1:
    def br1(self, q2):
        return (self.par.a - self.par.MC) / (2 * self.par.b) - q2 / 2

    # We define the best response function for firm 2:
    def br2(self, q1):
        return (self.par.a - self.par.MC) / (2 * self.par.b) - q1 / 2
    
    # We define the Cournot-Nash equilibrium quantity for each firm:
    def equilibrium_quantity(self):
        return (self.par.a - self.par.MC) / (3 * self.par.b)

    # We define the total quantity produced in equilibrium:
    def total_equilibrium_quantity(self):
        return 2 * (self.par.a - self.par.MC) / (3 * self.par.b)
    


