# imports for the model project:
from scipy import optimize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# We start by creating a class called cournot which will be used to create our functions
class cournot:

    # Define the parameters
    a = 5
    b = 1
    MC = 1

    # We define the cost function:
    def cost_function(self, qi, MC):
        return MC * qi

    # We define the profit function for firm 1:
    def profit_function1(self, q1, q2, a, b, MC):
        Q = q1 + q2
        p = a - b * Q
        return p * q1 - self.cost_function(q1, MC)

    # We define the profit function for firm 2:
    def profit_function2(self, q1, q2, a, b, MC):
        Q = q1 + q2
        p = a - b * Q
        return p * q2 - self.cost_function(q2, MC)

    # We define the first order condition for firm 1:
    def foc1(self, q1, q2, a, b, MC):
        return a - 2 * b * q1 - b * q2 - MC

    # We define the first order condition for firm 2:
    def foc2(self, q1, q2, a, b, MC):
        return a - b * q1 - 2 * b * q2 - MC

    # We define the best response function for firm 1:
    def br1(self, q2, a, b, MC):
        return (a - MC) / (2 * b) - q2 / 2

    # We define the best response function for firm 2:
    def br2(self, q1, a, b, MC):
        return (a - MC) / (2 * b) - q1 / 2
    
    # We define the Cournot-Nash equilibrium quantity for each firm:
    def equilibrium_quantity(self, a, b, MC):
        return (a - MC) / (3 * b)

    # We define the total quantity produced in equilibrium:
    def total_equilibrium_quantity(self, a, b, MC):
        return 2 * (a - MC) / (3 * b)
    
