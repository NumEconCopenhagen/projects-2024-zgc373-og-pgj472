# imports for the model project:
from scipy import optimize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
import ipywidgets as widgets
import sympy as sm
from scipy.optimize import fsolve

# We start by creating a class called Parameters which will be used to store the parameters of the model
class Parameters:
    def __init__(self, a, b, MC):
        self.a = a
        self.b = b
        self.MC = MC

# We start by creating a class called cournot which will be used to create our functions
class cournot:

    def __init__(self, a, b, MC):
        self.par = Parameters(a, b, MC)

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
    

# We define a class called CournotPlotter which will be used to create the interactive plot
class CournotPlotter:
    def __init__(self, cournot_model):
        self.model = cournot_model
        self.a_slider = widgets.FloatSlider(min=10, max=50, step=1, value=30, description='a:')
        self.b_slider = widgets.FloatSlider(min=1, max=5, step=0.1, value=3, description='b:')
        self.MC_slider = widgets.FloatSlider(min=1, max=25, step=0.1, value=13, description='MC:')
        widgets.interact(self.update_plot, a=self.a_slider, b=self.b_slider, MC=self.MC_slider)


    def update_plot(self, a, b, MC):
        self.model.par.a = a
        self.model.par.b = b
        self.model.par.MC = MC

        q = np.linspace(0, 10, 100)
        br1 = [self.model.br1(qi) for qi in q]
        br2 = [self.model.br2(qi) for qi in q]
        eq_q1 = self.model.equilibrium_quantity()
        eq_q2 = self.model.equilibrium_quantity()

        # Calculate the price in equilibrium
        eq_Q = eq_q1 + eq_q2
        eq_p = a - b * eq_Q

        # Calculate the profits in equilibrium
        eq_profit1 = eq_p * eq_q1 - MC * eq_q1
        eq_profit2 = eq_p * eq_q2 - MC * eq_q2

        plt.figure(figsize=(8, 8))
        plt.plot(q, br1, label='Best Response Firm 1')
        plt.plot(br2, q, label='Best Response Firm 2')
        plt.scatter(eq_q1, eq_q2, color='red', label='Equilibrium quantity')
        plt.xlabel('Quantity produced by Firm 1 (q1)')
        plt.ylabel('Quantity produced by Firm 2 (q2)')
        plt.legend()
        plt.show()

        print("Quantity produced by firm 1 in equilibrium:", eq_q1)
        print("Quantity produced by firm 2 in equilibrium:", eq_q2)
        print("Price in equilibrium:", eq_p)  
        print("Profit for firm 1 in equilibrium:", eq_profit1)  
        print("Profit for firm 2 in equilibrium:", eq_profit2) 


class Monopoly:

    def __init__(self, a, b, MC):
        self.par = Parameters(a, b, MC)

    # We define the cost function:
    def cost_function(self, Q):
        return self.par.MC * Q

    # We define the profit function for the monopoly:
    def profit_function(self, Q):
        p = self.par.a - self.par.b * Q
        return p * Q - self.cost_function(Q)

    # We define the first order condition for the monopoly:
    def foc(self, Q):
        return self.par.a - 2 * self.par.b * Q - self.par.MC

    # We define the monopoly quantity:
    def monopoly_quantity(self):
        return (self.par.a - self.par.MC) / (2 * self.par.b)

    # We define the monopoly price:
    def monopoly_price(self):
        Q = self.monopoly_quantity()
        return self.par.a - self.par.b * Q

    # We define the monopoly profit:
    def monopoly_profit(self):
        Q = self.monopoly_quantity()
        return self.profit_function(Q)
