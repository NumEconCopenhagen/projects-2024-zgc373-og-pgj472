# imports for the model project:
from scipy import optimize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
import ipywidgets as widgets
import sympy as sm
from scipy.optimize import fsolve

# We start by creating a class called parameters which will be used to store the parameters of the model
class parameters:
    def __init__(self, a, b, MC):
        self.a = a
        self.b = b
        self.MC = MC

# We start by creating a class called cournot which will be used to create our functions
class duopoly_model:

    def __init__(self, a, b, MC):
        self.par = parameters(a, b, MC)
        self.a_slider = widgets.FloatSlider(min=10, max=50, step=1, value=30, description='a:')
        self.b_slider = widgets.FloatSlider(min=1, max=5, step=0.1, value=3, description='b:')
        self.MC_slider = widgets.FloatSlider(min=1, max=25, step=0.1, value=13, description='MC:')
    
    # Add a new method for the interactive plot    
    def interactive_plot(self):    
        widgets.interact(self.update_plot, a=self.a_slider, b=self.b_slider, MC=self.MC_slider)

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
    def equilibrium_quantity_firm1(self):
        return (self.par.a - self.par.MC) / (3 * self.par.b)

    def equilibrium_quantity_firm2(self):
        return (self.par.a - self.par.MC) / (3 * self.par.b)

    # We define the total quantity produced in equilibrium:
    def total_equilibrium_quantity(self):
        return self.equilibrium_quantity_firm1() + self.equilibrium_quantity_firm2()
    
    # Calculate and return the equilibrium price
    def equilibrium_price(self):       
        return self.par.a - self.par.b * self.total_equilibrium_quantity()
    
    def update_plot(self, a, b, MC):
        self.par.a = a
        self.par.b = b
        self.par.MC = MC

        q = np.linspace(0, 10, 100)
        br1 = [self.br1(qi) for qi in q]
        br2 = [self.br2(qi) for qi in q]
        eq_q1d = self.equilibrium_quantity_firm1()
        eq_q2d = self.equilibrium_quantity_firm2()

        # Calculate the price in equilibrium
        eq_Qd = self.total_equilibrium_quantity()
        eq_pd = a - b * eq_Qd

        # Calculate the profits in equilibrium
        eqd_profit1 = eq_pd * eq_q1d - MC * eq_q1d
        eqd_profit2 = eq_pd * eq_q2d - MC * eq_q2d

        plt.figure(figsize=(8, 8))
        plt.plot(q, br1, label='Best Response Firm 1')
        plt.plot(br2, q, label='Best Response Firm 2')
        plt.scatter(eq_q1d, eq_q2d, color='red', label='Cournot-Nash Equilibrium')
        plt.xlabel('Quantity produced by Firm 1 (q1)')
        plt.ylabel('Quantity produced by Firm 2 (q2)')
        plt.legend()
        plt.show()

        print("Quantity produced by firm 1 in equilibrium:", self.equilibrium_quantity_firm1())
        print("Quantity produced by firm 2 in equilibrium:", self.equilibrium_quantity_firm2())
        print("Total quantity produced in equilibrium:", self.total_equilibrium_quantity())
        print("Price in equilibrium:", eq_pd)  # Print the price in equilibrium
        print("Profit for firm 1 in equilibrium:", eqd_profit1)  # Print the profit for firm 1 in equilibrium  
        print("Profit for firm 2 in equilibrium:", eqd_profit2)  # Print the profit for firm 2 in equilibrium


class monopoly_model:

    def __init__(self, a, b, MC):
        self.par = parameters(a, b, MC)
        self.par = parameters(a, b, MC)
        self.a_slider = widgets.FloatSlider(min=10, max=50, step=1, value=30, description='a:')
        self.b_slider = widgets.FloatSlider(min=1, max=5, step=0.1, value=3, description='b:')
        self.MC_slider = widgets.FloatSlider(min=1, max=25, step=0.1, value=13, description='MC:')
    
    # Add a new method for the interactive plot for monopoly
    def interactive_plot_monopoly(self): 
        widgets.interact(self.calculate_values, a=self.a_slider, b=self.b_slider, MC=self.MC_slider)

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
    
    def calculate_values(self, a, b, MC):
        self.par.a = a
        self.par.b = b
        self.par.MC = MC

        # Calculate the monopoly quantity, price, and profit
        monopoly_q = self.monopoly_quantity()
        monopoly_p = self.monopoly_price()
        monopoly_profit = self.monopoly_profit()

        print("Quantity produced under monopoly is:", monopoly_q)
        print("Price in monopoly:", monopoly_p)
        print("Profit for monopoly:", monopoly_profit)


class plotcomparison:

    def __init__(self, monopoly_model, duopoly_model):
        self.monopoly_model = monopoly_model
        self.duopoly_model = duopoly_model

    def calculate_values(self, a, b, MC):
        self.monopoly_model.par.a = a
        self.monopoly_model.par.b = b
        self.monopoly_model.par.MC = MC
        self.duopoly_model.par.a = a
        self.duopoly_model.par.b = b
        self.duopoly_model.par.MC = MC

    def plot(self, a, b, MC):
        self.calculate_values(a, b, MC)

        # Calculate the monopoly quantity, price, and profit
        monopoly_q = self.monopoly_model.monopoly_quantity()
        monopoly_p = self.monopoly_model.monopoly_price()

        # Calculate the duopoly quantities, price, and profits
        duopoly_q1 = self.duopoly_model.equilibrium_quantity_firm1()
        duopoly_q2 = self.duopoly_model.equilibrium_quantity_firm2()
        duopoly_p = self.duopoly_model.equilibrium_price()

        # Plot the quantities and prices
        plt.figure(figsize=(8, 8))
        plt.plot([monopoly_q, duopoly_q1 + duopoly_q2], [monopoly_p, duopoly_p], marker='o')
        plt.xlabel('Quantity')
        plt.ylabel('Price')
        plt.title('Price vs Quantity for Monopoly and Duopoly')
        plt.legend(['Monopoly', 'Duopoly'])
        plt.grid(True)
        plt.show()

