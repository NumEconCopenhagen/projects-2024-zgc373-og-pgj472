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
        self.a_slider = widgets.FloatSlider(min=50, max=150, step=1, value=100, description='a:')
        self.b_slider = widgets.FloatSlider(min=5, max=25, step=0.1, value=16, description='b:')
        self.MC_slider = widgets.FloatSlider(min=0.5, max=5, step=0.1, value=4, description='MC:')
    
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
    


class duopoly_model_extension(duopoly_model):
    def __init__(self):
        super().__init__(100, 16, 0)
    
    def cost_function(self, qi):
        return 0

    def profit_function1(self, q1, q2):
        Q = q1 + q2
        p = self.par.a - self.par.b * Q
        return p * q1

    def profit_function2(self, q1, q2):
        Q = q1 + q2
        p = self.par.a - self.par.b * Q
        return p * q2

    def foc1(self, q1, q2):
        return self.par.a - 2 * self.par.b * q1 - self.par.b * q2

    def foc2(self, q1, q2):
        return self.par.a - self.par.b * q1 - 2 * self.par.b * q2

    def br1(self, q2):
        return self.par.a / (2 * self.par.b) - q2 / 2

    def br2(self, q1):
        return self.par.a / (2 * self.par.b) - q1 / 2

    def equilibrium_quantity_firm1(self):
        return self.par.a / (3 * self.par.b)

    def equilibrium_quantity_firm2(self):
        return self.par.a / (3 * self.par.b)

    def update_plot(self):
        super().update_plot(100, 16, 0)

    def print_results(self, q1, q2):
        print("Quantity produced by firm 1 in equilibrium:", self.equilibrium_quantity_firm1())
        print("Quantity produced by firm 2 in equilibrium:", self.equilibrium_quantity_firm2())
        print("Total quantity produced in equilibrium:", self.total_equilibrium_quantity())
        print("Price in equilibrium:", self.equilibrium_price())  
        print("Profit for firm 1 in equilibrium:", self.profit_function1(q1, q2))  
        print("Profit for firm 2 in equilibrium:", self.profit_function2(q1, q2))  



class monopoly_model_extension(duopoly_model):
    def __init__(self):
        super().__init__(100, 16, 0)
    
    def cost_function(self, qi):
        return 0

    def profit_function(self, Q_ex):
        p = self.par.a - self.par.b * Q_ex
        return p * Q_ex
    
    def profit_per_firm(self, Q_ex):
        return self.profit_function(Q_ex) / 2

    def foc(self, Q_ex):
        return self.par.a - 2 * self.par.b * Q_ex

    def equilibrium_quantity_monopoly(self):
        return self.par.a / (2 * self.par.b)
    
    def quantity_per_firm(self):
        return self.equilibrium_quantity_monopoly() / 2
    
    def price_monopoly(self):
        return self.par.a - self.par.b * self.equilibrium_quantity_monopoly()

    def update_plot(self):
        super().update_plot(100, 16, 0)

    def print_results(self, Q_ex):
        print("Quantity produced per firm:", self.quantity_per_firm())
        print("Total quantity produced::", self.equilibrium_quantity_monopoly())
        print("Price in equilibrium:", self.price_monopoly())  
        print("Profit per firm:", self.profit_per_firm(Q_ex))  








#class monopoly_model_extension:

    def __init__(self, a, b, MC):
        self.par = parameters(a, b, MC)
        self.par = parameters(a, b, MC)
        self.a_slider = widgets.FloatSlider(min=50, max=150, step=0.1, value=100, description='a:')
        self.b_slider = widgets.FloatSlider(min=5, max=25, step=0.1, value=16, description='b:')
        self.MC_slider = widgets.FloatSlider(min=0.5, max=5, step=0.1, value=4, description='MC:')
    
    # Add a new method for the interactive plot for monopoly
    def interactive_plot_monopoly(self): 
        widgets.interact(self.calculate_values, a=self.a_slider, b=self.b_slider, MC=self.MC_slider)

    # We define the cost function:
    def cost_function(self, Q):
        return 0

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

