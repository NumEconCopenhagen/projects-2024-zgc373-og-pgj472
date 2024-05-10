# imports for the model project:
from scipy import optimize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
import ipywidgets as widgets
import sympy as sm
from scipy.optimize import fsolve
from scipy.optimize import root
from prettytable import PrettyTable

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
        self.sol = SimpleNamespace()
    
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
    
    def solve_numerical(self):
        def inv_br2(x):
            return -(x - (self.par.a - self.par.MC) / (2*self.par.b))*2

        def objective_function(x):
            return self.br1(x) - inv_br2(x)
        x0 = 3
        result = root(objective_function, x0)
        return result.x
    
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

    def solve_numerical_ex(self):
        def inv_br2(x):
            return -(x - self.par.a / (2*self.par.b))*2

        def objective_function(x):
            return self.br1(x) - inv_br2(x)
        x0 = 3
        result = root(objective_function, x0)
        return result.x
    
    def plot_duo(self):
        q = np.linspace(0, self.par.a / self.par.b, 100)  # range of quantities
        p_demand = self.par.a - self.par.b * q  # inverse demand function
        equilibrium_quantity = self.equilibrium_quantity_firm1() + self.equilibrium_quantity_firm2()  # equilibrium quantity
        equilibrium_price = self.par.a - self.par.b * equilibrium_quantity  # equilibrium price

        plt.figure(figsize=(8, 6))
        plt.plot(q, p_demand, label='Demand')
        plt.axvline(x=equilibrium_quantity, color='r', linestyle='-', label='Supply')
        plt.axhline(y=33.33, color='g', linestyle='--', label='Price at 33.33')  # horizontal line at price 33.33
        plt.scatter(equilibrium_quantity, equilibrium_price, color='b')  # intersection point
        plt.text(equilibrium_quantity + 0.20 * (self.par.a / self.par.b), equilibrium_price + 0.05 * self.par.a, 
                f'({equilibrium_quantity:.2f}, {equilibrium_price:.2f})', horizontalalignment='right')  # text indicating the intersection point
        plt.xlabel('Quantity')
        plt.ylabel('Price')
        plt.title('Inverse Demand and Supply Curves')
        plt.legend()
        plt.show()

    def consumer_surplus_duo(self):
        return 0.5 * (self.par.a - 33.33) * (self.equilibrium_quantity_firm1()+self.equilibrium_quantity_firm2())

    def producer_surplus_duo(self):
        return 33.33 * (self.equilibrium_quantity_firm1()+self.equilibrium_quantity_firm2())

    def print_results(self, q1, q2):
        print("Quantity produced by firm 1 in equilibrium:", self.equilibrium_quantity_firm1())
        print("Quantity produced by firm 2 in equilibrium:", self.equilibrium_quantity_firm2())
        print("Total quantity produced in equilibrium:", self.total_equilibrium_quantity())
        print("Price in equilibrium:", self.equilibrium_price())  
        print("Profit for firm 1 in equilibrium:", self.profit_function1(q1, q2))  
        print("Profit for firm 2 in equilibrium:", self.profit_function2(q1, q2))  



class monopoly_model_extension(duopoly_model):
    def __init__(self):
        self.sol = SimpleNamespace()
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

    def solve_numerical_mon(self):
        
        x0 = 3
        result = optimize.minimize(lambda x: -self.profit_function(x), x0)
        
        self.sol = SimpleNamespace()
        self.sol.Q = result.x
        return result.x
    
    def plot_mon(self):
        q = np.linspace(0, self.par.a / self.par.b, 100)  # range of quantities
        p_demand = self.par.a - self.par.b * q  # inverse demand function
        equilibrium_quantity = self.equilibrium_quantity_monopoly()  # equilibrium quantity
        equilibrium_price = self.par.a - self.par.b * equilibrium_quantity  # equilibrium price

        plt.figure(figsize=(8, 6))
        plt.plot(q, p_demand, label='Demand')
        plt.axvline(x=equilibrium_quantity, color='r', linestyle='-', label='Supply')
        plt.axhline(y=50, color='g', linestyle='--', label='Price at 50')  # horizontal line at price 50
        plt.scatter(equilibrium_quantity, equilibrium_price, color='b')  # intersection point
        plt.text(equilibrium_quantity + 0.20 * (self.par.a / self.par.b), equilibrium_price + 0.05 * self.par.a, 
                f'({equilibrium_quantity:.2f}, {equilibrium_price:.2f})', horizontalalignment='right')  # text indicating the intersection point
        plt.xlabel('Quantity')
        plt.ylabel('Price')
        plt.title('Inverse Demand and Supply Curves')
        plt.legend()
        plt.show()

    def consumer_surplus_mon(self):
        return 0.5 * (self.par.a - 50) * self.equilibrium_quantity_monopoly()

    def producer_surplus_mon(self):
        return 50 * self.equilibrium_quantity_monopoly()

    def print_results(self):
        print("Quantity produced per firm:", self.quantity_per_firm())
        print("Total quantity produced::", self.equilibrium_quantity_monopoly())
        print("Price in equilibrium:", self.price_monopoly())  
        print("Profit per firm:", self.profit_per_firm(self.sol.Q))  

    def print_results_table(self, duopoly_model_extension, monopoly_model_extension):
        # Create a PrettyTable object
        table = PrettyTable()

        # Set the column names
        table.field_names = ["", "Duopoly Extension Model", "Monopoly Extension Model"]

        self.equilibrium_quantity_firm1() + self.equilibrium_quantity_firm2()

        # Add rows for each metric
        table.add_row(["Quantity Produced", f"{duopoly_model_extension.equilibrium_quantity_firm1() + duopoly_model_extension.equilibrium_quantity_firm2():.3f}", f"{monopoly_model_extension.equilibrium_quantity_monopoly():.3f}"])
        table.add_row(["Price", f"{duopoly_model_extension.equilibrium_price():.3f}", f"{monopoly_model_extension.price_monopoly():.3f}"])
        table.add_row(["Profit", f"{duopoly_model_extension.profit_function1(2.083, 2.083):.3f}", f"{monopoly_model_extension.profit_per_firm(3.125):.3f}"])
        table.add_row(["Consumer Surplus", f"{duopoly_model_extension.consumer_surplus_duo():.3f}", f"{monopoly_model_extension.consumer_surplus_mon():.3f}"])
        table.add_row(["Producer Surplus", f"{duopoly_model_extension.producer_surplus_duo():.3f}", f"{monopoly_model_extension.producer_surplus_mon():.3f}"])

        # Print the table
        print(table)