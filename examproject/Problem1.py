#We have used the Copilot as permitted as a tool to adjust and correct our code when solving the problem. 
#We write in the code below the parts where we have mainly copied the code from the Copilot.

#Us writing the code:
from types import SimpleNamespace
from scipy import optimize
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
import numpy as np

class ProductionEconomyClass:
    #We define parameters:
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

        # We create the range given for p1 and p2 respectively:       
        par.p1_range = np.linspace(0.1, 2.0, 10)
        par.p2_range = np.linspace(0.1, 2.0, 10)

        # We create a dataframe to store the results:
        sol = self.sol = SimpleNamespace()

        # We set the wage as numeraire:
        sol.w = 1

    #For the following functions we have used the structure from the lecture "Production Economy" and adjust to this problem:
    def optimal_labor_firms(self, pj):
        """ Calculate optimal labor for the firms for a given price """
        par = self.par
        return (pj * par.A * par.gamma / par.w) ** (1 / (1 - par.gamma))

    def optimal_output_firms(self, l_star):
        """ Calculate optimal output for the firms for a given labor """
        par = self.par
        return par.A * l_star ** par.gamma
    
    def implied_profits_firms(self, pj, l_star):
        """ Calculate implied profits for the firms for a given price and labor """
        par = self.par
        return (1 - par.gamma) / par.gamma * par.w * l_star

    def optimal_consumption_cs(self, l, p1, p2, pi1, pi2):
        """ Calculate optimal consumption for the consumers given labor supply """
        par = self.par
        budget = par.w * l + par.T + pi1 + pi2
        c1 = par.alpha * budget / p1
        c2 = (1 - par.alpha) * budget / p2 + par.tau
        return c1, c2
    
    def utility(self, l, p1, p2):
        """ Utility function for the consumers """
        par = self.par
        #We store the optimal values from the functions above:
        l1_star = self.optimal_labor_firms(p1)
        l2_star = self.optimal_labor_firms(p2)
        pi1_star = self.implied_profits_firms(p1, l1_star)
        pi2_star = self.implied_profits_firms(p2, l2_star)
        c1, c2 = self.optimal_consumption_cs(l, p1, p2, pi1_star, pi2_star)
        return np.log(c1 ** par.alpha * c2 ** (1 - par.alpha)) - par.nu * l ** (1 + par.epsilon) / (1 + par.epsilon)

    def find_optimal_labor(self, p1, p2):
        """ Find optimal labor supply by maximizing the utility function for the consumers """
        sol = self.sol
        #We define an objective function to minimize:
        def obj(l):
            return -self.utility(l, p1, p2)

        res = minimize_scalar(obj, bounds=(0, 1), method='bounded')
        
        if not res.success:
            raise RuntimeError(f"Optimization failed: {res.message}")
        
        #We save results
        sol.l_star = res.x
        sol.utility_star = -res.fun
        return sol.l_star
        
    # We evaluate the equilibrium and check the market clearing conditions:
    #For the next part we used copilot to get it to work: 
    def evaluate_equilibrium(self, p1, p2):
        """ Evaluate equilibrium and check market clearing conditions """

        l_star = self.find_optimal_labor(p1, p2)
        l1_star = self.optimal_labor_firms(p1)
        l2_star = self.optimal_labor_firms(p2)
        y1_star = self.optimal_output_firms(l1_star)
        y2_star = self.optimal_output_firms(l2_star)
        pi1_star = self.implied_profits_firms(p1, l1_star)
        pi2_star = self.implied_profits_firms(p2, l2_star)
        c1_star, c2_star = self.optimal_consumption_cs(l_star, p1, p2, pi1_star, pi2_star)

        labor_market_clear = np.isclose(l_star, l1_star + l2_star)
        good1_market_clear = np.isclose(c1_star, y1_star)
        good2_market_clear = np.isclose(c2_star, y2_star)

        return {
            'p1': p1,
            'p2': p2,
            'l_star': l_star,
            'l1_star': l1_star,
            'l2_star': l2_star,
            'y1_star': y1_star,
            'y2_star': y2_star,
            'c1_star': c1_star,
            'c2_star': c2_star,
            'labor_market_clear': labor_market_clear,
            'good1_market_clear': good1_market_clear,
            'good2_market_clear': good2_market_clear
        }

    #Us writing the code:
    def check_market_clearing(self):
        """ Check market clearing conditions for all combinations of p1 and p2 by looping through the ranges given """
        par = self.par
        results = []
        market_clearing_found = False 

        #We loop through the ranges given for p1 and p2:
        for p1 in par.p1_range:
            for p2 in par.p2_range:
                result = self.evaluate_equilibrium(p1, p2)
                results.append(result)
                #We check if any market clearing condition is met:
                if result['labor_market_clear'] or result['good1_market_clear'] or result['good2_market_clear']:
                    market_clearing_found = True
                    print(f"p1: {result['p1']}, p2: {result['p2']}, "
                        f"Labor Market Clearing: {result['labor_market_clear']}, "
                        f"Good 1 Market Clearing: {result['good1_market_clear']}, "
                        f"Good 2 Market Clearing: {result['good2_market_clear']}")

        if not market_clearing_found:
            print("No market clearing")

        return results

    #Next  part is made with help from Copilot:
    #We calculate the equilibrium prices:
    def find_equilibrium_prices(self):
        #We define the initial guesses for the prices:
        initial_guess = [1.0, 1.0]

        # We Define the objective function to minimize
        def objective(prices):
            p1, p2 = prices
            result = self.evaluate_equilibrium(p1, p2)
            labor_market_clearing = abs(result['l_star'] - (result['l1_star'] + result['l2_star']))
            good1_market_clearing = abs(result['c1_star'] - result['y1_star'])
            # No need to check good2_market_imbalance due to Walras' law
            return labor_market_clearing + good1_market_clearing

        # We use an optimization algorithm to find equilibrium prices
        result = optimize.minimize(objective, initial_guess, method='Nelder-Mead')

        if result.success:
            equilibrium_prices = result.x
            print(f"Equilibrium prices found: p1 = {equilibrium_prices[0]}, p2 = {equilibrium_prices[1]}")
            return equilibrium_prices
        else:
            raise RuntimeError("Failed to find equilibrium prices")

    #Us writing the code:
    # We calculate the social welfare function
    def calculate_swf(self, p1, p2):
        equilibrium = self.evaluate_equilibrium(p1, p2)
        y2_star = equilibrium['y2_star']
        U = self.sol.utility_star
        SWF = U - self.par.kappa * y2_star
        return SWF

    # We optimize the social welfare function (used copilot to adjust the code):
    def swf_optimization_objective_with_T(self, params):
        tau, T = params 
        self.par.tau = tau 
        self.par.T = T
        for p1 in self.par.p1_range:
            for p2 in self.par.p2_range:
                SWF = self.calculate_swf(p1, p2)
                return -SWF 

    #Us writing the code:
    # We find the optimal value og tau, T and SWF
    def optimize_tau_and_T(self):
        # We set initial guesses for tau and T
        initial_guess = [0.01, 0.01]
        result = minimize(self.swf_optimization_objective_with_T, initial_guess, method='Nelder-Mead')
        optimal_tau, optimal_T = result.x
        optimal_SWF = -result.fun
        print(f"Optimal tau: {optimal_tau}, Optimal T: {optimal_T}, Maximized SWF: {optimal_SWF}")


