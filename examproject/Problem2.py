import numpy as np
from types import SimpleNamespace

class CareerChoiceModelClass():
    def __init__(self):
        par = self.par = SimpleNamespace()
        par.J = 3
        par.N = 10
        par.K = 10000
        par.sigma = 2
        par.v = np.array([1, 2, 3])

    def simulate_and_calculate_utilities(self):
        # Initialize arrays to store results
        expected_utilities = np.zeros(self.par.J)
        average_realized_utilities = np.zeros(self.par.J)

        # Simulate and calculate utilities
        for j in range(self.par.J):
            # Simulate ε for each career choice
            epsilons = np.random.normal(0, self.par.sigma**2, self.par.K)
        
            # Calculate utilities for each draw
            utilities = self.par.v[j] + epsilons
            
            # Calculate expected utility
            expected_utilities[j] = np.mean(utilities)
            
            # Calculate average realized utility
            average_realized_utilities[j] = self.par.v[j] + np.mean(epsilons)

        # Print results
        for j in range(self.par.J):
            print(f"Career choice {j+1}: Expected Utility = {expected_utilities[j]}, Average Realized Utility = {average_realized_utilities[j]}")





