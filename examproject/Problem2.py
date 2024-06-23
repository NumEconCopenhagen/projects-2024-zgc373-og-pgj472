import numpy as np
from types import SimpleNamespace
import matplotlib.pyplot as plt

class CareerChoiceModelClass():
    def __init__(self):
        par = self.par = SimpleNamespace()
        par.J = 3
        par.N = 10
        par.K = 10000
        par.sigma = 2
        par.v = np.array([1, 2, 3])
        par.v_j = np.random.choice([1, 2, 3], par.J)

    def simulation_q1(self):
        #We create two arrays to store our results
        expected_utilities = np.zeros(self.par.J)
        average_realized_utilities = np.zeros(self.par.J)

        #Next we loop over each career choice to simulate and calculate utilities, where we start by simulating epsilon for each career choice:
        for j in range(self.par.J):
            epsilons = np.random.normal(0, self.par.sigma**2, self.par.K)
        
            #From the epsilons found above, we can calculate the utilities using the formula given:
            utilities = self.par.v[j] + epsilons
            
            #We use np.mean to calculate the mean on the simulated utilities for each career choice, because by the Law of Large Numbers the average of the simulated utilities will converge to the expected utility:
            expected_utilities[j] = np.mean(utilities)
            
            #We calculate the average realized utility which theoretically should be equal to the simulated expected utility:
            average_realized_utilities[j] = self.par.v[j] + np.mean(epsilons)

        for j in range(self.par.J):
            print(f"Career choice {j+1}: Expected Utility = {expected_utilities[j]}, Average Realized Utility = {average_realized_utilities[j]}")


    def simulation_q2(self):
        # Initialize storage
        chosen_careers = np.zeros((self.par.N, self.par.J, self.par.K))
        expected_utilities = np.zeros((self.par.N, self.par.K))
        realized_utilities = np.zeros((self.par.N, self.par.K))

        # Simulation
        for k in range(self.par.K):
            for i in range(self.par.N):
                Fi = i + 1  # Number of friends for graduate i
                prior_expected_utilities = np.zeros(self.par.J)
                for j in range(self.par.J):
                    friends_noise = np.random.normal(0, self.par.sigma**2, Fi)
                    prior_expected_utilities[j] = np.mean(self.par.v_j[j] + friends_noise)
                personal_noise = np.random.normal(0, self.par.sigma**2, self.par.J)
                #Next we do the step that each person i chooses the career track with the highest expected utility. We use the np.argmax:
                chosen_career = np.argmax(prior_expected_utilities)
                #First we store the chosen career $j^k*_i$ 
                chosen_careers[i, chosen_career, k] = 1
                #Next we store the prior expectation of the value of their chosen career:
                expected_utilities[i, k] = prior_expected_utilities[chosen_career]
                #Lastly we store the realized value of their chosen career track:
                realized_utilities[i, k] = self.par.v_j[chosen_career] + personal_noise[chosen_career]

        # Calculate averages for plotting:
        career_shares = np.mean(chosen_careers, axis=2)
        average_expected_utilities = np.mean(expected_utilities, axis=1)
        average_realized_utilities = np.mean(realized_utilities, axis=1)

        # Visualization
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        for j in range(self.par.J):
            axs[0].plot(range(1, self.par.N+1), career_shares[:, j], label=f'Career {j+1}')
        axs[0].set_title('Share of Graduates Choosing Each Career')
        axs[0].set_xlabel('Graduate Type')
        axs[0].set_ylabel('Share')
        axs[0].legend()

        axs[1].plot(range(1, self.par.N+1), average_expected_utilities, label='Average Expected Utility')
        axs[1].set_title('Average Subjective Expected Utility')
        axs[1].set_xlabel('Graduate Type')
        axs[1].set_ylabel('Utility')
        axs[1].legend()

        axs[2].plot(range(1, self.par.N+1), average_realized_utilities, label='Average Realized Utility')
        axs[2].set_title('Average Ex Post Realized Utility')
        axs[2].set_xlabel('Graduate Type')
        axs[2].set_ylabel('Utility')
        axs[2].legend()

        plt.tight_layout()
        plt.show()

