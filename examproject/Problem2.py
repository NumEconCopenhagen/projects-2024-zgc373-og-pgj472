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
        #We create three arrays to store our results
        #chosen_careers is a 3D array.
        chosen_careers = np.zeros((self.par.N, self.par.J, self.par.K))
        #Whereas expected_utilities and realized_utilities are 2D arrays, because they only need to store the utility for the career that is actually chosen by each graduate in each simulation, so they don't need the self.par.J:
        expected_utilities = np.zeros((self.par.N, self.par.K))
        realized_utilities = np.zeros((self.par.N, self.par.K))

        #First we loop over the 10.000 simulations
        for k in range(self.par.K):
            #Next we loop over each graduate i including the condition that each graduate has i + 1 friends.
            for i in range(self.par.N):
                Fi = i + 1 
                #We create an empty array to store the prior expected utilities that we find doing the loop:
                prior_expected_utilities = np.zeros(self.par.J)
                for j in range(self.par.J):
                    friends_noise = np.random.normal(0, self.par.sigma**2, Fi)
                    prior_expected_utilities[j] = np.mean(self.par.v_j[j] + friends_noise)
                graduate_noise = np.random.normal(0, self.par.sigma**2, self.par.J)
                #Next we do the step that each person i chooses the career track with the highest expected utility. We use the np.argmax:
                highest_utility_career = np.argmax(prior_expected_utilities)
                #First we store the chosen career $j^k*_i$ 
                chosen_careers[i, highest_utility_career, k] = 1
                #Next we store the prior expectation of the value of their chosen career:
                expected_utilities[i, k] = prior_expected_utilities[highest_utility_career]
                #Lastly we store the realized value of their chosen career track:
                realized_utilities[i, k] = self.par.v_j[highest_utility_career] + graduate_noise[highest_utility_career]

        #We calculate the average proportion of times each career is chosen by the graduates:
        career_shares = np.mean(chosen_careers, axis=2)
        avg_expected_utilities = np.mean(expected_utilities, axis=1)
        avg_realized_utilities = np.mean(realized_utilities, axis=1)


        #Plot 1: Share of Graduates Choosing Each Career
        for j in range(self.par.J):
            plt.plot(range(1, self.par.N+1), career_shares[:, j], label=f'Career {j+1}')
        plt.title('Share of Graduates Choosing Each Career')
        plt.xlabel('Graduate Type')
        plt.ylabel('Share')
        plt.legend()
        plt.show()

        #Plot 2: Average Subjective Expected Utility
        plt.plot(range(1, self.par.N+1), avg_expected_utilities, label='Average Expected Utility')
        plt.title('Average Subjective Expected Utility')
        plt.xlabel('Graduate Type')
        plt.ylabel('Utility')
        plt.legend()
        plt.show()

        #Plot 3: Average Ex Post Realized Utility
        plt.plot(range(1, self.par.N+1), avg_realized_utilities, label='Average Realized Utility')
        plt.title('Average Ex Post Realized Utility')
        plt.xlabel('Graduate Type')
        plt.ylabel('Utility')
        plt.legend()
        plt.show()

