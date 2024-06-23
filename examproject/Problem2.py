#We have used the Copilot as permitted as a tool to adjust and correct our code when solving the problem. 
#We write in the code below the parts where we have mainly copied the code from the Copilot.

#Us writing the code:
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
        par.c = 1

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
                #For the next piece we used copilot to generate the code:
                #First we store the chosen career $j^k*_i$ 
                chosen_careers[i, highest_utility_career, k] = 1
                #Next we store the prior expectation of the value of their chosen career:
                expected_utilities[i, k] = prior_expected_utilities[highest_utility_career]
                #Lastly we store the realized value of their chosen career track:
                realized_utilities[i, k] = self.par.v_j[highest_utility_career] + graduate_noise[highest_utility_career]

        #Us writing the code: 
        #We calculate the average proportion of times each career is chosen by the graduates and the average expected and realized utilities for each graduate type:
        career_shares = np.mean(chosen_careers, axis=2)
        avg_expected_utilities = np.mean(expected_utilities, axis=1)
        avg_realized_utilities = np.mean(realized_utilities, axis=1)

        #For the following plot we used copilot to generate the code: 
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


    #Us writing the code: 
    #We will use the same approach as in q2, which is why the code will be very similar to the code provided in q2:
    def simulation_q3(self):
        #We create arrays to store our results
        chosen_careers = np.zeros((self.par.N, self.par.J, self.par.K))
        expected_utilities = np.zeros((self.par.N, self.par.K))
        realized_utilities = np.zeros((self.par.N, self.par.K))
        new_chosen_careers = np.zeros((self.par.N, self.par.J, self.par.K))
        new_expected_utilities = np.zeros((self.par.N, self.par.K))
        new_realized_utilities = np.zeros((self.par.N, self.par.K))
        switch_decisions = np.zeros((self.par.N, self.par.J))

        #We use the same style of loop, but now we add an extra loop for the new career choice options:
        for k in range(self.par.K):
            for i in range(self.par.N):
                Fi = i + 1 
                prior_expected_utilities = np.zeros(self.par.J)
                for j in range(self.par.J):
                    friends_noise = np.random.normal(0, self.par.sigma**2, Fi)
                    prior_expected_utilities[j] = np.mean(self.par.v_j[j] + friends_noise)
                graduate_noise = np.random.normal(0, self.par.sigma**2, self.par.J)
                #Next, we calculate the highest expected utility given career choices:
                initial_career = np.argmax(prior_expected_utilities)
                chosen_careers[i, initial_career, k] = 1
                expected_utilities[i, k] = prior_expected_utilities[initial_career]
                realized_utilities[i, k] = self.par.v[initial_career] + graduate_noise[initial_career]

                #We used copilot for the next part: 
                #Here we create the extra loop for the new career choice options after a year:
                new_prior_expected_utilities = np.zeros(self.par.J)
                for j in range(self.par.J):
                    if j == initial_career:
                        new_prior_expected_utilities[j] = realized_utilities[i, k]
                    else:
                        new_prior_expected_utilities[j] = prior_expected_utilities[j] - self.par.c
                #We find the highest utility for the new career choice options:
                new_career = np.argmax(new_prior_expected_utilities)
                new_chosen_careers[i, new_career, k] = 1
                new_expected_utilities[i, k] = new_prior_expected_utilities[new_career]
                #The last part of this code, says that the graduate will only switch career if the new career has a higher expected utility than the initial career:
                new_realized_utilities[i, k] = self.par.v_j[new_career] + graduate_noise[new_career] - (self.par.c if new_career != initial_career else 0)
                #Here we say that if the new career is different from the initial career, then the graduate will switch their career:
                if new_career != initial_career:
                    switch_decisions[i, initial_career] += 1

        #Us writing the code:
        #We calculate the average proportion of times each career is chosen by the graduates and the average expected and realized utilities for each graduate type:
        career_shares = np.mean(chosen_careers, axis=2)
        avg_expected_utilities = np.mean(expected_utilities, axis=1)
        avg_realized_utilities = np.mean(realized_utilities, axis=1)
        new_career_shares = np.mean(new_chosen_careers, axis=2)
        new_avg_expected_utilities = np.mean(new_expected_utilities, axis=1)
        new_avg_realized_utilities = np.mean(new_realized_utilities, axis=1)
        switch_proportions = switch_decisions / self.par.K

        #For the following plot we used copilot to generate the code: 
        #We start by plotting the career shares old and new:
        for j in range(self.par.J):
            plt.plot(career_shares[:, j], label=f'Initial Career Shares {j+1}')
            plt.plot(new_career_shares[:, j], label=f'New Career Shares {j+1}', linestyle='--')
        plt.legend()
        plt.title('Career Shares Comparison')
        plt.xlabel('Graduate Type')
        plt.ylabel('Share')
        plt.show()

        #Second, we plot the expected utilities old and new:
        plt.plot(avg_expected_utilities, label='Initial Expected Utility')
        plt.plot(new_avg_expected_utilities, label='New Expected Utility')
        plt.legend()
        plt.title('Average Expected Utility Over Time')
        plt.xlabel('Graduate Type')
        plt.ylabel('Utility')
        plt.show()

        #Third, we plot the realized utilities old and new:
        plt.plot(avg_realized_utilities, label='Initial Realized Utility')
        plt.plot(new_avg_realized_utilities, label='New Realized Utility')
        plt.legend()
        plt.title('Average Realized Utility Over Time')
        plt.xlabel('Graduate Type')
        plt.ylabel('Utility')
        plt.show()
       
        #Fourth, we plot the switch decisions:
        for j in range(self.par.J):
            plt.plot(switch_proportions[:, j], label=f'Career {j+1}')
        plt.legend()
        plt.title('Proportion of Switch Decisions Over Time')
        plt.xlabel('Graduate Type')
        plt.ylabel('Proportion')
        plt.show()

