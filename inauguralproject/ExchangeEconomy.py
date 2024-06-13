from types import SimpleNamespace
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy import optimize
from scipy.optimize import brentq
from scipy.optimize import minimize
import numpy as np

class ExchangeEconomyClass:
    #Define parameters:
    def __init__(self):

        par = self.par = SimpleNamespace()

        # a. preferences
        par.alpha = 1/3
        par.beta = 2/3

        # b. endowments
        par.w1A = 0.8
        par.w2A = 0.3
        par.w1B = 0.2
        par.w2B = 0.7

    #Define utility functions:
    def utility_A(self,x1A,x2A):
        return(x1A**self.par.alpha)*x2A**(1-self.par.alpha)
   
    def utility_B(self,x1B,x2B):
        return(x1B**(self.par.beta)*x2B**(1-self.par.beta))

    #Define demand functions:
    def demand_A(self,p1,p2):
        par = self.par
        x1A = par.alpha*((p1*par.w1A+p2*par.w2A)/p1)
        x2A = (1-par.alpha)*((p1*par.w1A+p2*par.w2A)/p2)
        return x1A, x2A

    def demand_B(self,p1,p2):
        par = self.par
        x1B = par.beta*((p1*(1-par.w1A)+p2*(1-par.w2A))/p1)
        x2B = (1-par.beta)*((p1*(1-par.w1A)+p2*(1-par.w2A))/p2)
        return x1B, x2B

    #Finding market clearing:
    def check_market_clearing(self,p1,p2):
        par = self.par

        x1A, x2A = self.demand_A(p1, p2)
        x1B, x2B = self.demand_B(p1, p2)

        eps1 = x1A - par.w1A + x1B - (1 - par.w1A)
        eps2 = x2A - par.w2A + x2B - (1 - par.w2A)

        return eps1, eps2


##### Herfra starter den nye kode #####
    def Edgeworth(self):
        # We define the total endowment for both goods:
        w1bar = 1.0
        w2bar = 1.0

        #We set up the C-restrictions given:
        N = 75
        x_grid = np.arange(0, 1, 1/N)

        #We create an empty list for each good to store coordinates of Pareto improvements for goods 1 and 2. These lists will store the x-coordinates and y-coordinates respectively of Pareto improvements found during the loop.
        p_imp_good1 = []
        p_imp_good2 = []

        #We find the utility with the initial endowments:
        u_A_in = self.utility_A(self.par.w1A, self.par.w2A)
        u_B_in = self.utility_B(1-self.par.w1A, 1-self.par.w2A)

        #We use a for loop to iterate over the x_grid to find allocations that results in Pareto improvements over the initial endowments.
        #This gives us all the actual combinations of x1A and x2A that are pareto improvements from the initial endowments
        for x1A in x_grid:
            for x2A in x_grid:
                utility_A_q1 = self.utility_A(x1A, x2A)
                utility_B_q1 = self.utility_B(1-x1A, 1-x2A)
                if utility_A_q1 >= u_A_in and utility_B_q1 >= u_B_in:
                    p_imp_good1.append(x1A)
                    p_imp_good2.append(x2A)
                    print("Pareto improvement: (xA1 =", x1A, ", xA2 =", x2A, ")")

        # We set up the figure for the Edgeworth box:
        fig = plt.figure(frameon=False,figsize=(6,6), dpi=100)
        ax_A = fig.add_subplot(1, 1, 1)

        ax_A.set_xlabel("$x_1^A$")
        ax_A.set_ylabel("$x_2^A$")

        temp = ax_A.twinx()
        temp.set_ylabel("$x_2^B$")
        ax_B = temp.twiny()
        ax_B.set_xlabel("$x_1^B$")
        ax_B.invert_xaxis()
        ax_B.invert_yaxis()

        #We plot the initial endowment and Pareto improvement lists:
        ax_A.scatter(self.par.w1A,self.par.w2A,marker='s',color='black',label='endowment')
        ax_A.scatter(p_imp_good1,p_imp_good2,marker='o',color='green',label='possible allocations')

        # We define limits for the figure:
        ax_A.plot([0,w1bar],[0,0],lw=2,color='black')
        ax_A.plot([0,w1bar],[w2bar,w2bar],lw=2,color='black')
        ax_A.plot([0,0],[0,w2bar],lw=2,color='black')
        ax_A.plot([w1bar,w1bar],[0,w2bar],lw=2,color='black')

        ax_A.set_xlim([-0.1, w1bar + 0.1])
        ax_A.set_ylim([-0.1, w2bar + 0.1])    
        ax_B.set_xlim([w1bar + 0.1, -0.1])
        ax_B.set_ylim([w2bar + 0.1, -0.1])

        ax_A.legend(frameon=True,loc='upper right',bbox_to_anchor=(1.7,1.0))
        plt.show()

    def plot_errors(self):
        N = 75  # assuming N is 75 as in your previous code
        p1 = [0.5+2*t/N for t in range(N+1)]
        epsilon = [self.check_market_clearing(t, 1) for t in p1]

        error1 = [error[0] for error in epsilon]
        error2 = [error[1] for error in epsilon]

        for i in range(len(p1)):
            print(f"p1 = {p1[i]}, error1 = {error1[i]}, error2 = {error2[i]}")

        plt.plot(p1, error1, label='$\epsilon_1(p,\omega)$')
        plt.plot(p1, error2, label='$\epsilon_2(p,\omega)$')

        plt.xlabel('p1')
        plt.ylabel('Error with market clearing')
        plt.legend()

        plt.show()

    def excess_demand(self, p1):
        par = self.par

        x1A, x2A = self.demand_A(p1, 1)
        x1B, x2B = self.demand_B(p1, 1)

        aggregate_demand_x1 = x1A + x1B
        total_endowment_x1 = par.w1A + (1 - par.w1A)

        return aggregate_demand_x1 - total_endowment_x1

    def find_equilibrium(self, p1_guess=1.0):
        par = self.par

        # Counter:
        t = 0
        # Guess on price
        p1 = p1_guess

        # Additional parameters for find_equilibrium
        par.eps = 1e-8  # Tolerance level for equilibrium
        par.maxiter = 10000  # Maximum iterations
        par.kappa = 0.01  # Adjustment parameter
       
        # using a while loop as we don't know number of iterations a priori
        while True:

            # 1. excess demand for good 1
            Z1 = self.excess_demand(p1)
           
            # 2. check stop?
            if  np.abs(Z1) < self.par.eps or t >= self.par.maxiter:   # The first condition compares to the tolerance level and the second condition ensures that the loop does not go to infinity
                print(f'{t:3d}: p1 = {p1:12.8f} -> excess demand -> {Z1:14.8f}')
                self.p1_equilibrium = p1  # Store the equilibrium price in an instance variable
                break    
           
            # 3. Print the first 5 and every 25th iteration using the modulus operator
            if t < 5 or t%25 == 0:
                print(f'{t:3d}: p1 = {p1:12.8f} -> excess demand -> {Z1:14.8f}')
            elif t == 5:
                print('   ...')
           
            # 4. update p1
            p1 = p1 + par.kappa*Z1/2    # The price is updated by a small number (kappe) scaled to excess demand divded among the number of consumers, i.e. 2
           
            # 5. update counter and return to step 1
            t += 1    

        return p1

    def equilibrium_solutions(self):
        # Now we find the allocations and utility for consumer A and B given p1 = 0.944444460152919 as we found above.
        # First we find the allocations:
        # Consumer A's allocations:
        p1 = 0.944444460152919
        x1A_optimal_q3, x2A_optimal_q3 = self.demand_A(p1, 1)[0], self.demand_A(p1, 1)[1]
        print("Optimal allocations given p1 for consumer A:", "x1A =", x1A_optimal_q3, "x2A =", x2A_optimal_q3)

        # Consumer B's allocations:
        x1B_optimal_q3, x2B_optimal_q3 = 1-x1A_optimal_q3, 1-x2A_optimal_q3
        print("Optimal allocations given p1 for consumer B:", "x1B =", x1B_optimal_q3, "x2B =", x2B_optimal_q3)

        # Consumer A's utility:
        utility_A_q3 = self.utility_A(x1A_optimal_q3, x2A_optimal_q3)
        print("Utility for consumer A given p1 and optimal allocations:", utility_A_q3)

        # Consumer B's utility:
        utility_B_q3 = self.utility_B(x1B_optimal_q3, x2B_optimal_q3)
        print("Utility for consumer B given p1 and optimal allocations:", utility_B_q3)


    def optimal_allocation_q4a(self):
        N = 75
        p1_range = np.linspace(0.5, 2.5, N+1)
        p2 = 1

        max_utility = -np.inf
        optimal_p1 = np.nan
        xA1_optimal_q4a = np.nan
        xA2_optimal_q4a = np.nan

        for p1 in p1_range:
            x1B, x2B = self.demand_B(p1, p2)
            x1A = 1 - x1B
            x2A = 1 - x2B

            utility = self.utility_A(x1A, x2A)
            if utility > max_utility:
                max_utility = utility
                optimal_p1 = p1
                xA1_optimal_q4a = x1A
                xA2_optimal_q4a = x2A

        #We calculate the utility for consumer A given the optimal p1 and x1A and x2A:
        utility_A_optimal_q4a = self.utility_A(xA1_optimal_q4a, xA2_optimal_q4a)

        # Calculate the demand for goods by consumer B with the optimal p1
        x1B_optimal_q4a, x2B_optimal_q4a = self.demand_B(optimal_p1, p2)

        #We calculate the utility for consumer B:
        utility_B_optimal_q4a = self.utility_B(x1B_optimal_q4a, x2B_optimal_q4a)

        print("Optimal p1:", optimal_p1)
        print("Consumer A's optimal allocation: x1A =", xA1_optimal_q4a, "x2A =", xA2_optimal_q4a)
        print("Utility for consumer A:", utility_A_optimal_q4a)
        print("Consumer B's optimal allocation: x1B =", x1B_optimal_q4a, "x2B =", x2B_optimal_q4a)
        print("Utility for consumer B:", utility_B_optimal_q4a)

    def optimal_allocation_q4b(self):
        #We create the specified range for p1 and setting p2 as numeraire.
        N = 75
        p1 = [0.5+2*t/N for t in range(N+1)]
        p2 = 1

        # Define the utility function for consumer A
        def utility_A(x1B, x2B, p1, par):
            return (1-x1B)**par.alpha*(1-x2B)**(1-par.alpha)

        # Define a function that maximizes consumer A's utility given the demand for goods by consumer B and p1.
        def max_utility_A_given_B(p1, par):
            # Calculate demand for goods by consumer B given p1:
            x1B, x2B = self.demand_B(p1, 1)
            
            # Calculate utility of consumer A given demand of B and prices p1
            utility_A_q4b = utility_A(x1B, x2B, p1, par)
            
            # Use negative utility because we are maximizing from scipy
            return -utility_A_q4b

        # Initial guess for p1
        p1_initial_guess = 0.944444460152919

        # Optimize the utility of consumer A: 
        result = minimize(max_utility_A_given_B, p1_initial_guess, args=(self.par,))

        # Extract the optimal p1 from the result
        optimal_p1 = result.x[0]

        # Calculate the demand for goods by consumer B with the optimal p1
        x1B_optimal_q4b, x2B_optimal_q4b = self.demand_B(optimal_p1, 1)

        # Calculate the utility of consumer A with the optimal p1 and optimal demand of B
        utility_A_optimal_q4b = utility_A(x1B_optimal_q4b, x2B_optimal_q4b, optimal_p1, self.par)

        # Calculate the optimal amount of the two goods for consumer A
        x1A_optimal_q4b = 1 - x1B_optimal_q4b
        x2A_optimal_q4b = 1 - x2B_optimal_q4b

        # Calculate consumer B's utility:
        utility_B_optimal_q4b = self.utility_B(x1B_optimal_q4b, x2B_optimal_q4b)

        # Print out the optimal price, utility for both consumers plus both consumers optimal allocations
        print("Optimal p1:", optimal_p1)
        print("Consumer B's allocations:", "x1B =", x1B_optimal_q4b, "x2B =", x2B_optimal_q4b)
        print("Utility for consumer A given B's x1B and x2B as well p1:", utility_A_optimal_q4b)
        print("Consumer A's allocations:", "x1A =", x1A_optimal_q4b, "x2A =", x2A_optimal_q4b)
        print("Utility for consumer B:", utility_B_optimal_q4b)

    def optimal_allocation_q5a(self):
        # Initialize an empty list to store valid combinations
        N = 75
        xlist=[]

        #We define the initial utility for consumer A and B with initial endowments:
        uA_initial = self.utility_A(self.par.w1A, self.par.w2A)
        uB_initial = self.utility_B(self.par.w1B, self.par.w2B)
        print("Initial utility for A:", (uA_initial))
        print("Initial utility for B:", (uB_initial))

        # Loop through possible combinations of xA1 and xA2
        for xA1 in np.arange(0,1,1/N):
            for xA2 in np.arange(0,1,1/N):
                xB1 = 1-xA1
                xB2 = 1-xA2
                # Compute utility for consumers A and B
                uA = self.utility_A(xA1,xA2)
                uB = self.utility_B(xB1,xB2)

                # Check if the combination satisfies initial conditions. If they do, the allocation is considered valid and added to xlist.
                if uA >= uA_initial and uB >= uB_initial:
                    xlist.append((xA1,xA2))

        #The valid allocations stored in xlist are unpacked into x_values and y_values, which represent the allocations for good 1 and good 2 for consumer A.
        x_values, y_values = zip(*xlist)

        # Use the list that saves values in Z "xlist"
        uA_Z = -np.inf
        xA1_Z= np.nan
        xA2_Z= np.nan

        # We check and confirm that the new allocation gives a higher utility than the initial allocation.
        # We initialize varialbes to find the highest utility for A, meaning uA_Z and the belonging allocations xA1_Z and xA2_Z:
        for xA1, xA2 in xlist:
            if self.utility_A(xA1,xA2) > uA_Z:
                uA_Z = self.utility_A(xA1,xA2)
                xA1_Z= xA1
                xA2_Z= xA2

        #We calculate the price of good 1 by using the formula given in the assignment for x_1^A where we isolate p1:
        p1_q5a = self.par.alpha*self.par.w2A/(xA1_Z-self.par.alpha*self.par.w1A)

        print("Utility for consumer A with new allocation:", uA_Z)
        print("Utility for consumer B with new allocation:", self.utility_B(1-xA1_Z,1-xA2_Z))
        print("New allocation for consumer A:", "xA1 =", xA1_Z, "xA2 =", xA2_Z)
        print("New allocation for consumer B:", "xB1 =", 1-xA1_Z, "xB2 =", 1-xA2_Z)
        print("Price of good 1:", p1_q5a)

    def optimal_allocation_q5b(self):
        #We need to delete the constraints we put on A and therefore we only keep the ones for B:
        def constraint2(x):
            x1A, x2A = x
            return self.utility_B(1 - x1A, 1 - x2A) - self.utility_B(self.par.w1B, self.par.w2B)

        #We define the total utility of consumer B with initial endowments
        uA_initial = self.utility_A(self.par.w1A, self.par.w2A)
        uB_initial = self.utility_B(self.par.w1B, self.par.w2B)
        print("Initial utility for A:", (uA_initial))
        print("Initial utility for B:", (uB_initial))

        #We set the bounds for x1A and x2A, meaning that the allocation can only be between 0 and 1.
        bounds = [(0, 1), (0, 1)]

        #We perform the optimization using the bounds defined and the constraints above.
        result = minimize(lambda x: -self.utility_A(x[0], x[1]), x0=(0.5, 0.5), bounds=bounds, constraints=[{'type': 'ineq', 'fun': constraint2}])

        #We extract the result that we get into x1A_optimal and x2a_optimal
        x1A_optimal_q5b, x2A_optimal_q5b = result.x

        #We print out the results.
        print("x1A and x2A with no further restrictions (x1A =", x1A_optimal_q5b,", x2A =", x2A_optimal_q5b, ")")
        print("x1B and x2B (x1B =", 1-x1A_optimal_q5b,", x2B =", 1-x2A_optimal_q5b, ")")
        print("Consumer A's utility with no restrictions:", self.utility_A(x1A_optimal_q5b, x2A_optimal_q5b))
        print("Consumer B's utility:", self.utility_B(1-x1A_optimal_q5b, 1-x2A_optimal_q5b))

        #We calculate the price of good 1 by using the formula given in the assignment for x_1^A where we isolate p1:
        p1_q5b = self.par.alpha*self.par.w2A/(x1A_optimal_q5b-self.par.alpha*self.par.w1A)
        print("Optimal price:", p1_q5b)

        




