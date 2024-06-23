from types import SimpleNamespace
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy import optimize
from scipy.optimize import brentq
from scipy.optimize import minimize
import pandas as pd
import numpy as np


class ExchangeEconomyClass:
    #We define parameters:
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

    #We define utility functions:
    def utility_A(self,x1A,x2A):
        return(x1A**self.par.alpha)*x2A**(1-self.par.alpha)
   
    def utility_B(self,x1B,x2B):
        return(x1B**(self.par.beta)*x2B**(1-self.par.beta))

    #We define demand functions:
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

    #We find market clearing:
    def check_market_clearing(self,p1,p2):
        par = self.par

        x1A, x2A = self.demand_A(p1, p2)
        x1B, x2B = self.demand_B(p1, p2)

        eps1 = x1A - par.w1A + x1B - (1 - par.w1A)
        eps2 = x2A - par.w2A + x2B - (1 - par.w2A)

        return eps1, eps2

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
        #This gives us all the actual combinations of x1A and x2A that are pareto improvements from the initial endowments.
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
        #Next we plot the errors for the market clearing for different values of p1:
        N = 75
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

        #We add additional parameters for find_equilibrium which are tolarance level, maximum iterations and an adjustment parameter. 
        par.eps = 1e-8  #Tolerance level for equilibrium
        par.maxiter = 10000  #Maximum iterations
        par.kappa = 0.01  #Adjustment parameter
       
        #We use a while loop to find equilibrium price. The loop will run until the excess demand is below the tolerance level or/and the number of iterations reach the maximum number of iterations we have set to 10000.
        while True:

            #excess demand for good 1
            Z1 = self.excess_demand(p1)
           
            #The iteration stops when the price is close enough to equilibrium or a maximum iteration set to 10000.
            if  np.abs(Z1) < self.par.eps or t >= self.par.maxiter:   # The first condition compares to the tolerance level and the second condition ensures that the loop does not go to infinity.
                print(f'{t:3d}: p1 = {p1:12.8f} -> excess demand -> {Z1:14.8f}')
                self.p1_equilibrium = p1  #We Store the equilibrium price in an instance variable.
                break    
           
            #We print the first 5 and every 25th iteration using the modulus operator:
            if t < 5 or t%25 == 0:
                print(f'{t:3d}: p1 = {p1:12.8f} -> excess demand -> {Z1:14.8f}')
            elif t == 5:
                print('   ...')
           
            #We update p1
            p1 = p1 + par.kappa*Z1/2    # The price is updated by a small number (kappe) scaled to excess demand divded among the number of consumers, i.e. 2
           
            #We update the counter and return to step 1
            t += 1    
        
        print("Market clearing price: ", p1)


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

        #Then we find the utility for consumer A and B given the optimal allocations:
        # Consumer A's utility:
        utility_A_q3 = self.utility_A(x1A_optimal_q3, x2A_optimal_q3)
        print("Utility for consumer A given p1 and optimal allocations:", utility_A_q3)

        # Consumer B's utility:
        utility_B_q3 = self.utility_B(x1B_optimal_q3, x2B_optimal_q3)
        print("Utility for consumer B given p1 and optimal allocations:", utility_B_q3)

    def optimal_allocation_q4a(self):
        #We create the specified range for p1 and setting p2 as numeraire.
        N = 75
        p1_range = np.linspace(0.5, 2.5, N+1)
        p2 = 1

        #We initialize the variables to find the optimal price, x1A and x2A for consumer A:
        max_utility = -np.inf
        optimal_p1_q4a = np.nan
        xA1_optimal_q4a = np.nan
        xA2_optimal_q4a = np.nan

        for p1 in p1_range:
            x1B, x2B = self.demand_B(p1, p2)
            x1A = 1 - x1B
            x2A = 1 - x2B

            utility = self.utility_A(x1A, x2A)
            if utility > max_utility:
                max_utility = utility
                optimal_p1_q4a = p1
                xA1_optimal_q4a = x1A
                xA2_optimal_q4a = x2A

        #We calculate the utility for consumer A given the optimal p1 and x1A and x2A:
        utility_A_optimal_q4a = self.utility_A(xA1_optimal_q4a, xA2_optimal_q4a)

        #We calculate the demand for goods by consumer B with the optimal p1:
        x1B_optimal_q4a, x2B_optimal_q4a = self.demand_B(optimal_p1_q4a, p2)

        #We calculate the utility for consumer B:
        utility_B_optimal_q4a = self.utility_B(x1B_optimal_q4a, x2B_optimal_q4a)

        print("Optimal p1:", optimal_p1_q4a)
        print("Consumer A's optimal allocation: x1A =", xA1_optimal_q4a, "x2A =", xA2_optimal_q4a)
        print("Consumer B's optimal allocation: x1B =", x1B_optimal_q4a, "x2B =", x2B_optimal_q4a)
        print("Utility for consumer A:", utility_A_optimal_q4a)
        print("Utility for consumer B:", utility_B_optimal_q4a)

    def optimal_allocation_q4b(self):
        #We create the specified range for p1 and setting p2 as numeraire.
        N = 75
        p1 = [0.5+2*t/N for t in range(N+1)]
        p2 = 1

        #We define the utility function for consumer A:
        def utility_A(x1B, x2B, p1, par):
            return (1-x1B)**par.alpha*(1-x2B)**(1-par.alpha)

        #We define a function that maximizes consumer A's utility given the demand for goods by consumer B and p1.
        def max_utility_A_given_B(p1, par):
            #We calculate demand for goods by consumer B given p1:
            x1B, x2B = self.demand_B(p1, 1)
            
            #We calculate utility of consumer A given demand of B and prices p1:
            utility_A_q4b = utility_A(x1B, x2B, p1, par)
            
            #We use negative utility because we are maximizing from scipy:
            return -utility_A_q4b

        #We set the initial guess for p1 as the p1 found in question 3:
        p1_initial_guess = 0.944444460152919

        #We optimize the utility of consumer A: 
        result = minimize(max_utility_A_given_B, p1_initial_guess, args=(self.par,))

        #We extract the optimal p1 from the result:
        optimal_p1_q4b = result.x[0]

        #We calculate the demand for goods by consumer B with the optimal p1:
        x1B_optimal_q4b, x2B_optimal_q4b = self.demand_B(optimal_p1_q4b, 1)

        #We calculate the utility of consumer A with the optimal p1 and optimal demand of B:
        utility_A_optimal_q4b = utility_A(x1B_optimal_q4b, x2B_optimal_q4b, optimal_p1_q4b, self.par)

        #We calculate the optimal amount of the two goods for consumer A:
        x1A_optimal_q4b = 1 - x1B_optimal_q4b
        x2A_optimal_q4b = 1 - x2B_optimal_q4b

        #We calculate consumer B's utility:
        utility_B_optimal_q4b = self.utility_B(x1B_optimal_q4b, x2B_optimal_q4b)

        print("Optimal p1:", optimal_p1_q4b)
        print("Utility for consumer A given B's x1B and x2B as well p1:", utility_A_optimal_q4b)
        print("Utility for consumer B:", utility_B_optimal_q4b)
        print("Consumer A's allocations:", "x1A =", x1A_optimal_q4b, "x2A =", x2A_optimal_q4b)
        print("Consumer B's allocations:", "x1B =", x1B_optimal_q4b, "x2B =", x2B_optimal_q4b)

    def optimal_allocation_q5a(self):
        #We initialize an empty list to store valid combinations:
        N = 75
        xlist=[]

        #We define the initial utility for consumer A and B with initial endowments:
        uA_initial = self.utility_A(self.par.w1A, self.par.w2A)
        uB_initial = self.utility_B(self.par.w1B, self.par.w2B)
        print("Initial utility for A:", (uA_initial))
        print("Initial utility for B:", (uB_initial))

        #We loop through possible combinations of xA1 and xA2:
        for xA1 in np.arange(0,1,1/N):
            for xA2 in np.arange(0,1,1/N):
                xB1 = 1-xA1
                xB2 = 1-xA2
                #We find utility for consumers A and B:
                uA = self.utility_A(xA1,xA2)
                uB = self.utility_B(xB1,xB2)

                #We check if the combination satisfies the initial conditions. If they do, the allocation is considered valid and added to xlist.
                if uA >= uA_initial and uB >= uB_initial:
                    xlist.append((xA1,xA2))

        #We use the list that saves values in Z "xlist"
        uA_Z = -np.inf
        xA1_Z= np.nan
        xA2_Z= np.nan

        # We check and confirm that the new allocation gives a higher utility than the initial allocation.
        # We initialize variables to find the highest utility for A, meaning uA_Z and the belonging allocations xA1_Z and xA2_Z:
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

        #We extract the result that we get into x1A_optimal and x2a_optimal:
        x1A_optimal_q5b, x2A_optimal_q5b = result.x

        print("Consumer A's utility with no restrictions:", self.utility_A(x1A_optimal_q5b, x2A_optimal_q5b))
        print("Consumer B's utility:", self.utility_B(1-x1A_optimal_q5b, 1-x2A_optimal_q5b))
        print("x1A and x2A with no further restrictions (x1A =", x1A_optimal_q5b,", x2A =", x2A_optimal_q5b, ")")
        print("x1B and x2B (x1B =", 1-x1A_optimal_q5b,", x2B =", 1-x2A_optimal_q5b, ")")

        #We calculate the price using the same method as in q5a:
        p1_q5b = self.par.alpha*self.par.w2A/(x1A_optimal_q5b-self.par.alpha*self.par.w1A)
        print("Optimal price:", p1_q5b)

        
    def optimal_allocation_q6a(self):
        #We are defining the total utility of both consumer with their intital endownments.
        uA_initial = self.utility_A(self.par.w1A, self.par.w2A)
        uB_initial = self.utility_B(self.par.w1B, self.par.w2B)
        agg_initial = uA_initial + uB_initial
        print("Aggregated initial utility =", agg_initial)

        #We set the bounds for x1A and x2A, meaning that the allocation can only be between 0 and 1.
        bounds = [(0, 1), (0, 1)]

        #We perform the optimization using the bounds defined and the constraints above.
        result = minimize(lambda x: -(self.utility_A(x[0], x[1]) + self.utility_B(1 - x[0], 1 - x[1])), x0=(0.5, 0.5), bounds=bounds)

        #We extract the result that we get into x1A_optimal and x2a_optimal
        x1A_optimal_q6a, x2A_optimal_q6a = result.x

        print("Aggregated utility for consumer A and B is =", self.utility_A(x1A_optimal_q6a, x2A_optimal_q6a) + self.utility_B(1-x1A_optimal_q6a, 1-x2A_optimal_q6a))
        print("x1A and x2A are (x1A =", x1A_optimal_q6a,", x2A =", x2A_optimal_q6a, ")")
        print("x1B and x2B are (x1B =", 1-x1A_optimal_q6a, ", x2B =", 1-x2A_optimal_q6a, ")")

        #We calculate the price using the same method as in q5a:
        p1_q6a = self.par.alpha*self.par.w2A/(x1A_optimal_q6a-self.par.alpha*self.par.w1A)
        print("Optimal price:", p1_q6a)


    def comparing_results(self):
        #We create a list of all results to create table: 
        #Prices:
        p1_q3 = 0.944444460152919
        optimal_p1_q4a = 1.8866666666666667
        optimal_p1_q4b = 1.8992348940312658
        p1_q5a = 0.34090909090909083
        p1_q5b = 0.24142687341421534
        #Utilities:
        utility_A_q3 = 0.569273589063561
        utility_A_optimal_q4a = 0.633615985237553
        utility_A_optimal_q4b = 0.6336208503209155
        uA_Z = 0.7415523509091093
        utility_A_optimal_q5b = 0.7100258612285557
        utility_B_q3 = 0.4886095365292365
        utility_B_optimal_q4a = 0.37335220631773364
        utility_B_optimal_q4b = 0.37257053552742647
        utility_B_optimal_q5a = 0.30507896071427915
        utility_B_optimal_q5b = 0.30365889718737693
        #Allocations:
        x1A_optimal_q3 = 0.3725490178467546
        x1A_optimal_q4a = 0.619316843345112
        x1A_optimal_q4b = 0.6209536860694472
        xA1_Z = 0.56
        x1A_optimal_q5b = 0.6808707632770316
        x2A_optimal_q3 = 0.7037037120815569
        x2A_optimal_q4a = 0.6408888888888888
        x2A_optimal_q4b = 0.6400510070645823
        xA2_Z = 0.8533333333333334
        x2A_optimal_q5b = 0.7250682829856586
        x1B_optimal_q3 = 0.6274509821532455
        x1B_optimal_q4a = 0.38068315665488806
        x1B_optimal_q4b = 0.37904631393055277
        x1B_optimal_q5a = 0.43999999999999995
        x1B_optimal_q5b = 0.31912923672296845
        x2B_optimal_q3 = 0.2962962879184431
        x2B_optimal_q4a = 0.3591111111111111
        x2B_optimal_q4b = 0.3599489929354177
        x2B_optimal_q5a = 0.1466666666666666
        x2B_optimal_q5b = 0.2749317170143414
    
        #We create a table to make comparison easier from question 3-5: 
        questions = ['3', '4a', '4b', '5a', '5b']
        prices = [p1_q3, optimal_p1_q4a, optimal_p1_q4b, p1_q5a, p1_q5b]
        utility_A = [utility_A_q3, utility_A_optimal_q4a, utility_A_optimal_q4b, uA_Z, utility_A_optimal_q5b] 
        utility_B = [utility_B_q3, utility_B_optimal_q4a, utility_B_optimal_q4b, utility_B_optimal_q5a, utility_B_optimal_q5b]
        x1A_allocations = [x1A_optimal_q3, x1A_optimal_q4a, x1A_optimal_q4b, xA1_Z, x1A_optimal_q5b]
        x2A_allocations = [x2A_optimal_q3, x2A_optimal_q4a, x2A_optimal_q4b, xA2_Z, x2A_optimal_q5b]
        x1B_allocations = [x1B_optimal_q3, x1B_optimal_q4a, x1B_optimal_q4b, x1B_optimal_q5a, x1B_optimal_q5b]
        x2B_allocations = [x2B_optimal_q3, x2B_optimal_q4a, x2B_optimal_q4b, x2B_optimal_q5a, x2B_optimal_q5b]

        # We are creating a dictionary to hold data, and naming them after, what they hold.
        data = {'Question': questions,
                'Price (p1)': prices,
                'Utility for Consumer A': utility_A,
                'Utility for Consumer B': utility_B,
                'Allocation of x1A': x1A_allocations,
                'Allocation of x2A': x2A_allocations,
                'Allocation of x1B': x1B_allocations,
                'Allocation of x2B': x2B_allocations}

        #We create a DataFrame from the dictionary above, so that we can print it.
        df = pd.DataFrame(data)

        # We are displaying the DataFrame without index, because we don't want the standard index created by Pyhton.
        print(df.to_string(index=False))
       

    def Edgeworth_box_comparison(self):
        # We define the total endowment for both goods:
        w1bar = 1.0
        w2bar = 1.0
        
        #We set up the C-restrictions given:
        N = 75
        x_grid = np.arange(0, 1, 1/N)

        #We create an empty list for each good to store coordinates of Pareto improvements for goods 1 and 2.
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

        ax_A.scatter(self.par.w1A,self.par.w2A,marker='s',color='black',label='endowment')
        ax_A.scatter(p_imp_good1,p_imp_good2,marker='o',color='green',label='possible allocations')

        #Here we insert the different allocations that we have have found from question 3-5 into the Edgeworth box.
        x1A_optimal_q3 = 0.3725490178467546
        x1A_optimal_q4a = 0.619316843345112
        x1A_optimal_q4b = 0.6209536860694472
        xA1_Z = 0.56
        x1A_optimal_q5b = 0.6808707632770316
        x2A_optimal_q3 = 0.7037037120815569
        x2A_optimal_q4a = 0.6408888888888888
        x2A_optimal_q4b = 0.6400510070645823
        xA2_Z = 0.8533333333333334
        x2A_optimal_q5b = 0.7250682829856586

        ax_A.scatter(x1A_optimal_q3,x2A_optimal_q3,marker='s',color='blue',label='q3')
        ax_A.scatter(x1A_optimal_q4a,x2A_optimal_q4a,marker='s',color='cyan',label='q4A')
        ax_A.scatter(x1A_optimal_q4b,x2A_optimal_q4b,marker='s',color='yellow',label='q4B')
        ax_A.scatter(xA1_Z,xA2_Z,marker='s',color='orange',label='q5A')
        ax_A.scatter(x1A_optimal_q5b,x2A_optimal_q5b,marker='s',color='purple',label='q5B')

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


    def market_equilibrium(self):
        #First we want to just illustrate the endowment from the random drawn set: 
        #Here we get 50 random values for wA1 and wA2
        wA1 = np.random.uniform(0, 1, size=50)
        wA2 = np.random.uniform(0, 1, size=50)

        #We define the function to calculate market equilibrium allocation for a given pair of wA1 and wA2
        def market_equilibrium_allocation(wA1, wA2):
            #We define the parameter values:
            alpha = 1/3
            beta = 2/3

            #We define the demand functions for consumer A and B with wA1 and wA2 that we have defined in the beginning:
            def demand_A(p1, p2):
                x1A = alpha*((p1*wA1+p2*wA2)/p1)
                x2A = (1-alpha)*((p1*wA1+p2*wA2)/p2)
                return x1A, x2A
            
            def demand_B(p1, p2):
                x1B = beta*((p1*(1-wA1)+p2*(1-wA2))/p1)
                x2B = (1-beta)*((p1*(1-wA1)+p2*(1-wA2))/p2)
                return x1B, x2B

            #We define the error functions for market clearing condition with the wA1 and wA2 that we have defined in the beginning:
            def error1(p1, p2):
                x1A, x2A = demand_A(p1, p2)
                x1B, x2B = demand_B(p1, p2)
                return x1A + x1B - (wA1 + (1 - wA1))  # Market clearing for good 1

            def error2(p1, p2):
                x1A, x2A = demand_A(p1, p2)
                x1B, x2B = demand_B(p1, p2)
                return x2A + x2B - (wA2 + (1 - wA2))  # Market clearing for good 2

            #We find the market equilibrium prices using scipy's root-finding function.
            #By using the root-finding function it helps us to find the market equilibrium prices by solving the equations representing the excess demand for each good.
            from scipy.optimize import root
            result = root(lambda x: [error1(*x), error2(*x)], x0=[1, 1]) 
            p1, p2 = result.x

            #We calculate the market equilibrium allocations based on the p1 and p2 we found right above.
            x1A, x2A = demand_A(p1, p2)
            x1B, x2B = demand_B(p1, p2)
            return p1, p2, x1A, x2A, x1B, x2B

        #We calculate the market equilibrium allocation for both wA1 and wA2:
        market_allocations = [market_equilibrium_allocation(w1, w2) for w1, w2 in zip(wA1, wA2)]

        #We extracting allocations for plotting:
        x1A_values = [allocation[2] for allocation in market_allocations] #It is [2] because it is the 2 element in the return function under where we calculate the market equilibrium allocations based on the p1 and p2.
        x2A_values = [allocation[3] for allocation in market_allocations] #Same for [3] being 3 element

        #We plot the W set with the market equilibrium allocations:
        plt.figure(figsize=(8, 6))
        plt.scatter(wA1, wA2, color='blue', marker='o', label='Set W')
        plt.scatter(x1A_values, x2A_values, color='red', marker='x', label='Market Equilibrium Allocation')
        plt.title('Set W with Market Equilibrium Allocation')
        plt.xlabel('wA1')
        plt.ylabel('wA2')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()
