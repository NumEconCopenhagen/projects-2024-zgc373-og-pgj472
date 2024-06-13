from types import SimpleNamespace
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy import optimize 
from scipy.optimize import brentq
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

    def find_market_clearing_price(self):
        p1_clearing = brentq(self.excess_demand, 0.01, 10)
        return p1_clearing, 1

    def find_optimal_allocations_and_utilities(self):
        # We initialize variables to track minimum absolute errors and corresponding p1 values:
        min_abs_error1 = float('inf')
        min_abs_error2 = float('inf')
        p1_min_abs_error1 = None
        p1_min_abs_error2 = None

        N = 75  # assuming N is 75 as in your previous code
        p1 = [0.5+2*t/N for t in range(N+1)]

        # We iterate over each value of p1
        for i in p1:
            # Calculating errors for the current value of p1
            p2 = 1
            error1, error2 = self.check_market_clearing(i, p2)
            
            # Updating minimum absolute errors and corresponding p1 values if smaller absolute errors are found:
            if abs(error1) < min_abs_error1:
                min_abs_error1 = abs(error1)
                p1_min_abs_error1 = i
            if abs(error2) < min_abs_error2:
                min_abs_error2 = abs(error2)
                p1_min_abs_error2 = i

        # We print the p1 values where error1 and error2 are minimized as much as possible
        print("p1 value for minimum error1:", p1_min_abs_error1, p1_min_abs_error2)

        # Now we find the allocations and utility for consumer A and B given p1 = 0.95 as we found above. 
        # First we find the allocations: 
        # Consumer A's allocations: 
        x1A_optimal_q3, x2A_optimal_q3 = self.demand_A(p1_min_abs_error1, 1)[0], self.demand_A(p1_min_abs_error2, 1)[1]
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

