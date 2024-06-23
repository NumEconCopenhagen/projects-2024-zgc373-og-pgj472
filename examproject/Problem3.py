#We have used the Copilot as permitted as a tool to adjust and correct our code when solving the problem. 
#We write in the code below the parts where we have mainly copied the code from the Copilot.

#Us writing the code:
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt

class BarycentricClass:
    def __init__(self):
        par = self.par = SimpleNamespace()
        #Given seed:
        par.rng = np.random.default_rng(2024)

        # Given generate random points:
        par.X = par.rng.uniform(size=(50, 2))
        par.y = par.rng.uniform(size=(2,))
        
        #Given function f(x1, x2) = x1 * x2:
        par.f = lambda x: x[0] * x[1]
        #Given array of f(x) for all x in X:
        par.F = np.array([par.f(x) for x in par.X])

    # We define the points given under building block II in the assignment text
    def building_block_II(self, p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    # Now we can use this to find the points A, B, C, and D:
    #Copilot:
    def find_points(self, X, y):
        A, B, C, D = None, None, None, None
        min_dist_A, min_dist_B, min_dist_C, min_dist_D = float('inf'), float('inf'), float('inf'), float('inf')
        
        for point in X:
            if point[0] > y[0] and point[1] > y[1]:
                dist = self.building_block_II(point, y)
                if dist < min_dist_A:
                    A = point
                    min_dist_A = dist
            elif point[0] > y[0] and point[1] < y[1]:
                dist = self.building_block_II(point, y)
                if dist < min_dist_B:
                    B = point
                    min_dist_B = dist
            elif point[0] < y[0] and point[1] < y[1]:
                dist = self.building_block_II(point, y)
                if dist < min_dist_C:
                    C = point
                    min_dist_C = dist
            elif point[0] < y[0] and point[1] > y[1]:
                dist = self.building_block_II(point, y)
                if dist < min_dist_D:
                    D = point
                    min_dist_D = dist

        return A, B, C, D

    #Us writing the code:
    #Next we find Barycentric coordinates for the point y with respect to the triangles ABC and CDA:
    #Because the denominator in the functions given are the same, we just define it and refer to it as seen below: 
    def barycentric_coordinates(self, A, B, C, y):
        denominator = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1])
        r1 = ((B[1] - C[1]) * (y[0] - C[0]) + (C[0] - B[0]) * (y[1] - C[1])) / denominator
        r2 = ((C[1] - A[1]) * (y[0] - C[0]) + (A[0] - C[0]) * (y[1] - C[1])) / denominator
        r3 = 1 - r1 - r2
        return r1, r2, r3

    def is_inside_triangle(self, A, B, C, y):
        r1, r2, r3 = self.barycentric_coordinates(A, B, C, y)
        #Returing r1, r2, r3 where the conditions for r1, r2 and r3 that are given are include:
        return r1, r2, r3, (0 <= r1 <= 1 and 0 <= r2 <= 1 and 0 <= r3 <= 1)

    #We can now interpolate the function f(x1, x2) = x1 * x2 at point y:
    def interpolate(self):
        X = self.par.X
        y = self.par.y
        f = self.par.f
        A, B, C, D = self.find_points(X, y)
        
        #As told in the assignment we need to have under the algorithm's first pullet that if A, B, C or D is None, we return NaN: 
        if A is None or B is None or C is None or D is None:
            return float('nan')

        #We check if the point y is inside the triangle ABC:
        r1, r2, r3, inside_ABC = self.is_inside_triangle(A, B, C, y)
        if inside_ABC:
            return r1 * f(A) + r2 * f(B) + r3 * f(C)

        #We check if it is inside the triangle CDA
        r1, r2, r3, inside_CDA = self.is_inside_triangle(C, D, A, y)
        if inside_CDA:
            return r1 * f(C) + r2 * f(D) + r3 * f(A)
        
        return float('nan')

    #We used Copilot to write the plot-code:
    def plot_points_and_triangles(self):
        X = self.par.X
        y = self.par.y
        A, B, C, D = self.find_points(X, y)

        plt.scatter(X[:, 0], X[:, 1], label='Points in X', color='blue')
        plt.scatter(y[0], y[1], label='Point y', color='red')
        if A is not None and B is not None and C is not None:
            plt.scatter([A[0], B[0], C[0]], [A[1], B[1], C[1]], label='Points A, B, C', color='green')
            plt.plot([A[0], B[0], C[0], A[0]], [A[1], B[1], C[1], A[1]], 'g--')
        if C is not None and D is not None and A is not None:
            plt.scatter([C[0], D[0], A[0]], [C[1], D[1], A[1]], label='Points C, D, A', color='orange')
            plt.plot([C[0], D[0], A[0], C[0]], [C[1], D[1], A[1], C[1]], 'orange')

        plt.legend()
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Points and Triangles ABC and CDA')
        plt.show()

    #Us writing the code: 
    #We can now compute the barycentric coordinates for the point y for question 2:
    def compute_barycentric_coordinates(self):
        X = self.par.X
        y = self.par.y
        A, B, C, D = self.find_points(X, y)

        if A is None or B is None or C is None or D is None:
            print("One or more of the points A, B, C, D are None. Cannot compute barycentric coordinates.")
            return "NaN", "NaN"

        #We compute the barycentric coordinates for triangles ABC and CDA with respect to point y:")
        r_ABC = self.barycentric_coordinates(A, B, C, y)
        r_CDA = self.barycentric_coordinates(C, D, A, y)
        
        print(f"For triangle ABC: r1 = {r_ABC[0]}, r2 = {r_ABC[1]}, r3 = {r_ABC[2]}")
        print(f"For triangle CDA: r1 = {r_CDA[0]}, r2 = {r_CDA[1]}, r3 = {r_CDA[2]}")

        #We check if the point y is inside the triangle ABC: ex. if r_ABC[0] is not between 0 and 1, it is not inside the triangle.
        inside_ABC = 0 <= r_ABC[0] <= 1 and 0 <= r_ABC[1] <= 1 and 0 <= r_ABC[2] <= 1

        if inside_ABC:
            print("Point y is inside triangle ABC.")

    #Here we evaluate the points in Y that is given in the assignment for question 4:
    def evaluate_points_in_Y(self): 
        #Here we note that two coordinates in the assignment text are identical so we only write it ones.
        Y = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.5, 0.5)] 
        results = [] 
        
        #We loop through the given points in Y
        for y in Y: 
            self.par.y = y
            interpolated_value = self.interpolate()
            true_value = self.par.f(y)
            results.append((y, true_value, interpolated_value)) 
            print(f"For point y = {y}:") 
            print(f"True value of f(y): {true_value}") 
            print(f"Interpolated value of f(y): {interpolated_value}") 
            print() 
            