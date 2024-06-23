from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt

class BarycentricClass:
    def __init__(self):
        par = self.par = SimpleNamespace()
        # Seed for reproducibility
        par.rng = np.random.default_rng(2024)

        # Generate random points
        par.X = par.rng.uniform(size=(50, 2))
        par.y = par.rng.uniform(size=(2,))
        
        # Example function f(x1, x2)
        par.f = lambda x1, x2: np.sin(np.pi * x1) * np.cos(np.pi * x2)

    # Function to compute the Euclidean distance
    def euclidean_distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    # Find points A, B, C, and D
    def find_points(self, X, y):
        A, B, C, D = None, None, None, None
        min_dist_A, min_dist_B, min_dist_C, min_dist_D = float('inf'), float('inf'), float('inf'), float('inf')
        
        for point in X:
            if point[0] > y[0] and point[1] > y[1]:
                dist = self.euclidean_distance(point, y)
                if dist < min_dist_A:
                    A = point
                    min_dist_A = dist
            elif point[0] > y[0] and point[1] < y[1]:
                dist = self.euclidean_distance(point, y)
                if dist < min_dist_B:
                    B = point
                    min_dist_B = dist
            elif point[0] < y[0] and point[1] < y[1]:
                dist = self.euclidean_distance(point, y)
                if dist < min_dist_C:
                    C = point
                    min_dist_C = dist
            elif point[0] < y[0] and point[1] > y[1]:
                dist = self.euclidean_distance(point, y)
                if dist < min_dist_D:
                    D = point
                    min_dist_D = dist

        return A, B, C, D

    # Function to check if a point is inside a triangle using barycentric coordinates
    def barycentric_coordinates(self, A, B, C, P):
        denom = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1])
        r1 = ((B[1] - C[1]) * (P[0] - C[0]) + (C[0] - B[0]) * (P[1] - C[1])) / denom
        r2 = ((C[1] - A[1]) * (P[0] - C[0]) + (A[0] - C[0]) * (P[1] - C[1])) / denom
        r3 = 1 - r1 - r2
        return r1, r2, r3

    def is_inside_triangle(self, A, B, C, P):
        r1, r2, r3 = self.barycentric_coordinates(A, B, C, P)
        return r1, r2, r3, (0 <= r1 <= 1 and 0 <= r2 <= 1 and 0 <= r3 <= 1)

    def interpolate(self):
        X = self.par.X
        y = self.par.y
        f = self.par.f
        A, B, C, D = self.find_points(X, y)
        
        if A is None or B is None or C is None or D is None:
            return float('nan')

        # Check if y is inside triangle ABC
        r1, r2, r3, inside_ABC = self.is_inside_triangle(A, B, C, y)
        if inside_ABC:
            return r1 * f(A[0], A[1]) + r2 * f(B[0], B[1]) + r3 * f(C[0], C[1])

        # Check if y is inside triangle CDA
        r1, r2, r3, inside_CDA = self.is_inside_triangle(C, D, A, y)
        if inside_CDA:
            return r1 * f(C[0], C[1]) + r2 * f(D[0], D[1]) + r3 * f(A[0], A[1])
        
        return float('nan')

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

    def compute_barycentric_coordinates(self):
        X = self.par.X
        y = self.par.y
        A, B, C, D = self.find_points(X, y)

        if A is None or B is None or C is None or D is None:
            print("One or more of the points A, B, C, D are None. Cannot compute barycentric coordinates.")
            return "NaN", "NaN"

        #Computing barycentric coordinates for triangles ABC and CDA with respect to point y:")
        r_ABC = self.barycentric_coordinates(A, B, C, y)
        r_CDA = self.barycentric_coordinates(C, D, A, y)
        
        print(f"Barycentric coordinates for triangle ABC: {r_ABC}")
        print(f"Barycentric coordinates for triangle CDA: {r_CDA}")

        inside_ABC = 0 <= r_ABC[0] <= 1 and 0 <= r_ABC[1] <= 1 and 0 <= r_ABC[2] <= 1
        inside_CDA = 0 <= r_CDA[0] <= 1 and 0 <= r_CDA[1] <= 1 and 0 <= r_CDA[2] <= 1

        if inside_ABC:
            print("Point y is inside triangle ABC.")




