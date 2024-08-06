import numpy as np


def cramers_rule(A, b):
    # Compute the determinant of the coefficient matrix A
    det_A = np.linalg.det(A)

    # Check if det_A is zero
    if det_A == 0:
        raise ValueError("The determinant of A is zero, the system may have no solution or infinitely many solutions.")

    # Number of variables
    n = A.shape[0]

    # List to store solutions
    solutions = []

    # Iterate over each variable
    for i in range(n):
        # Create a copy of A to form A_i
        A_i = A.copy()

        # Replace the i-th column of A_i with the vector b
        A_i[:, i] = b

        # Compute the determinant of A_i
        det_A_i = np.linalg.det(A_i)

        # Compute the solution for the i-th variable
        x_i = det_A_i / det_A
        solutions.append(x_i)

    return solutions

def main():
    # Example coefficient matrix and vector
    A = np.array([[2, 3], [2, 6]])
    b = np.array([10, 10])

    try:
        # Solution using Cramer's rule
        solution_cramer = cramers_rule(A, b)
        print(f"Solutions using Cramer's rule: {solution_cramer}")
    except ValueError as e:
        print(f"Cramer's rule: {e}")

    try:
        # Solution using NumPy's built-in solver
        solution_numpy = np.linalg.solve(A, b)
        print(f"Solutions using NumPy's solver: {solution_numpy}")
    except np.linalg.LinAlgError as e:
        print(f"NumPy's solver: {e}")

if __name__ == "__main__":
    main()