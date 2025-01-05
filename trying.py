import numpy as np
from scipy.optimize import minimize

def calculate_m_height(c, m):
    # Get the absolute values of the codeword and sort them in descending order
    abs_values = np.abs(c)
    sorted_indices = np.argsort(-abs_values)  # Indices sorted by descending abs value
    # Compute m-height if c[sorted_indices[m]] is non-zero
    if abs_values[sorted_indices[m]] != 0:
        return abs_values[sorted_indices[0]] / abs_values[sorted_indices[m]]
    else:
        return float('inf')  # If m-th value is zero, return infinity

def find_maximum_m_height(G, m):
    k, n = G.shape  # Get the dimensions of the matrix G

    def objective(x):
        c = np.dot(x, G)  # Generate codeword c for given vector x
        m_height = calculate_m_height(c, m)
        return -m_height  # Negate to maximize (since we want maximum m-height)

    # Initial guess for x
    x0 = np.random.normal(0, 1, k)

    # Run optimization to maximize m-height
    result = minimize(objective, x0, method='SLSQP', options={'disp': False})
    max_m_height = -result.fun if result.success else float('inf')
    
    return max_m_height

def find_best_matrix(matrix_size, m, num_matrices=10000):
    best_m_height = float('inf')
    best_matrix = None

    for _ in range(num_matrices):
        # Generate a random matrix G of specified size (e.g., 5x11 or 6x11)
        G = np.random.normal(0, 0.5, matrix_size)  # Adjust std deviation as needed

        # Find the maximum possible m-height for this matrix
        max_m_height = find_maximum_m_height(G, m)

        # Check if this is the best m-height (smallest) seen so far
        if max_m_height < best_m_height:
            best_m_height = max_m_height
            best_matrix = G

    return best_matrix, best_m_height

# Parameters
matrix_size = (5, 11)  # Matrix size (rows, columns) - adjust for G1 (5x11) or G2 (6x11)
m = 4  # Target m value
num_matrices = 10000  # Number of random matrices to sample

# Find the best matrix with the lowest maximum m-height
best_matrix, best_m_height = find_best_matrix(matrix_size, m, num_matrices)

print("Best Matrix with lowest maximum m-height:\n", best_matrix)
print("Lowest maximum m-height achieved:", best_m_height)
