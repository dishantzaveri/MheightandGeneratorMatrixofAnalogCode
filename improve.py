import numpy as np
import time
import pickle
from scipy.optimize import minimize

# Function to calculate m-height for a given codeword
def calculate_m_height(c, m):
    abs_values = np.abs(c)
    sorted_values = np.sort(abs_values)[::-1]  # Sort in descending order
    if m < len(sorted_values) and sorted_values[m] != 0:
        return sorted_values[0] / sorted_values[m]
    else:
        return float('inf')

# Evaluate m-height for a matrix using optimization
def evaluate_matrix_with_optimization(G, num_samples=1000):
    k, n = G.shape
    m = 4  # Given m value
    max_m_height = 0

    for _ in range(num_samples):
        # Use linear programming to find x that maximizes the m-height
        def objective(x):
            c = np.dot(x, G)
            return -calculate_m_height(c, m)  # Minimize negative m-height to maximize m-height

        # Bounds for x: each element can range from -1 to 1
        bounds = [(-1, 1)] * k

        # Initial guess
        x0 = np.random.uniform(-1, 1, k)

        # Minimize using SLSQP
        result = minimize(objective, x0, bounds=bounds, method='SLSQP')

        if result.success:
            c = np.dot(result.x, G)
            m_height = calculate_m_height(c, m)
            max_m_height = max(max_m_height, m_height)

    return max_m_height

# Randomly generate matrices and evaluate them
def find_best_matrix_with_optimization(shape, rank, num_matrices=10000, value_range=(-100000, 100000), num_samples=1000):
    best_matrix = None
    best_m_height = float('inf')
    valid_rank_count = 0

    start_time = time.time()
    for i in range(num_matrices):
        # Generate a random integer matrix
        G = np.random.randint(value_range[0], value_range[1] + 1, shape)

        # Ensure the matrix has the required rank
        if np.linalg.matrix_rank(G) == rank:
            valid_rank_count += 1

            # Evaluate the matrix using optimization
            m_height = evaluate_matrix_with_optimization(G, num_samples)
            if m_height < best_m_height:
                best_matrix = G
                best_m_height = m_height

        # Log progress every 1000 matrices
        if (i + 1) % 1000 == 0:
            elapsed_time = time.time() - start_time
            print("Processed {}/{} matrices, valid rank: {}, current best m-height: {:.4f}, elapsed time: {:.2f} seconds".format(
                i + 1, num_matrices, valid_rank_count, best_m_height, elapsed_time
            ))

    return best_matrix, best_m_height

# Parameters
G1_shape = (5, 11)  # Matrix shape for G1
G2_shape = (6, 11)  # Matrix shape for G2
G1_rank = 5         # Rank for G1
G2_rank = 6         # Rank for G2
num_matrices = 10000  # Number of random matrices to generate
num_samples = 500  # Increase samples per matrix for thorough evaluation

# Find the best matrices
print("Finding best G1 matrix...")
best_G1, best_G1_m_height = find_best_matrix_with_optimization(G1_shape, G1_rank, num_matrices, num_samples=num_samples)

print("\nFinding best G2 matrix...")
best_G2, best_G2_m_height = find_best_matrix_with_optimization(G2_shape, G2_rank, num_matrices, num_samples=num_samples)

# Output the results
print("\nBest G1 Matrix:\n{}".format(best_G1))
print("Minimum m-height for G1: {:.4f}".format(best_G1_m_height))

print("\nBest G2 Matrix:\n{}".format(best_G2))
print("Minimum m-height for G2: {:.4f}".format(best_G2_m_height))

# # Save the best matrices
# matrices = {"setting1": best_G1, "setting2": best_G2}
# with open("Task3_BestMatrices.pkl", "wb") as f:
#     pickle.dump(matrices, f)
# print("\nBest matrices saved to 'Task3_BestMatrices.pkl'")
