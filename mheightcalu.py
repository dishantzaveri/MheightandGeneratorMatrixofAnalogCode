import numpy as np
from scipy.optimize import minimize

# Provided matrices G1 and G2
G1 = np.array([
    [9649, 46867, 31932, 3694, 68894, 19879, 10268, -45114, 37337, 68266, -12502],
    [418, 26324, 75203, 91335, 2985, -83977, -58910, -32779, -35180, -99231, -40265],
    [9328, 99041, -94689, 3355, 66602, 84779, -14695, 59765, 30608, 56730, -15522],
    [10228, 49503, 30523, -97253, 73028, 23855, 90222, -34275, 29981, -15346, 66845],
    [7037, -32565, -43114, -33197, 50244, -68449, 16216, -88606, -30908, -96110, -58394]
])

G2 = np.array([
    [21958, 46867, 31932, 3694, 3887, 19879, 10268, -45114, 37337, 68266, -12502],
    [12727, 26324, 75203, 91335, 2253, -83977, -58910, -32779, -35180, -99231, -40265],
    [94027, -35075, 99041, -94689, 16644, 3355, 84779, -14695, 59765, 30608, 56730],
    [-15522, 22537, 49503, 30523, -2748, -97253, 23855, 90222, -34275, 29981, -15346],
    [66845, 19346, -32565, -43114, 11832, -33197, -68449, 16216, -88606, -30908, -96110],
    [-58394, -12687, 41699, -91208, -10336, 74073, 54969, -31852, 24243, 20174, 54555]
])

# Function to calculate the m-height of a codeword c
def calculate_m_height(c, m):
    abs_values = np.abs(c)
    sorted_values = np.sort(abs_values)[::-1]
    if sorted_values[m] != 0:
        return sorted_values[0] / sorted_values[m]
    else:
        return float('inf')

# Function to find the minimum m-height for a given generator matrix G and parameter m
def min_m_height_for_matrix(G, m):
    k = G.shape[0]

    def objective(x):
        c = np.dot(x, G)
        return calculate_m_height(c, m)  # Directly minimize m-height

    x0 = np.random.normal(0, 1, k)
    result = minimize(objective, x0, method='SLSQP', bounds=[(-1, 1)] * k)
    if result.success:
        return result.fun, result.x  # Return the min m-height and the optimized x
    else:
        return float('inf'), None

# Compute minimum m-heights and optimized vectors for G1 and G2
m = 4  # Given m value in your task
G1_min_m_height, G1_optimized_x = min_m_height_for_matrix(G1, m)
G2_min_m_height, G2_optimized_x = min_m_height_for_matrix(G2, m)

print("Minimum m-height for G1:", G1_min_m_height)
print("Optimized vector x for G1:", G1_optimized_x)
print("\nMinimum m-height for G2:", G2_min_m_height)
print("Optimized vector x for G2:", G2_optimized_x)
