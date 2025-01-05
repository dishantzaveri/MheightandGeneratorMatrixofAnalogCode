import numpy as np
import pickle

# Define your matrices G1 and G2

G1 = np.array([
    [-2841,  2549, -1228,  3868,  4201,   -21,   460, -1794, -2376,   -71,  1716],
    [-3039,  -949,  4758, -2677,  2725, -1535,  4655, -2644, -4526,  5379,  3774],
    [  253,  4089,  4637, -3391,  -595,  1858,  -611,   118, -1454,  -824,  3229],
    [  875,   905, -1741,   431, -4373,  4503, -2534, -3653,   136, -1026,   224],
    [-2190, -3156, -4712,   112,  -685,  4833,   950,  1126,  4131, -2570, -1429]
])


G2 = np.array([
    [-4692, -2215, -1300, -3181, -2154,  2980, -4832, -1523,    87,   -24,  2246],
    [-1927,  4250, -4595,  3236, -1266,  2646,  3827,  -624,  3136, -4933, -4296],
    [ 4122,   621, -1771,  1097,   212,  4166,  1427,  3999,  2509, -1854,  4002],
    [ 5075,  4292,  4951, -4344,  -884,  3385, -1593,  2575, -3859,  1759,  1862],
    [ 4102, -3328,  4102, -3265,  3607, -4405, -2025, -4733,   964, -1588,  3477],
    [ 3776, -2287,  -888,  3944,  5311, -1105,  -628, -5227,  -671, -2441,   374]
])

# Create a dictionary with the matrices
matrices = {
    "setting1": G1,
    "setting2": G2
}

# Save the dictionary to a .pkl file named "Task1.pkl"
with open("Task1", "wb") as f:
    pickle.dump(matrices, f)

print("Matrices saved successfully to 'Task1.pkl'")



import pickle

# Load the .pkl file
with open("Task1", "rb") as f:
    loaded_data = pickle.load(f)

# Check if loaded_data is a dictionary
if isinstance(loaded_data, dict):
    print("Data is saved as a dictionary.")
    print("Dictionary keys:", loaded_data.keys())
else:
    print("Data is not saved as a dictionary.")

# Optionally, print the contents to verify
print("Contents of the dictionary:")
for key, value in loaded_data.items():
    print(f"{key}: \n{value}\n")

# import numpy as np
# import pickle

# # Load matrices from the .pkl file
# with open("Task1", "rb") as f:
#     matrices = pickle.load(f)

# # Define the range limits
# min_value, max_value = -100000, 100000

# # Check if matrices only contain integers within the specified range
# def check_integer_range(matrix, min_value, max_value):
#     return np.all((matrix >= min_value) & (matrix <= max_value)) and np.issubdtype(matrix.dtype, np.integer)

# # Check if matrices have full rank
# def check_full_rank(matrix, expected_rank):
#     return np.linalg.matrix_rank(matrix) == expected_rank

# # Perform checks on G1
# G1 = matrices["setting1"]
# G1_in_range = check_integer_range(G1, min_value, max_value)
# G1_full_rank = check_full_rank(G1, 5)

# # Perform checks on G2
# G2 = matrices["setting2"]
# G2_in_range = check_integer_range(G2, min_value, max_value)
# G2_full_rank = check_full_rank(G2, 6)

# # Print results
# print("G1 integer and range check:", G1_in_range)
# print("G1 full rank check:", G1_full_rank)
# print("G2 integer and range check:", G2_in_range)
# print("G2 full rank check:", G2_full_rank)

# # Optional: Display matrices if checks fail for further investigation
# if not G1_in_range or not G1_full_rank:
#     print("G1 matrix:\n", G1)
# if not G2_in_range or not G2_full_rank:
#     print("G2 matrix:\n", G2)

# import numpy as np
# from scipy.optimize import minimize
# import pickle

# # Load matrices from the .pkl file
# with open("Task1.pkl", "rb") as f:
#     matrices = pickle.load(f)

# # Function to calculate the m-height of a codeword c
# def calculate_m_height(c, m):
#     abs_values = np.abs(c)
#     sorted_values = np.sort(abs_values)[::-1]  # Sort in descending order
#     if m < len(sorted_values) and sorted_values[m] != 0:
#         return sorted_values[0] / sorted_values[m]  # Ratio of largest to m-th largest element
#     elif sorted_values[m] == 0:
#         return float('inf')  # m-height is infinite if m-th largest is zero
#     else:
#         return 0  # If codeword is zero vector, m-height is zero

# # Function to find the minimum m-height for a given generator matrix G
# def min_m_height_for_matrix(G, m):
#     k = G.shape[0]  # Number of rows in G

#     def objective(x):
#         c = np.dot(x, G)
#         return calculate_m_height(c, m)  # Minimize m-height directly

#     x0 = np.random.uniform(-1, 1, k)  # Initial guess for x
#     result = minimize(objective, x0, method='SLSQP', bounds=[(-1, 1)] * k)
    
#     if result.success:
#         return result.fun, result.x  # Return minimum m-height and the corresponding vector x
#     else:
#         return float('inf'), None

# # Set m value for m-height calculation (as per your task requirements)
# m = 4

# # Calculate minimum m-height for G1 and G2
# G1 = matrices["setting1"]
# G2 = matrices["setting2"]

# G1_min_m_height, G1_optimal_x = min_m_height_for_matrix(G1, m)
# G2_min_m_height, G2_optimal_x = min_m_height_for_matrix(G2, m)

# # Print results
# print("Minimum m-height for G1:", G1_min_m_height)
# print("Optimal vector x for G1:", G1_optimal_x)
# print("\nMinimum m-height for G2:", G2_min_m_height)
# print("Optimal vector x for G2:", G2_optimal_x)