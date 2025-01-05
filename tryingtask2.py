import numpy as np

# Function to calculate m-height for a given codeword
def calculate_m_height(c, m=4):
    abs_values = np.abs(c)
    sorted_values = np.sort(abs_values)[::-1]  # Sort in descending order
    if m < len(sorted_values) and sorted_values[m] != 0:
        return sorted_values[0] / sorted_values[m]
    else:
        return float('inf')

# Provided G_star_1 matrix
G_star_1 = np.array([
    [91284, -68414, 32408, -98072, -6982, -55851, -42851, -62338, -71407, 61383, -71301],
    [-3059, -86594, -26438, 71015, -82317, 10756, -44480, -16913, 67411, -18620, 28353],
    [-87986, 15134, -64791, -72712, -71010, -61745, 10349, 13168, -56396, 33026, 90859],
    [-46804, 61097, 45454, -60237, 87879, -26699, -32358, 97856, 25755, 21955, 4670],
    [16788, 78713, -41461, 16924, 95319, 80494, 76954, 71910, 7124, 34320, -34427]
])

# Provided x vector
x = np.array([
    -35020578.57816721, 52849685.90092874, 33244931.14253846,
    29578240.38537534, -81037498.35450645, -3609464.0318093,
    37455752.87909681, 30433912.32308179, 55124824.14798382,
    -44315846.49059241, 45975090.09603113
])

# Calculate the codeword
c = np.dot(x, G_star_1)

# Calculate the m-height
m_height = calculate_m_height(c, m=4)

# Print the results
print("Codeword (c):", c)
print("M-height:", m_height)
