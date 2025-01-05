import numpy as np
from scipy.optimize import minimize, differential_evolution

# Function to calculate m-height for a given codeword
def calculate_m_height(c, m):
    abs_values = np.abs(c)
    sorted_values = np.sort(abs_values)[::-1]  # Sort in descending order
    if m < len(sorted_values) and sorted_values[m] != 0:
        return sorted_values[0] / sorted_values[m]
    else:
        return float('inf')

# Random sampling for initial broad search
def evaluate_batches(G, m, num_samples=1_000_000, batch_size=10_000):
    k, n = G.shape  # Ensure G is (k, n)
    max_m_height = 0
    best_x = None

    for _ in range(num_samples // batch_size):
        x_batch = np.random.uniform(-1e8, 1e8, size=(batch_size, k))  # Ensure x is (batch_size, k)
        c_batch = np.dot(x_batch, G)  # Correct dot product (batch_size x n)
        m_heights = np.array([calculate_m_height(c, m) for c in c_batch])

        # Track the best m-height and corresponding x
        if m_heights.max() > max_m_height:
            max_m_height = m_heights.max()
            best_x = x_batch[np.argmax(m_heights)]

    return best_x, max_m_height

# Global optimization using differential evolution
def optimize_with_differential_evolution(G, m, bounds):
    def objective(x):
        c = np.dot(x, G)
        return -calculate_m_height(c, m)  # Minimize negative m-height

    result = differential_evolution(objective, bounds, strategy='best1bin', maxiter=1000, tol=1e-7)
    if result.success:
        return result.x, -result.fun
    else:
        raise ValueError("Differential evolution optimization failed.")

# Gradient-based refinement for precise tuning
def refine_with_gradient(G, m, x_init):
    def objective(x):
        c = np.dot(x, G)
        return -calculate_m_height(c, m)  # Minimize negative m-height

    result = minimize(objective, x_init, method='L-BFGS-B', options={'disp': False, 'maxiter': 2000})
    if result.success:
        return result.x, -result.fun
    else:
        return x_init, calculate_m_height(np.dot(x_init, G), m)

# Hybrid approach combining random sampling, differential evolution, and gradient refinement
def find_best_x_hybrid(G, m, num_samples=1_000_000, batch_size=10_000):
    # Step 1: Random sampling for broad exploration
    print("Starting random sampling...")
    best_x, max_m_height = evaluate_batches(G, m, num_samples, batch_size)
    print(f"After random sampling: Best m-height = {max_m_height:.6f}")

    # Step 2: Global optimization using differential evolution
    print("Starting global optimization with differential evolution...")
    bounds = [(-1e10, 1e10)] * G.shape[0]  # Wide bounds for x
    best_x_de, max_m_height_de = optimize_with_differential_evolution(G, m, bounds)
    if max_m_height_de > max_m_height:
        best_x, max_m_height = best_x_de, max_m_height_de
    print(f"After differential evolution: Best m-height = {max_m_height:.6f}")

    # Step 3: Gradient-based refinement for fine-tuning
    print("Starting gradient-based refinement...")
    refined_x, refined_m_height = refine_with_gradient(G, m, best_x)
    if refined_m_height > max_m_height:
        best_x, max_m_height = refined_x, refined_m_height
    print(f"After gradient refinement: Best m-height = {max_m_height:.6f}")

    return best_x, max_m_height

# Input matrix G and m value
G_star_1 = np.array([
    [91284, -68414, 32408, -98072, -6982, -55851, -42851, -62338, -71407, 61383, -71301],
    [-3059, -86594, -26438, 71015, -82317, 10756, -44480, -16913, 67411, -18620, 28353],
    [-87986, 15134, -64791, -72712, -71010, -61745, 10349, 13168, -56396, 33026, 90859],
    [-46804, 61097, 45454, -60237, 87879, -26699, -32358, 97856, 25755, 21955, 4670],
    [16788, 78713, -41461, 16924, 95319, 80494, 76954, 71910, 7124, 34320, -34427]
])  # Ensure correct shape (k, n)

m = 4  # Given m value

# Find the best x vector and its m-height
best_x, max_m_height = find_best_x_hybrid(G_star_1, m)

# Print results
print("\nFinal Results:")
print("Best x vector:", best_x)
print("Maximum m-height:", max_m_height)
