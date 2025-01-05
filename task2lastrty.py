import numpy as np
from scipy.optimize import differential_evolution, minimize
import pickle

# Function to calculate m-height for a given codeword
def calculate_m_height(c, m):
    abs_values = np.abs(c)
    sorted_values = np.sort(abs_values)[::-1]  # Sort in descending order
    if m < len(sorted_values) and sorted_values[m] != 0:
        return sorted_values[0] / sorted_values[m]
    else:
        return float('inf')

# Differential Evolution Optimization
def optimize_with_de(G, m):
    k, n = G.shape

    def objective(x):
        c = np.dot(x, G)
        return -calculate_m_height(c, m)

    bounds = [(-1e15, 1e15)] * k  # Extremely wide bounds for better exploration
    try:
        result = differential_evolution(objective, bounds, maxiter=5000, popsize=100, disp=False)
        if result.success:
            x_opt = result.x
            m_height = -result.fun
            return x_opt, m_height
        else:
            raise ValueError("Differential evolution optimization failed.")
    except Exception as e:
        raise ValueError(f"Optimization failed with error: {e}")

# Gradient-Based Refinement
def refine_with_gradient(G, m, x_init):
    def objective(x):
        c = np.dot(x, G)
        return -calculate_m_height(c, m)

    result = minimize(objective, x_init, method='L-BFGS-B', options={'maxiter': 3000})
    if result.success:
        x_opt = result.x
        m_height = -result.fun
        return x_opt, m_height
    else:
        raise ValueError("Gradient refinement failed.")

# Large-Scale Random Sampling
def random_sampling(G, m, num_samples=10_000_000, batch_size=100_000):
    k, n = G.shape
    max_m_height = 0
    best_x = None

    for _ in range(num_samples // batch_size):
        x_batch = np.random.uniform(-1e15, 1e15, size=(batch_size, k))  # Wider range
        c_batch = np.dot(x_batch, G)
        m_heights = np.array([calculate_m_height(c, m) for c in c_batch])

        if m_heights.max() > max_m_height:
            max_m_height = m_heights.max()
            best_x = x_batch[np.argmax(m_heights)]

    return best_x, max_m_height

# Hybrid Optimization Approach
def hybrid_optimization(G, m):
    try:
        # Step 1: Differential Evolution
        best_x, max_m_height = optimize_with_de(G, m)
        print(f"DE Optimization: Max m-height = {max_m_height:.6f}")
    except ValueError as e:
        print(f"Matrix optimization failed with DE: {e}. Switching to fallback...")
        best_x, max_m_height = None, 0

    if best_x is not None:
        try:
            # Step 2: Gradient-Based Refinement
            refined_x, refined_m_height = refine_with_gradient(G, m, best_x)
            if refined_m_height > max_m_height:
                best_x, max_m_height = refined_x, refined_m_height
            print(f"Gradient Refinement: Max m-height = {max_m_height:.6f}")
        except ValueError:
            print("Gradient refinement failed. Proceeding with current best result.")

    if best_x is None or max_m_height == 0:
        # Step 3: Large-Scale Random Sampling
        print("Switching to large-scale random sampling...")
        best_x, max_m_height = random_sampling(G, m)

    return best_x, max_m_height

# Task 2 Implementation
def task_2(input_file, output_task2_file, output_text_file):
    # Load input matrices
    data = pickle.load(open(input_file, "rb"))
    results = {}
    with open(output_text_file, "w") as output_file:
        for matrix_id, matrix_data in data.items():
            n = matrix_data["n"]
            k = matrix_data["k"]
            m = matrix_data["m"]
            G = np.array(matrix_data["GeneratorMatrix"])

            print(f"Processing Matrix ID: {matrix_id} (n={n}, k={k}, m={m})")

            try:
                best_x, max_m_height = hybrid_optimization(G, m)
                results[matrix_id] = best_x
                output_file.write(f"Matrix ID: {matrix_id}, Max m-height: {max_m_height:.6f}\n")
                print(f"Matrix ID: {matrix_id}, Max m-height: {max_m_height:.6f}")
            except Exception as e:
                output_file.write(f"Matrix ID: {matrix_id}, Error: {str(e)}\n")
                print(f"Matrix ID: {matrix_id}, Error: {str(e)}")

    # Save results as pickle
    pickle.dump(results, open(output_task2_file, "wb"))
    print(f"Task 2 results saved to {output_task2_file}")

# Usage Example
input_file = "Task2GeneratorMatrices.pkl"
output_task2_file = "Task2lastry.pkl"
output_text_file = "Task2Output2.txt"

task_2(input_file, output_task2_file, output_text_file)
