import numpy as np
from scipy.optimize import differential_evolution, minimize
import pickle
import time
from multiprocessing import Pool, cpu_count

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

    bounds = [(-1e20, 1e20)] * k  # Very wide bounds for higher exploration
    try:
        result = differential_evolution(objective, bounds, maxiter=10000, popsize=200, disp=False)
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

    result = minimize(objective, x_init, method='L-BFGS-B', options={'maxiter': 5000})
    if result.success:
        x_opt = result.x
        m_height = -result.fun
        return x_opt, m_height
    else:
        raise ValueError("Gradient refinement failed.")

# Large-Scale Random Sampling
def random_sampling(G, m, num_samples=50_000_000, batch_size=1_000_000):
    k, n = G.shape
    max_m_height = 0
    best_x = None

    for _ in range(num_samples // batch_size):
        x_batch = np.random.uniform(-1e20, 1e20, size=(batch_size, k))
        c_batch = np.dot(x_batch, G)
        m_heights = np.array([calculate_m_height(c, m) for c in c_batch])

        if m_heights.max() > max_m_height:
            max_m_height = m_heights.max()
            best_x = x_batch[np.argmax(m_heights)]
            print(f"    Random Sampling: New max m-height = {max_m_height:.6f}, x = {best_x}")

    return best_x, max_m_height

# Hybrid Optimization Approach
def hybrid_optimization(matrix_data):
    matrix_id, matrix_details = matrix_data
    n = matrix_details["n"]
    k = matrix_details["k"]
    m = matrix_details["m"]
    G = np.array(matrix_details["GeneratorMatrix"])

    print(f"\nProcessing Matrix ID: {matrix_id} (n={n}, k={k}, m={m})")
    start_time = time.time()

    best_x, max_m_height = None, 0

    # Stage 1: Differential Evolution
    try:
        print("  Stage 1: Differential Evolution...")
        best_x, max_m_height = optimize_with_de(G, m)
        print(f"    DE Optimization: Max m-height = {max_m_height:.6f}, x = {best_x}")
    except ValueError as e:
        print(f"    DE Optimization failed: {e}")

    # Stage 2: Gradient-Based Refinement
    if best_x is not None:
        try:
            print("  Stage 2: Gradient-Based Refinement...")
            refined_x, refined_m_height = refine_with_gradient(G, m, best_x)
            if refined_m_height > max_m_height:
                best_x, max_m_height = refined_x, refined_m_height
            print(f"    Gradient Refinement: Max m-height = {max_m_height:.6f}, x = {best_x}")
        except ValueError:
            print("    Gradient refinement failed.")

    # Stage 3: Large-Scale Random Sampling
    if best_x is None or max_m_height == 0:
        print("  Stage 3: Large-Scale Random Sampling...")
        best_x, max_m_height = random_sampling(G, m)
        print(f"    Random Sampling Final: Max m-height = {max_m_height:.6f}, x = {best_x}")

    elapsed_time = time.time() - start_time
    print(f"  Matrix ID {matrix_id} processed in {elapsed_time:.2f} seconds")
    return matrix_id, best_x, max_m_height

# Function to process matrices in parallel
def process_matrices_in_parallel(input_file, output_task2_file, output_text_file):
    # Load input matrices
    data = pickle.load(open(input_file, "rb"))
    matrix_list = list(data.items())
    results = {}

    with open(output_text_file, "w") as output_file:
        # Use multiprocessing Pool to process matrices in parallel
        with Pool(processes=48) as pool:  # Adjust number of processes to available cores
            results_list = pool.map(hybrid_optimization, matrix_list)

        # Save results and write to text file
        for matrix_id, best_x, max_m_height in results_list:
            if best_x is not None:
                results[matrix_id] = best_x
                output_file.write(f"Matrix ID: {matrix_id}, Max m-height: {max_m_height:.6f}\n")
                print(f"Matrix ID: {matrix_id}, Max m-height: {max_m_height:.6f}, x = {best_x}")
            else:
                output_file.write(f"Matrix ID: {matrix_id}, Error: Optimization failed.\n")
                print(f"Matrix ID: {matrix_id}, Error: Optimization failed.")

    # Save results as pickle
    pickle.dump(results, open(output_task2_file, "wb"))
    print(f"\nTask 2 results saved to {output_task2_file}")

# Usage Example
input_file = "Task2GeneratorMatrices.pkl"
output_task2_file = "Task2parallel.pkl"
output_text_file = "Task2parallel.txt"

process_matrices_in_parallel(input_file, output_task2_file, output_text_file)
