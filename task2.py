import numpy as np
import pickle
from scipy.optimize import differential_evolution

# Function to calculate m-height for a given codeword
def calculate_m_height(c, m):
    abs_values = np.abs(c)
    sorted_values = np.sort(abs_values)[::-1]  # Sort in descending order
    if m < len(sorted_values) and sorted_values[m] != 0:
        return sorted_values[0] / sorted_values[m]
    else:
        return float('inf')

# Normalize matrix to improve numerical stability
def normalize_matrix(G):
    return G / np.max(np.abs(G), axis=1, keepdims=True)

# Differential evolution optimization
def optimize_with_differential_evolution(G, m, bounds, popsize=100, maxiter=3000):
    def objective(x):
        c = np.dot(x, G)
        return -calculate_m_height(c, m)  # Maximize m-height by minimizing negative m-height

    try:
        result = differential_evolution(
            objective,
            bounds,
            strategy="randtobest1bin",
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-7,
            mutation=(0.5, 1),
            recombination=0.7,
            polish=False,  # Avoid expensive local optimization
        )
        if result.success:
            return result.x, -result.fun
        else:
            raise ValueError("Differential evolution optimization failed.")
    except Exception as e:
        raise RuntimeError(f"Optimization failed with error: {e}")

# Fallback random sampling with adaptive range
def random_sampling_fallback(G, m, num_samples=500_000, range_scale=1e6):
    k, _ = G.shape
    best_x = None
    max_m_height = 0

    for _ in range(num_samples):
        x = np.random.uniform(-range_scale, range_scale, k)
        c = np.dot(x, G)
        m_height = calculate_m_height(c, m)
        if m_height > max_m_height:
            max_m_height = m_height
            best_x = x
    return best_x, max_m_height

# Hybrid optimization
def find_best_x_hybrid(G, m, bounds, fallback_range_scale=1e6):
    G_normalized = normalize_matrix(G)
    k, n = G_normalized.shape

    try:
        # Step 1: Differential evolution optimization
        best_x, max_m_height = optimize_with_differential_evolution(G_normalized, m, bounds)
    except RuntimeError as e:
        print(f"Matrix optimization failed with DE: {e}. Switching to fallback...")
        # Step 2: Random sampling fallback
        best_x, max_m_height = random_sampling_fallback(G_normalized, m, range_scale=fallback_range_scale)

    return best_x, max_m_height

# Task 2 processing
def process_task2(task2_input_file, output_task2_file, output_text_file):
    # Load input matrices
    with open(task2_input_file, "rb") as f:
        matrices = pickle.load(f)

    task2_results = {}
    text_results = []

    for matrix_id, details in matrices.items():
        G = details["GeneratorMatrix"]
        m = details["m"]
        k, n = details["k"], details["n"]

        print(f"Processing Matrix ID: {matrix_id} (n={n}, k={k}, m={m})")
        bounds = [(-1e8, 1e8)] * k  # Adaptive bounds for DE

        try:
            # Find the best x vector and maximum m-height
            best_x, max_m_height = find_best_x_hybrid(G, m, bounds)
            task2_results[matrix_id] = best_x

            # Write to text output
            text_results.append(f"Matrix ID: {matrix_id}\n")
            text_results.append(f"Maximum m-height: {max_m_height:.6f}\n")
            text_results.append(f"Best x vector: {best_x.tolist()}\n")
            text_results.append("=" * 40 + "\n")

        except Exception as e:
            print(f"Matrix ID: {matrix_id}, Error: {e}")
            text_results.append(f"Matrix ID: {matrix_id}, Error: {e}\n")
            text_results.append("=" * 40 + "\n")

    # Save x vectors to Task2 file
    with open(output_task2_file, "wb") as f:
        pickle.dump(task2_results, f)
    print(f"Task2 file saved as {output_task2_file}")

    # Save results to Task2Output.txt
    with open(output_text_file, "w") as f:
        f.writelines(text_results)
    print(f"Task2Output text file saved as {output_text_file}")

# Example usage
task2_input_file = "Task2GeneratorMatrices.pkl"  # Input file containing matrices
output_task2_file = "Task2"                 # Output file for x vectors
output_text_file = "Task2Output.txt"        # Text output for m-heights and x vectors

process_task2(task2_input_file, output_task2_file, output_text_file)
