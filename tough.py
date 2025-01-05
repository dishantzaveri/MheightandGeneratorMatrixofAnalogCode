import numpy as np
from scipy.optimize import differential_evolution, minimize
import pickle
import time

# Function to calculate m-height for a given codeword
def calculate_m_height(c, m):
    abs_values = np.abs(c)
    sorted_values = np.sort(abs_values)[::-1]  # Sort in descending order
    if m < len(sorted_values) and sorted_values[m] != 0:
        return sorted_values[0] / sorted_values[m]
    else:
        return float('inf')

# Multi-Scale Exhaustive Random Sampling
def exhaustive_random_sampling(G, m, iterations=5, num_samples_per_iter=5_000_000, batch_size=50_000, scale_factor=10):
    k, n = G.shape
    best_x = None
    max_m_height = 0

    for scale in range(1, scale_factor + 1):
        bounds = (-1e10 * scale, 1e10 * scale)
        print(f"Random Sampling at Scale {scale}: Range = {bounds}")
        for iteration in range(iterations):
            print(f"  Iteration {iteration + 1}/{iterations}...")
            for _ in range(num_samples_per_iter // batch_size):
                x_batch = np.random.uniform(bounds[0], bounds[1], size=(batch_size, k))
                c_batch = np.dot(x_batch, G)
                m_heights = np.array([calculate_m_height(c, m) for c in c_batch])

                if m_heights.max() > max_m_height:
                    max_m_height = m_heights.max()
                    best_x = x_batch[np.argmax(m_heights)]
                    print(f"    New max m-height found: {max_m_height:.6f}")

    return best_x, max_m_height

# Differential Evolution Optimization
def optimize_with_de(G, m, maxiter=3000, popsize=50, scale=1e12):
    k, n = G.shape

    def objective(x):
        c = np.dot(x, G)
        return -calculate_m_height(c, m)

    bounds = [(-scale, scale)] * k
    print(f"DE Optimization: Bounds = {bounds[0]}")
    try:
        result = differential_evolution(objective, bounds, maxiter=maxiter, popsize=popsize, disp=True)
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

# Multi-Stage Optimization with Adaptive Search
def multi_stage_optimization(G, m, exhaustive_iters=3, de_scale=1e12):
    # Stage 1: Exhaustive Random Sampling
    print("\nStage 1: Exhaustive Random Sampling...")
    best_x, max_m_height = exhaustive_random_sampling(G, m, iterations=exhaustive_iters)

    # Stage 2: Differential Evolution
    print("\nStage 2: Differential Evolution Optimization...")
    try:
        de_x, de_m_height = optimize_with_de(G, m, scale=de_scale)
        if de_m_height > max_m_height:
            best_x, max_m_height = de_x, de_m_height
            print(f"New max m-height from DE: {max_m_height:.6f}")
    except Exception as e:
        print(f"DE Optimization failed: {e}")

    # Stage 3: Gradient-Based Refinement
    print("\nStage 3: Gradient-Based Refinement...")
    try:
        refined_x, refined_m_height = refine_with_gradient(G, m, best_x)
        if refined_m_height > max_m_height:
            best_x, max_m_height = refined_x, refined_m_height
            print(f"New max m-height from Gradient Refinement: {max_m_height:.6f}")
    except Exception as e:
        print(f"Gradient refinement failed: {e}")

    return best_x, max_m_height

# Task 2 Implementation
def task_2(input_file, output_task2_file, output_text_file):
    data = pickle.load(open(input_file, "rb"))
    results = {}
    with open(output_text_file, "w") as output_file:
        for matrix_id, matrix_data in data.items():
            n = matrix_data["n"]
            k = matrix_data["k"]
            m = matrix_data["m"]
            G = np.array(matrix_data["GeneratorMatrix"])

            print(f"\nProcessing Matrix ID: {matrix_id} (n={n}, k={k}, m={m})")
            start_time = time.time()

            try:
                best_x, max_m_height = multi_stage_optimization(G, m)
                results[matrix_id] = best_x
                output_file.write(f"Matrix ID: {matrix_id}, Max m-height: {max_m_height:.6f}\n")
                print(f"Matrix ID: {matrix_id}, Max m-height: {max_m_height:.6f}")
            except Exception as e:
                output_file.write(f"Matrix ID: {matrix_id}, Error: {str(e)}\n")
                print(f"Matrix ID: {matrix_id}, Error: {str(e)}")

            elapsed_time = time.time() - start_time
            print(f"Elapsed time for Matrix ID {matrix_id}: {elapsed_time:.2f} seconds\n")

    # Save results as pickle
    pickle.dump(results, open(output_task2_file, "wb"))
    print(f"Task 2 results saved to {output_task2_file}")

# Usage Example
input_file = "Task2GeneratorMatrices.pkl"
output_task2_file = "Task2f.pkl"
output_text_file = "Task2Outputf.txt"

task_2(input_file, output_task2_file, output_text_file)
