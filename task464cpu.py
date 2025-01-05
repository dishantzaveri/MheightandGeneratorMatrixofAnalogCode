import numpy as np
from scipy.optimize import differential_evolution, minimize, linprog
import pickle
from multiprocessing import Pool
import time

# Function to calculate M-height for a given codeword
def calculate_m_height(c, m):
    abs_values = np.abs(c)
    sorted_values = np.sort(abs_values)[::-1]  # Sort in descending order
    if m < len(sorted_values) and sorted_values[m] != 0:
        return sorted_values[0] / sorted_values[m]
    else:
        return float('inf')

# Multi-Scale Exhaustive Random Sampling
def exhaustive_random_sampling(G, m, iterations=10, num_samples_per_iter=10_000_000, batch_size=50_000, scale_factor=20):
    k, n = G.shape
    best_x = None
    max_m_height = 0

    for scale in range(1, scale_factor + 1):
        bounds = (-1e10 * scale, 1e10 * scale)
        for iteration in range(iterations):
            for _ in range(num_samples_per_iter // batch_size):
                x_batch = np.random.uniform(bounds[0], bounds[1], size=(batch_size, k))
                c_batch = np.dot(x_batch, G)
                m_heights = np.array([calculate_m_height(c, m) for c in c_batch])

                if np.isinf(m_heights).any():
                    idx = np.where(np.isinf(m_heights))[0][0]
                    return x_batch[idx], float('inf')

                if m_heights.max() > max_m_height:
                    max_m_height = m_heights.max()
                    best_x = x_batch[np.argmax(m_heights)]

    return best_x, max_m_height

# Differential Evolution Optimization
def optimize_with_de(G, m, maxiter=2000, popsize=128, scale=1e15):
    k, n = G.shape

    def objective(x):
        c = np.dot(x, G)
        m_height = calculate_m_height(c, m)
        if np.isinf(m_height):
            return -1e20  # Return a large negative value to prioritize infinite m-height
        return -m_height

    bounds = [(-scale, scale)] * k
    try:
        result = differential_evolution(objective, bounds, maxiter=maxiter, popsize=popsize, workers=-1, disp=False)
        if result.success:
            x_opt = result.x
            m_height = calculate_m_height(np.dot(x_opt, G), m)
            return x_opt, m_height
        else:
            print("Differential Evolution did not converge.")
            return None, None
    except Exception as e:
        print(f"Differential Evolution failed: {e}")
        return None, None

# Linear Programming Optimization
def lp_optimization(G, m):
    k, n = G.shape

    # Objective: Maximize the difference between the largest and m-th largest absolute values
    c = np.zeros(k)
    A_eq = G
    b_eq = np.ones(n)

    # Bounds for the variables
    bounds = [(-1e12, 1e12) for _ in range(k)]

    try:
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if result.success:
            x_opt = result.x
            c_opt = np.dot(x_opt, G)
            m_height = calculate_m_height(c_opt, m)
            return x_opt, m_height
        else:
            print("LP Optimization did not converge.")
            return None, None
    except Exception as e:
        print(f"LP Optimization failed: {e}")
        return None, None

# Gradient-Based Refinement
def refine_with_gradient(G, m, x_init):
    def objective(x):
        c = np.dot(x, G)
        m_height = calculate_m_height(c, m)
        if np.isinf(m_height):
            return -1e20  # Prioritize infinite m-height
        return -m_height

    try:
        result = minimize(objective, x_init, method='L-BFGS-B', options={'maxiter': 1000})
        if result.success:
            x_opt = result.x
            m_height = calculate_m_height(np.dot(x_opt, G), m)
            return x_opt, m_height
        else:
            print("Gradient Refinement did not converge.")
            return None, None
    except Exception as e:
        print(f"Gradient refinement error: {e}")
        return None, None

# Multi-Stage Optimization
def multi_stage_optimization(args):
    matrix_id, G, m = args
    best_x, max_m_height = None, 0

    try:
        # Stage 1: Exhaustive Random Sampling
        try:
            print(f"Matrix ID {matrix_id} - Starting Exhaustive Random Sampling")
            random_x, random_m_height = exhaustive_random_sampling(G, m)
            if random_x is not None and random_m_height is not None:
                best_x, max_m_height = random_x, random_m_height
        except Exception as e:
            print(f"Matrix ID {matrix_id} - Exhaustive Random Sampling failed: {e}")

        # Stage 2: Differential Evolution
        try:
            print(f"Matrix ID {matrix_id} - Starting Differential Evolution")
            de_x, de_m_height = optimize_with_de(G, m)
            if de_x is not None and de_m_height is not None and de_m_height > max_m_height:
                best_x, max_m_height = de_x, de_m_height
        except Exception as e:
            print(f"Matrix ID {matrix_id} - Differential Evolution failed: {e}")

        # Stage 3: Linear Programming Optimization
        try:
            print(f"Matrix ID {matrix_id} - Starting LP Optimization")
            lp_x, lp_m_height = lp_optimization(G, m)
            if lp_x is not None and lp_m_height is not None and lp_m_height > max_m_height:
                best_x, max_m_height = lp_x, lp_m_height
        except Exception as e:
            print(f"Matrix ID {matrix_id} - LP Optimization failed: {e}")

        # Stage 4: Gradient-Based Refinement
        try:
            if best_x is not None:
                print(f"Matrix ID {matrix_id} - Starting Gradient Refinement")
                refined_x, refined_m_height = refine_with_gradient(G, m, best_x)
                if refined_x is not None and refined_m_height is not None and refined_m_height > max_m_height:
                    best_x, max_m_height = refined_x, refined_m_height
        except Exception as e:
            print(f"Matrix ID {matrix_id} - Gradient refinement failed: {e}")

        return matrix_id, best_x, max_m_height
    except Exception as e:
        print(f"Matrix ID {matrix_id} - Error during optimization: {e}")
        return matrix_id, None, None

# Task Implementation with Multiprocessing
def task_implementation(input_file, output_task_file, output_text_file, num_processes=64):
    data = pickle.load(open(input_file, "rb"))
    results = {}

    # Prepare arguments for multiprocessing
    args = [(matrix_id, np.array(matrix_data["GeneratorMatrix"]), matrix_data["m"])
            for matrix_id, matrix_data in data.items()]

    # Start multiprocessing
    with Pool(processes=num_processes) as pool:
        results_list = pool.map(multi_stage_optimization, args)

    # Save results
    with open(output_text_file, "w") as output_file:
        for matrix_id, best_x, max_m_height in results_list:
            if best_x is not None:
                results[matrix_id] = np.array(best_x)  # Ensure NumPy array
                output_file.write(f"Matrix ID: {matrix_id}, Max m-height: {max_m_height}, Best x: {best_x.tolist()}\n")
            else:
                output_file.write(f"Matrix ID: {matrix_id}, Optimization failed.\n")

    pickle.dump(results, open(output_task_file, "wb"))
    print(f"Results saved to {output_task_file} and {output_text_file}")

# Usage Example
input_file = "Task4GeneratorMatrices.pkl"
output_task_file = "Task4best.pkl"
output_text_file = "Task4best.txt"

task_implementation(input_file, output_task_file, output_text_file)
