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

# Multi-Scale Exhaustive Random Sampling with increased iterations and expanded x-vector range
def exhaustive_random_sampling(G, m, iterations=10, num_samples_per_iter=20_000_000, batch_size=100_000, scale_factor=20):
    k, n = G.shape
    best_x = None
    max_m_height = 0

    # Define the expanded range: e^(-100) to e^(100)
    lower_bound = np.exp(-100)
    upper_bound = np.exp(100)

    for scale in range(1, scale_factor + 1):
        bounds = (lower_bound * scale, upper_bound * scale)
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

# Differential Evolution Optimization with multiple starts and expanded x-vector range
def optimize_with_de(G, m, maxiter=5000, popsize=200, scale=1e15, num_starts=5):
    k, n = G.shape

    # Define the expanded range: e^(-100) to e^(100)
    lower_bound = np.exp(-100)
    upper_bound = np.exp(100)

    def objective(x):
        c = np.dot(x, G)
        m_height = calculate_m_height(c, m)
        if np.isinf(m_height):
            return -1e20  # Return a large negative value to prioritize infinite m-height
        return -m_height

    bounds = [(lower_bound, upper_bound)] * k
    best_x, max_m_height = None, 0

    for start in range(num_starts):
        try:
            result = differential_evolution(objective, bounds, maxiter=maxiter, popsize=popsize, workers=-1, disp=False)
            if result.success:
                x_opt = result.x
                m_height = calculate_m_height(np.dot(x_opt, G), m)
                if m_height > max_m_height:
                    best_x, max_m_height = x_opt, m_height
        except Exception as e:
            print(f"Differential Evolution failed on start {start}: {e}")

    return best_x, max_m_height

# Simulated Annealing with expanded x-vector range
def simulated_annealing(G, m, maxiter=1000, temp=1000, cooling_rate=0.95):
    k, n = G.shape

    # Define the expanded range: e^(-100) to e^(100)
    lower_bound = np.exp(-100)
    upper_bound = np.exp(100)

    def objective(x):
        c = np.dot(x, G)
        return -calculate_m_height(c, m)

    # Initial guess for the solution
    x = np.random.uniform(lower_bound, upper_bound, size=k)
    current_m_height = -objective(x)
    best_x, best_m_height = x.copy(), current_m_height

    for iteration in range(maxiter):
        # Generate new candidate by perturbing the current solution
        new_x = x + np.random.normal(0, 0.1, size=k)
        new_x = np.clip(new_x, lower_bound, upper_bound)  # Ensure bounds are respected
        new_m_height = -objective(new_x)

        # Accept the new solution with some probability
        if new_m_height > current_m_height or np.random.rand() < np.exp((new_m_height - current_m_height) / temp):
            x, current_m_height = new_x, new_m_height

        # Update best solution
        if new_m_height > best_m_height:
            best_x, best_m_height = new_x, new_m_height

        # Cool down temperature
        temp *= cooling_rate

    return best_x, best_m_height

# Gradient-Based Refinement with reduced iterations and adaptive learning rate
def refine_with_gradient(G, m, x_init):
    def objective(x):
        c = np.dot(x, G)
        return -calculate_m_height(c, m)

    try:
        result = minimize(objective, x_init, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-6})
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

# Multi-Stage Optimization with Simulated Annealing and Multiple DE starts
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

        # Stage 2: Differential Evolution (Multiple Starts)
        try:
            print(f"Matrix ID {matrix_id} - Starting Differential Evolution")
            de_x, de_m_height = optimize_with_de(G, m)
            if de_x is not None and de_m_height is not None and de_m_height > max_m_height:
                best_x, max_m_height = de_x, de_m_height
        except Exception as e:
            print(f"Matrix ID {matrix_id} - Differential Evolution failed: {e}")

        # Stage 3: Simulated Annealing
        try:
            print(f"Matrix ID {matrix_id} - Starting Simulated Annealing")
            sa_x, sa_m_height = simulated_annealing(G, m)
            if sa_x is not None and sa_m_height is not None and sa_m_height > max_m_height:
                best_x, max_m_height = sa_x, sa_m_height
        except Exception as e:
            print(f"Matrix ID {matrix_id} - Simulated Annealing failed: {e}")

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

# Task 2 Implementation with Multiprocessing
def task_2(input_file, output_task2_file, output_text_file, num_processes=48):
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
                results[matrix_id] = best_x.tolist()
                output_file.write(f"Matrix ID: {matrix_id}, Max m-height: {max_m_height:.6f}, Best x: {best_x.tolist()}\n")
            else:
                output_file.write(f"Matrix ID: {matrix_id}, Optimization failed.\n")

    pickle.dump(results, open(output_task2_file, "wb"))
    print(f"Results saved to {output_task2_file} and {output_text_file}")

# Usage Example
input_file = "Task4GeneratorMatrices.pkl"
output_task2_file = "Task4final.pkl"
output_text_file = "Task4final.txt"

task_2(input_file, output_task2_file, output_text_file)
