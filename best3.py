import numpy as np
import pickle
import multiprocessing
from datetime import datetime
from scipy.optimize import differential_evolution, minimize

# Function to calculate M-height
def calculate_m_height(matrix, vector):
    result = np.dot(matrix.T, vector)
    abs_result = np.abs(result)
    sorted_result = np.sort(abs_result)[::-1]
    return sorted_result[0] / sorted_result[4] if len(sorted_result) >= 5 else 0

# Random Sampling with Adaptive Range
def adaptive_random_sampling(matrix, k, initial_range=1e15, scale_factor=10, iterations=10, num_samples=5_000_000, batch_size=100_000):
    best_m_height = -np.inf
    best_vector = None

    for iteration in range(iterations):
        current_range = initial_range * (scale_factor ** iteration)
        print(f"Random Sampling Iteration {iteration + 1}/{iterations}, Range: ±{current_range:.2e}")
        for _ in range(num_samples // batch_size):
            x_batch = np.random.uniform(-current_range, current_range, size=(batch_size, k))
            m_heights = np.array([calculate_m_height(matrix, vector) for vector in x_batch])

            if m_heights.max() > best_m_height:
                best_m_height = m_heights.max()
                best_vector = x_batch[np.argmax(m_heights)]
                print(f"  New Best M-height: {best_m_height:.6f}")

    return best_vector, best_m_height

# Differential Evolution Optimization
def optimize_with_de(matrix, k, maxiter=5000, popsize=50, vector_range=1e18):
    def objective(x):
        return -calculate_m_height(matrix, x)

    bounds = [(-vector_range, vector_range)] * k
    print(f"Starting DE Optimization with bounds ±{vector_range:.2e}...")
    result = differential_evolution(objective, bounds, maxiter=maxiter, popsize=popsize, disp=True)

    if result.success:
        return result.x, -result.fun
    else:
        raise ValueError("Differential evolution optimization failed.")

# Gradient-Based Refinement
def refine_with_gradient(matrix, initial_vector):
    def objective(x):
        return -calculate_m_height(matrix, x)

    result = minimize(objective, initial_vector, method='L-BFGS-B', options={'maxiter': 5000})
    if result.success:
        return result.x, -result.fun
    else:
        raise ValueError("Gradient refinement failed.")

# Hybrid Optimization
def hybrid_optimization(matrix, k, adaptive_iterations=5, num_samples=5_000_000):
    print("\nStarting Adaptive Random Sampling...")
    vector, m_height = adaptive_random_sampling(matrix, k, iterations=adaptive_iterations, num_samples=num_samples)

    print(f"Adaptive Random Sampling: Best M-height = {m_height:.6f}")
    try:
        print("\nStarting Differential Evolution...")
        de_vector, de_m_height = optimize_with_de(matrix, k)
        if de_m_height > m_height:
            vector, m_height = de_vector, de_m_height
            print(f"Differential Evolution: Best M-height = {m_height:.6f}")
    except Exception as e:
        print(f"DE failed: {e}")

    try:
        print("\nStarting Gradient Refinement...")
        refined_vector, refined_m_height = refine_with_gradient(matrix, vector)
        if refined_m_height > m_height:
            vector, m_height = refined_vector, refined_m_height
            print(f"Gradient Refinement: Best M-height = {m_height:.6f}")
    except Exception as e:
        print(f"Gradient refinement failed: {e}")

    return vector, m_height

# Worker function for multiprocessing
def process_matrix(args):
    matrix_id, data = args
    n = data["n"]
    k = data["k"]
    matrix = np.array(data["GeneratorMatrix"])

    print(f"\nProcessing Matrix ID: {matrix_id}...")
    best_vector, best_m_height = hybrid_optimization(matrix, k)
    print(f"Matrix ID: {matrix_id}, Best M-height: {best_m_height:.6f}")
    return matrix_id, best_vector, best_m_height

# Main function for multiprocessing
def process_all_matrices(input_file, output_pkl_file, output_text_file):
    data = pickle.load(open(input_file, "rb"))
    matrices = list(data.items())

    num_processes = 48  # Number of processes for parallel execution
    with multiprocessing.Pool(num_processes) as pool:
        results = pool.map(process_matrix, matrices)

    with open(output_text_file, "w") as text_file, open(output_pkl_file, "wb") as pkl_file:
        final_dict = {}
        for matrix_id, vector, m_height in results:
            final_dict[int(matrix_id)] = vector.tolist()
            text_file.write(f"{matrix_id}: {vector.tolist()}, M-height: {m_height:.6f}\n")
        pickle.dump(final_dict, pkl_file)

    print(f"Results saved to {output_pkl_file} and {output_text_file}")

# Run the script
if __name__ == "__main__":
    input_file = "Task2GeneratorMatrices.pkl"
    output_pkl_file = "best3.pkl"
    output_text_file = "best3.txt"
    process_all_matrices(input_file, output_pkl_file, output_text_file)
