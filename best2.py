import numpy as np
import pickle
import multiprocessing
from datetime import datetime

# Function to calculate M-height for a given codeword
def calculate_m_height(matrix, vector):
    result = np.dot(matrix.T, vector)  # Matrix-vector multiplication
    abs_result = np.abs(result)
    sorted_result = np.sort(abs_result)[::-1]  # Descending order
    return sorted_result[0] / sorted_result[4] if len(sorted_result) >= 5 else 0

# Random Sampling with Extended Iterations and Higher Range
def random_sampling(matrix, k, num_iterations=10_000_000, vector_range=1e15):
    best_m_height = -np.inf
    best_vector = None

    for _ in range(num_iterations):
        vector = np.random.uniform(-vector_range, vector_range, size=k)  # Wider range
        m_height = calculate_m_height(matrix, vector)
        if m_height > best_m_height:
            best_m_height = m_height
            best_vector = vector
    return best_vector, best_m_height

# Perturbation Optimization to Refine Results
def perturb_vector(vector, scale=0.1):
    perturbation = np.random.uniform(-scale, scale, size=vector.shape)
    return vector + perturbation

def perform_optimization(matrix_id, initial_vector, matrix, original_m_height, num_attempts=1_000_000, perturbation_scale=1e12):
    best_vector = np.array(initial_vector)
    best_m_height = calculate_m_height(matrix, best_vector)

    print(f"Optimizing Matrix {matrix_id}... Initial M-height: {best_m_height}")
    for attempt in range(num_attempts):
        new_vector = perturb_vector(best_vector, perturbation_scale)
        new_m_height = calculate_m_height(matrix, new_vector)
        if new_m_height > best_m_height:
            best_m_height = new_m_height
            best_vector = new_vector
            print(f"Matrix {matrix_id} - New best M-height: {best_m_height}")

    return best_m_height, best_vector

# Worker function for multiprocessing
def process_matrix(args):
    matrix_id, data = args
    n = data["n"]
    k = data["k"]
    matrix = np.array(data["GeneratorMatrix"])

    print(f"Processing Matrix ID: {matrix_id}")

    # Random Sampling for Initial Best Vector
    best_vector, best_m_height = random_sampling(matrix, k)

    # Perturbation-Based Optimization for Refinement
    best_m_height, best_vector = perform_optimization(
        matrix_id, best_vector, matrix, best_m_height
    )

    print(f"Matrix ID: {matrix_id}, Best M-height: {best_m_height}")
    return matrix_id, best_vector, best_m_height

# Main function for multiprocessing
def process_all_matrices(input_file, output_pkl_file, output_text_file):
    # Load input matrices
    data = pickle.load(open(input_file, "rb"))
    matrices = list(data.items())

    # Use multiprocessing pool for parallel processing
    num_processes = 48
    with multiprocessing.Pool(num_processes) as pool:
        results = pool.map(process_matrix, matrices)

    # Save results in the specified formats
    with open(output_text_file, "w") as text_file, open(output_pkl_file, "wb") as pkl_file:
        final_dict = {}
        for matrix_id, vector, m_height in results:
            final_dict[int(matrix_id)] = vector.tolist()  # Ensure integer keys
            text_file.write(f"{matrix_id}: {vector.tolist()}, M-height: {m_height:.6f}\n")
        pickle.dump(final_dict, pkl_file)

    print(f"Results saved to {output_pkl_file} and {output_text_file}")

# Run the script
if __name__ == "__main__":
    input_file = "Task2GeneratorMatrices.pkl"
    output_pkl_file = "best2.pkl"
    output_text_file = "best2.txt"
    process_all_matrices(input_file, output_pkl_file, output_text_file)
