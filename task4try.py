import numpy as np
from scipy.optimize import differential_evolution, minimize
import pickle
from multiprocessing import Pool

# Function to calculate M-height for a given codeword
def calculate_m_height(c, m):
    abs_values = np.abs(c)
    sorted_values = np.sort(abs_values)[::-1]  # Sort in descending order
    if m < len(sorted_values) and sorted_values[m] != 0:
        return sorted_values[0] / sorted_values[m]
    else:
        return float('inf')  # Returning infinity if condition met

# Multi-Scale Exhaustive Random Sampling with larger search space
def exhaustive_random_sampling(G, m, iterations=50, num_samples_per_iter=100_000_000, batch_size=500_000, scale_factor=100):
    k, n = G.shape
    best_x = None
    max_m_height = 0

    for scale in range(1, scale_factor + 1):
        bounds = (-1e30 * scale, 1e30 * scale)
        for iteration in range(iterations):
            for _ in range(num_samples_per_iter // batch_size):
                x_batch = np.random.uniform(bounds[0], bounds[1], size=(batch_size, k))
                c_batch = np.dot(x_batch, G)
                m_heights = np.array([calculate_m_height(c, m) for c in c_batch])

                if m_heights.max() > max_m_height:
                    max_m_height = m_heights.max()
                    best_x = x_batch[np.argmax(m_heights)]

    return best_x, max_m_height

# Differential Evolution Optimization with very high iterations
def optimize_with_de(G, m, maxiter=100000, popsize=500, scale=1e30, recombination=0.9, mutation=(0.7, 1.0)):
    k, n = G.shape

    def objective(x):
        c = np.dot(x, G)
        return -calculate_m_height(c, m)

    bounds = [(-scale, scale)] * k
    try:
        result = differential_evolution(objective, bounds, maxiter=maxiter, popsize=popsize, recombination=recombination, mutation=mutation, disp=True)
        if result.success:
            x_opt = result.x
            m_height = -result.fun
            return x_opt, m_height
        else:
            raise ValueError("Differential evolution optimization failed.")
    except Exception as e:
        raise ValueError(f"Optimization failed with error: {e}")

# Genetic Algorithm for Enhanced Global Search with larger population and mutation rate
def genetic_algorithm(G, m, generations=2000, population_size=500, mutation_rate=0.1, crossover_rate=0.8):
    k, n = G.shape

    # Initialize population
    population = np.random.uniform(-1e30, 1e30, size=(population_size, k))

    def fitness(x):
        c = np.dot(x, G)
        return -calculate_m_height(c, m)

    def crossover(parent1, parent2):
        point = np.random.randint(1, k)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2

    def mutate(individual):
        mutation = np.random.uniform(-1e5, 1e5, size=k)
        return individual + mutation

    best_x = None
    best_m_height = 0
    for generation in range(generations):
        fitness_values = np.array([fitness(individual) for individual in population])
        best_idx = np.argmax(fitness_values)
        if fitness_values[best_idx] > best_m_height:
            best_m_height = fitness_values[best_idx]
            best_x = population[best_idx]

        next_population = []
        while len(next_population) < population_size:
            parents = population[np.random.choice(population_size, 2, p=fitness_values/fitness_values.sum())]
            if np.random.rand() < crossover_rate:
                child1, child2 = crossover(parents[0], parents[1])
                next_population.extend([child1, child2])
            else:
                next_population.extend(parents)

        population = np.array([mutate(ind) if np.random.rand() < mutation_rate else ind for ind in next_population])

    return best_x, best_m_height

# Gradient-Based Refinement with precision and iterations
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

# Multi-Stage Optimization with Adaptive Search and High Precision
def multi_stage_optimization(args):
    matrix_id, G, m = args
    try:
        # Stage 1: Exhaustive Random Sampling
        best_x, max_m_height = exhaustive_random_sampling(G, m)

        # Stage 2: Differential Evolution
        try:
            de_x, de_m_height = optimize_with_de(G, m)
            if de_m_height > max_m_height:
                best_x, max_m_height = de_x, de_m_height
        except Exception as e:
            print(f"Matrix ID {matrix_id} - DE Optimization failed: {e}")

        # Stage 3: Genetic Algorithm
        try:
            ga_x, ga_m_height = genetic_algorithm(G, m)
            if ga_m_height > max_m_height:
                best_x, max_m_height = ga_x, ga_m_height
        except Exception as e:
            print(f"Matrix ID {matrix_id} - GA Optimization failed: {e}")

        # Stage 4: Gradient-Based Refinement
        try:
            refined_x, refined_m_height = refine_with_gradient(G, m, best_x)
            if refined_m_height > max_m_height:
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
output_task2_file = "Task4Optimized.pkl"
output_text_file = "Task4Optimized.txt"

task_2(input_file, output_task2_file, output_text_file)
