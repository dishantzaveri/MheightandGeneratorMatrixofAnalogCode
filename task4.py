import numpy as np
from scipy.optimize import differential_evolution, minimize
import pickle
import time
from multiprocessing import Pool

# Function to calculate M-height for a given codeword
def calculate_m_height(c, m):
    abs_values = np.abs(c)
    sorted_values = np.sort(abs_values)[::-1]  # Sort in descending order
    if m < len(sorted_values) and sorted_values[m] != 0:
        return sorted_values[0] / sorted_values[m]
    else:
        return float('inf')

# Multi-Scale Exhaustive Random Sampling
def exhaustive_random_sampling(G, m, iterations=10, num_samples_per_iter=10_000_000, batch_size=100_000, scale_factor=20):
    k, n = G.shape
    best_x = None
    max_m_height = 0

    for scale in range(1, scale_factor + 1):
        bounds = (-1e20 * scale, 1e20 * scale)
        for iteration in range(iterations):
            for _ in range(num_samples_per_iter // batch_size):
                x_batch = np.random.uniform(bounds[0], bounds[1], size=(batch_size, k))
                c_batch = np.dot(x_batch, G)
                m_heights = np.array([calculate_m_height(c, m) for c in c_batch])

                if m_heights.max() > max_m_height:
                    max_m_height = m_heights.max()
                    best_x = x_batch[np.argmax(m_heights)]

    return best_x, max_m_height

# Differential Evolution Optimization with additional parameters
def optimize_with_de(G, m, maxiter=5000, popsize=100, scale=1e20, recombination=0.9):
    k, n = G.shape

    def objective(x):
        c = np.dot(x, G)
        return -calculate_m_height(c, m)

    bounds = [(-scale, scale)] * k
    try:
        result = differential_evolution(objective, bounds, maxiter=maxiter, popsize=popsize, recombination=recombination, disp=False)
        if result.success:
            x_opt = result.x
            m_height = -result.fun
            return x_opt, m_height
        else:
            raise ValueError("Differential evolution optimization failed.")
    except Exception as e:
        raise ValueError(f"Optimization failed with error: {e}")

# Genetic Algorithm for Enhanced Global Search
def genetic_algorithm(G, m, generations=1000, population_size=200, mutation_rate=0.05, crossover_rate=0.7):
    k, n = G.shape

    # Initialize population
    population = np.random.uniform(-1e20, 1e20, size=(population_size, k))

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
output_task2_file = "Task4Optimized.pkl"
output_text_file = "Task4Optimized.txt"

task_2(input_file, output_task2_file, output_text_file)
