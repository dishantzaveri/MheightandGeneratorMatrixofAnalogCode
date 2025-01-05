import numpy as np
from scipy.optimize import differential_evolution, minimize
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

# Multi-Scale Exhaustive Random Sampling with dynamically reduced search space
def exhaustive_random_sampling(G, m, iterations=3, num_samples_per_iter=1_000_000, batch_size=50_000, scale_factor=5):
    k, n = G.shape
    best_x = None
    max_m_height = 0

    # Start with a larger range and reduce over time
    for scale in range(1, scale_factor + 1):
        bounds = (-1e10 * scale, 1e10 * scale)  # Expanded range for search space
        for iteration in range(iterations):
            for _ in range(num_samples_per_iter // batch_size):
                x_batch = np.random.uniform(bounds[0], bounds[1], size=(batch_size, k))
                c_batch = np.dot(x_batch, G)
                m_heights = np.array([calculate_m_height(c, m) for c in c_batch])

                if m_heights.max() > max_m_height:
                    max_m_height = m_heights.max()
                    best_x = x_batch[np.argmax(m_heights)]

        # Reduce search space progressively after a few iterations to improve speed
        bounds = (-1e5 * scale, 1e5 * scale)  # Narrowing bounds

    return best_x, max_m_height

# Differential Evolution Optimization with reduced iterations and population size
def optimize_with_de(G, m, maxiter=1000, popsize=30, scale=1e10, recombination=0.7, mutation=(0.6, 1.0)):
    k, n = G.shape

    def objective(x):
        c = np.dot(x, G)
        return -calculate_m_height(c, m)

    bounds = [(-scale, scale)] * k
    try:
        result = differential_evolution(objective, bounds, maxiter=maxiter, popsize=popsize, recombination=recombination, mutation=mutation, disp=False)
        if result.success:
            x_opt = result.x
            m_height = -result.fun
            return x_opt, m_height
        else:
            raise ValueError("Differential evolution optimization failed.")
    except Exception as e:
        raise ValueError(f"Optimization failed with error: {e}")

# Genetic Algorithm with reduced generations and population size
def genetic_algorithm(G, m, generations=500, population_size=200, mutation_rate=0.05, crossover_rate=0.7):
    k, n = G.shape

    # Initialize population with smaller values to optimize search speed
    population = np.random.uniform(-1e10, 1e10, size=(population_size, k))

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

# Gradient-Based Refinement with reduced iterations for faster results
def refine_with_gradient(G, m, x_init):
    def objective(x):
        c = np.dot(x, G)
        return -calculate_m_height(c, m)

    result = minimize(objective, x_init, method='L-BFGS-B', options={'maxiter': 1000})
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

# Task 2 Implementation without Multithreading
def task_2(input_file, output_task2_file, output_text_file):
    data = pickle.load(open(input_file, "rb"))
    results = {}

    for matrix_id, matrix_data in data.items():
        G = np.array(matrix_data["GeneratorMatrix"], dtype=np.float32)
        m = matrix_data["m"]

        print(f"Optimizing Matrix ID: {matrix_id}")
        matrix_id, best_x, max_m_height = multi_stage_optimization((matrix_id, G, m))  # Pass the arguments as a tuple
        if best_x is not None:
            results[matrix_id] = best_x.tolist()

            with open(output_text_file, "a") as output_file:
                output_file.write(f"Matrix ID {matrix_id}, Max m-height: {max_m_height:.6f}, Best x: {best_x.tolist()}\n")
        else:
            with open(output_text_file, "a") as output_file:
                output_file.write(f"Matrix ID {matrix_id}, Optimization failed.\n")

    pickle.dump(results, open(output_task2_file, "wb"))
    print(f"Results saved to {output_task2_file} and {output_text_file}")


# Usage Example
input_file = "Task4GeneratorMatrices.pkl"
output_task2_file = "Task4psotry.pkl"
output_text_file = "Task4psotry.txt"

task_2(input_file, output_task2_file, output_text_file)
