import numpy as np
import pickle
from scipy.optimize import minimize

# Function to calculate m-height for a given codeword
def calculate_m_height(c, m):
    abs_values = np.abs(c)
    sorted_values = np.sort(abs_values)[::-1]  # Sort in descending order
    if m < len(sorted_values) and sorted_values[m] != 0:
        return sorted_values[0] / sorted_values[m]
    else:
        return float('inf')

# Evaluate m-height for a matrix using optimization
def evaluate_matrix_with_optimization(G, num_samples=1000):
    k, n = G.shape
    m = 4  # Given m value
    max_m_height = float('inf')

    for _ in range(num_samples):
        # Use linear programming to find x that maximizes the m-height
        def objective(x):
            c = np.dot(x, G)
            return -calculate_m_height(c, m)  # Minimize negative m-height

        bounds = [(-1, 1)] * k
        x0 = np.random.uniform(-1, 1, k)

        result = minimize(objective, x0, bounds=bounds, method='SLSQP')
        if result.success:
            c = np.dot(result.x, G)
            m_height = calculate_m_height(c, m)
            max_m_height = min(max_m_height, m_height)

    return max_m_height

# Genetic algorithm for optimizing matrices
def genetic_algorithm(shape, rank, population_size=50, generations=100, mutation_rate=0.1, num_samples=500):
    def generate_random_matrix():
        G = np.random.randint(-100000, 100001, shape)
        return G if np.linalg.matrix_rank(G) == rank else None

    def mutate(matrix):
        G = matrix.copy()
        num_mutations = max(1, int(mutation_rate * np.prod(G.shape)))
        for _ in range(num_mutations):
            i, j = np.random.randint(0, G.shape[0]), np.random.randint(0, G.shape[1])
            G[i, j] += np.random.randint(-1000, 1001)
        return G

    def crossover(parent1, parent2):
        mask = np.random.rand(*shape) < 0.5
        return np.where(mask, parent1, parent2)

    population = []
    while len(population) < population_size:
        candidate = generate_random_matrix()
        if candidate is not None:
            population.append(candidate)

    for generation in range(generations):
        fitness = [evaluate_matrix_with_optimization(G, num_samples) for G in population]
        sorted_indices = np.argsort(fitness)
        population = [population[i] for i in sorted_indices]
        best_m_height = fitness[sorted_indices[0]]

        print(f"Generation {generation + 1}/{generations}, Best m-height: {best_m_height:.4f}")

        next_population = population[:population_size // 2]
        while len(next_population) < population_size:
            parent1, parent2 = np.random.choice(next_population, 2, replace=False)
            child = crossover(parent1, parent2)
            child = mutate(child)
            if np.linalg.matrix_rank(child) == rank:
                next_population.append(child)

        population = next_population

    return population[0], evaluate_matrix_with_optimization(population[0], num_samples)

# Random exploration to refine matrices
def random_exploration(shape, rank, num_matrices=5000, num_samples=500):
    best_matrix = None
    best_m_height = float('inf')

    for _ in range(num_matrices):
        G = np.random.randint(-100000, 100001, shape)
        if np.linalg.matrix_rank(G) == rank:
            m_height = evaluate_matrix_with_optimization(G, num_samples)
            if m_height < best_m_height:
                best_matrix = G
                best_m_height = m_height

    return best_matrix, best_m_height

# Task 3: Generate the best G1 and G2 matrices
def task_3():
    G1_shape = (5, 11)
    G2_shape = (6, 11)
    G1_rank = 5
    G2_rank = 6
    num_samples = 500  # Samples per matrix for evaluation

    print("Optimizing G1...")
    best_G1, best_G1_m_height = genetic_algorithm(G1_shape, G1_rank)
    best_G1_refined, best_G1_m_height_refined = random_exploration(G1_shape, G1_rank)
    best_G1 = best_G1 if best_G1_m_height < best_G1_m_height_refined else best_G1_refined
    best_G1_m_height = min(best_G1_m_height, best_G1_m_height_refined)

    print("\nOptimizing G2...")
    best_G2, best_G2_m_height = genetic_algorithm(G2_shape, G2_rank)
    best_G2_refined, best_G2_m_height_refined = random_exploration(G2_shape, G2_rank)
    best_G2 = best_G2 if best_G2_m_height < best_G2_m_height_refined else best_G2_refined
    best_G2_m_height = min(best_G2_m_height, best_G2_m_height_refined)

    # Save the results
    matrices = {"setting1": best_G1, "setting2": best_G2}
    with open("Task3_OptimizedMatrices.pkl", "wb") as f:
        pickle.dump(matrices, f)

    print("\nBest G1 Matrix:\n", best_G1)
    print("Minimum m-height for G1:", best_G1_m_height)

    print("\nBest G2 Matrix:\n", best_G2)
    print("Minimum m-height for G2:", best_G2_m_height)

# Run Task 3
task_3()
