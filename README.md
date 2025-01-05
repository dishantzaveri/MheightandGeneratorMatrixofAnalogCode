# m-Height and Generator Matrix Optimization

## Problem Statement

The increasing complexity of machine learning applications and analog computing systems necessitates the development of efficient algorithms to handle large-scale vector-matrix operations reliably. Analog Error-Correcting Codes (ECCs) have emerged as a promising solution to improve the robustness and performance of analog systems, particularly in scenarios requiring extensive vector-matrix computations.

Analog ECCs use the concept of a **Generator Matrix (G)**, which transforms input vectors into analog codewords. These codewords represent vectors in an n-dimensional space, with properties crucial for ensuring error resilience in analog systems. A key metric in Analog ECCs is the **m-height**, analogous to the minimum distance in traditional digital ECCs. The m-height measures the error correction capability and reliability of the code.

The optimization of the m-height is critical for enhancing the reliability and efficiency of analog computing systems, including in-memory computing architectures and machine learning frameworks. This project focuses on two core problems:

1. **Maximizing the m-height**: Finding codewords that maximize the m-height for a given generator matrix \( G \), ensuring improved error correction.
2. **Minimizing the m-height**: Designing generator matrices that minimize the maximum m-height, optimizing the robustness of the analog code.

To address these problems, the project leverages a **multi-stage optimization framework** integrating:
- **Exhaustive Sampling**: For exploring the solution space broadly.
- **Evolutionary Strategies**: Such as Differential Evolution and Genetic Algorithms for global optimization.
- **Gradient-Based Methods**: For precise local refinement.

This hybrid approach ensures a balance between global exploration and local optimization, targeting near-optimal solutions efficiently.

# m-Height Optimization in Analog ECCs

## Overview

This repository presents a comprehensive framework for optimizing the **m-height** of Analog Error-Correcting Codes (ECCs). The m-height metric, critical for analog computing systems, measures the error-correction capability and robustness of vector-matrix operations. This project introduces a hybrid optimization framework that combines exhaustive sampling, evolutionary strategies, and gradient-based refinement to tackle the challenges of m-height optimization effectively.

---

## Multi-Stage Optimization Framework

The hybrid framework integrates multiple optimization methods, leveraging their strengths to provide near-optimal solutions:

### 1. Exhaustive Random Sampling
- Explores the solution space by generating random codewords and evaluating their m-heights.
- Ensures broad exploration but may miss global optima due to its stochastic nature.

### 2. Differential Evolution (DE)
- A population-based optimization method that applies mutation, crossover, and selection iteratively.
- Balances exploration and exploitation, improving upon random sampling.
- Often converges to local optima but performs better than purely random strategies.

### 3. Linear Programming (LP)
- Models the m-height optimization as a linear programming problem.
- Effective for linear constraints but struggles with the inherent non-linear nature of m-height computations.

### 4. Gradient-Based Refinement
- Refines solutions obtained from other stages using gradient-based optimization techniques (e.g., L-BFGS-B).
- Converges to local optima efficiently but relies heavily on the quality of the initial solution.

### 5. Hybrid Multi-Stage Process
- Combines all the above methods:
  - **Broad exploration**: Exhaustive Random Sampling.
  - **Iterative improvement**: Differential Evolution and Linear Programming.
  - **Precision refinement**: Gradient-Based Refinement.
- Capable of identifying extreme cases with infinite m-heights through intelligent adjustments to solution vectors.

---

## Methodology

### Inputs
- **`G`**: The generator matrix defining the analog code.
- **`m`**: Target index for m-height computation.
- **Optimization Parameters**:
  - Iteration counts, population sizes, mutation rates, etc., for each algorithm.

### Outputs
- **`best_x`**: Solution vector achieving the highest m-height.
- **`max_mh`**: Maximum m-height value found.

---

## Algorithm Descriptions

### 1. Exhaustive Random Sampling
- **Goal**: Broadly explore the solution space by generating random vectors.
- **Process**:
  1. Generate random solution vectors \(x\).
  2. Compute corresponding codewords \(c = x \cdot G\).
  3. Evaluate m-heights and track the maximum.
- **Optimizations**:
  - Parallelization for faster sampling.
  - Early stopping when high m-height values are detected.

### 2. Differential Evolution
- **Goal**: Improve solutions iteratively using evolutionary principles.
- **Process**:
  1. Initialize a population of solution vectors.
  2. Apply mutation and crossover to generate new solutions.
  3. Select the best solutions based on m-height evaluations.
- **Optimizations**:
  - Adaptive population sizes and mutation strategies for faster convergence.

### 3. Linear Programming
- **Goal**: Formulate m-height optimization as a linear programming problem.
- **Process**:
  1. Define an objective function and constraints based on \(G\) and \(m\).
  2. Solve using LP solvers (e.g., `scipy.optimize.linprog`).
- **Challenges**:
  - Limited effectiveness due to the non-linear characteristics of m-height computation.

### 4. Gradient-Based Refinement
- **Goal**: Refine solutions using local optimization methods.
- **Process**:
  1. Use a gradient-based optimizer (e.g., L-BFGS-B) to maximize m-height.
  2. Start from a high-quality initial solution obtained from earlier stages.
- **Optimizations**:
  - Preconditioning for faster convergence.
  - Adaptive learning rates.

### 5. Multi-Stage Optimization
- Combines all the above techniques in the following sequence:
  1. Exhaustive Random Sampling.
  2. Differential Evolution.
  3. Linear Programming (optional).
  4. Gradient-Based Refinement.
- **Special Mechanism**: Captures infinite m-heights by intelligently setting components of the solution vector to zero.

---

## Implementation

### Core Functions
- **`calculate_m_height(c, m)`**: Computes the m-height for a given codeword.
- **`exhaustive_random_sampling(G, m)`**: Implements exhaustive random sampling.
- **`optimize_with_de(G, m)`**: Uses differential evolution for m-height optimization.
- **`lp_optimization(G, m)`**: Attempts linear programming-based optimization.
- **`refine_with_gradient(G, m, x_init)`**: Refines solutions using gradient-based methods.
- **`multi_stage_optimization(args)`**: Integrates all stages into a unified process.

### Task Implementation
The `task_implementation` function performs the following steps:
1. **Input**:
   - Reads generator matrices from a file (e.g., `Task4GeneratorMatrices.pkl`).
2. **Processing**:
   - Applies the multi-stage optimization framework to each generator matrix.
3. **Output**:
   - Saves the optimized vectors and their corresponding m-heights.

---

## Results and Performance

### Achievements
- **Infinite m-heights**: Successfully identified cases where infinite m-heights were achievable by modifying solution vectors.
- **Efficiency**: The hybrid approach consistently achieved high m-height values across runs.
- **Comparison**:
  - Outperformed traditional methods like Particle Swarm Optimization (PSO) and Linear Programming (LP).

### Time Complexity
- **Exhaustive Random Sampling**: \(O(\text{iterations} \cdot \text{samples} \cdot (kn + n \log n))\)
- **Differential Evolution**: \(O(\text{generations} \cdot \text{population size} \cdot (kn + n \log n))\)
- **Gradient Refinement**: \(O(\text{iterations} \cdot (kn + n \log n))\)
- **Hybrid Framework**: Aggregates the complexities of individual methods.

---

## Dependencies

- Python 3.8+
- Libraries:
  - NumPy
  - SciPy
  - Pickle
  - Multiprocessing

Install dependencies:
```bash
pip install numpy scipy



