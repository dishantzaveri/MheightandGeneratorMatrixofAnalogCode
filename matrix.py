import numpy as np
import pickle

def design_best_generator_matrices():
    # Define parameters
    n = 11  # Number of columns (fixed for both matrices)
    k1, k2 = 5, 6  # Number of rows for G1 and G2 respectively
    std_dev = 0.2  # Standard deviation for controlled random values

    # Generate G1 (5x11) and G2 (6x11) with controlled random values
    G1 = np.random.normal(0, std_dev, (k1, n))
    G2 = np.random.normal(0, std_dev, (k2, n))

    # Introduce row dependencies in G1 and G2 to reduce variability
    for i in range(1, k1):
        G1[i] = 0.8 * G1[i] + 0.2 * G1[i-1]
    
    for i in range(1, k2):
        G2[i] = 0.8 * G2[i] + 0.2 * G2[i-1]

    # Print matrices for inspection
    print("Generated Matrix G1 (5x11):\n", G1)
    print("\nGenerated Matrix G2 (6x11):\n", G2)

    # Save matrices in a dictionary
    matrices = {"setting1": G1, "setting2": G2}

    # Save the dictionary as a pickle file named "Task1"
    with open("Task1", "wb") as f:
        pickle.dump(matrices, f)

    return matrices

# Run the function to create, display, and save matrices
matrices = design_best_generator_matrices()
