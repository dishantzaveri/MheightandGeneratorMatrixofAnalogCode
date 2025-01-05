import pickle
import numpy as np

# Function to calculate M-height for a given codeword
def calculate_m_height(c, m):
    abs_values = np.abs(c)
    sorted_values = np.sort(abs_values)[::-1]  # Sort in descending order
    if m < len(sorted_values) and sorted_values[m] != 0:
        return sorted_values[0] / sorted_values[m]
    else:
        return float('inf')

# Function to compute m-heights for x vectors in Task4 with generator matrices
def compute_m_heights(task4_file, generator_matrices_file, output_file):
    try:
        # Load the Task4 file (contains x vectors)
        with open(task4_file, 'rb') as f:
            task4_data = pickle.load(f)
        
        # Load the Task4GeneratorMatrices file (contains generator matrices)
        with open(generator_matrices_file, 'rb') as f:
            generator_matrices_data = pickle.load(f)
        
        # Open the output file to save the results
        with open(output_file, 'w') as output_f:
            # Loop through each key-value pair in Task4 (x vectors)
            for key, x in task4_data.items():
                # Get the generator matrix for the current key
                if key not in generator_matrices_data:
                    print(f"Error: Generator matrix for key {key} not found.")
                    continue
                G = np.array(generator_matrices_data[key]["GeneratorMatrix"])
                m = generator_matrices_data[key]["m"]

                # Ensure x is a numpy array
                x = np.array(x)

                # Calculate the m-height for the current x vector
                m_height = calculate_m_height(np.dot(x, G), m)

                # Print the m-height to terminal
                print(f"Key {key}: m-height = {m_height}")

                # Save the result in the output file
                output_f.write(f"Key {key}: m-height = {m_height}\n")
        
        print(f"m-heights have been saved to {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
task4_file = 'Task4'  # Replace with the actual path to your Task4 pickle file
generator_matrices_file = 'Task4GeneratorMatrices.pkl'  # Replace with the actual path to your Task4GeneratorMatrices file
output_file = 'Task4mheights.txt'  # Output file to save the results

compute_m_heights(task4_file, generator_matrices_file, output_file)
