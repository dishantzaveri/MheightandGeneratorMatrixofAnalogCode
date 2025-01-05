import pickle

# Function to load matrices from a pickle file and print them
def print_matrices_from_pkl(pkl_file):
    # Load the data from the pickle file
    with open(pkl_file, 'rb') as file:
        data = pickle.load(file)

    # Print the structure of the loaded data
    print("Loaded Data Structure:")
    print(data)
    print("-" * 50)
    
    # Check if the data is a list of two matrices or something else
    if isinstance(data, list) and len(data) == 2:
        g1, g2 = data  # Extract the two matrices
        print("Matrix g1:")
        print(g1)
        print("-" * 50)
        print("Matrix g2:")
        print(g2)
    else:
        print("The data does not contain exactly two matrices in a list format.")
        print("Please check the structure of the loaded data.")
        
# Example usage
pkl_file = "Task1dishant"  # Path to your pickle file
print_matrices_from_pkl(pkl_file)
