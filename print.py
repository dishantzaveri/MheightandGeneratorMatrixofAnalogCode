import pickle
import numpy as np

# Function to load and print matrices from Task3 file
def load_and_print_task3(file_path):
    try:
        # Load the file
        with open(file_path, 'rb') as file:
            data = pickle.load(file)

        # Verify the structure
        if not isinstance(data, dict):
            raise ValueError("The file content is not a dictionary.")
        
        if "setting1" not in data or "setting2" not in data:
            raise KeyError("The dictionary does not have the keys 'setting1' and 'setting2'.")

        # Extract matrices
        G_star_1 = data["setting1"]
        G_star_2 = data["setting2"]

        # Print matrices
        print("G*1 Matrix (5x11):")
        print(np.array(G_star_1))

        print("\nG*2 Matrix (6x11):")
        print(np.array(G_star_2))

        # Additional verification
        print("\nMatrix G*1 Dimensions:", np.array(G_star_1).shape)
        print("Matrix G*2 Dimensions:", np.array(G_star_2).shape)

    except Exception as e:
        print("An error occurred while reading the Task3 file:", str(e))

# Specify the path to the Task3 file
file_path = "Task3"  # Ensure this is the correct path to your Task3 file

# Call the function
load_and_print_task3(file_path)
