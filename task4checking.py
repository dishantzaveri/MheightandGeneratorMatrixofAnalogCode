import pickle
import numpy as np

# Function to check the format of Task4 and print values in the desired format
def check_task4_and_print(pkl_file):
    try:
        # Load the pickle file
        with open(pkl_file, 'rb') as f:
            task4_data = pickle.load(f)
        
        # Check if the data is a dictionary
        if not isinstance(task4_data, dict):
            print(f"Error: The loaded data is not a dictionary.")
            return False
        
        # Loop through each key-value pair in the dictionary
        for key, value in task4_data.items():
            # Check if the key is an integer (or string)
            if not isinstance(key, (int, str)):
                print(f"Error: Key {key} is not an integer or string.")
                return False
            
            # Check if the value is a numpy array
            if not isinstance(value, np.ndarray):
                print(f"Error: The value for key {key} is not a numpy array.")
                return False

            # Print in the desired format
            print(f"{key} : {value.tolist()}")  # Convert numpy array to list for display
        
        return True
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

# Example usage
pkl_file = 'Task4'  # Replace with the actual path to your pickle file
if check_task4_and_print(pkl_file):
    print("The file passed all checks.")
else:
    print("The file failed the checks.")
