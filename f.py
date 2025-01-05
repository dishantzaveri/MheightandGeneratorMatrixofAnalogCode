import pickle
import numpy as np

def verify_and_fix_task2_pkl(file_path):
    try:
        # Load the pickle file
        data = pickle.load(open(file_path, "rb"))

        # Check if it's a dictionary
        if not isinstance(data, dict):
            print("Error: The file does not contain a dictionary.")
            return

        is_valid = True
        corrected_data = {}

        for key, value in data.items():
            # Check if the key is an integer and convert it to a string
            if isinstance(key, int):
                corrected_key = str(key)
            elif isinstance(key, str) and key.isdigit():
                corrected_key = key
            else:
                print(f"Error: Key '{key}' is not valid (neither an integer nor a string representing an integer).")
                is_valid = False
                continue

            # Check if the value is a numpy array
            if not isinstance(value, np.ndarray):
                print(f"Error: Value for key '{corrected_key}' is not a numpy array.")
                is_valid = False
            else:
                # Convert the numpy array to a plain list
                corrected_data[int(corrected_key)] = value.tolist()

        # Print the dictionary in the desired format
        for key, vector in corrected_data.items():
            print(f"{key} : {vector}")

        if is_valid:
            print("\nThe file is in the correct format!")
        else:
            print("\nThe file is NOT in the correct format but was corrected where possible.")

        # Save the corrected data back to the file
        with open("Corrected_Task2.pkl", "wb") as corrected_file:
            pickle.dump(corrected_data, corrected_file)
            print("Corrected data saved to 'Corrected_Task2.pkl'.")

    except Exception as e:
        print(f"Error: Unable to load and process the file. Details: {e}")

# Replace 'Task2lastry.pkl' with your file path
file_path = "Task2lastry.pkl"
verify_and_fix_task2_pkl(file_path)
