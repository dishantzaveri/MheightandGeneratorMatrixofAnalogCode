import pickle
import numpy as np

# Function to load and print Task2 results
def read_and_print_task2_results(task2_file):
    try:
        # Load the Task2 results
        task2_results = pickle.load(open(task2_file, "rb"))
        
        # Print results
        print(f"Contents of {task2_file}:\n")
        for matrix_id, x_vector in task2_results.items():
            print(f"Matrix ID: {matrix_id}")
            print(f"x Vector: {x_vector}\n")
            print("-" * 50)
    except Exception as e:
        print(f"Error reading {task2_file}: {e}")

# Usage example
task2_file = "Task2demo.pkl"  # Path to the Task2 file
read_and_print_task2_results(task2_file)
