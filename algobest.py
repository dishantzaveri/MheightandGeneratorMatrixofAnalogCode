import numpy as np
import pickle

# Define your matrices G*1 and G*2
G1_star = np.array([
    [91284, -68414, 32408, -98072, -6982, -55851, -42851, -62338, -71407, 61383, -71301],
    [-3059, -86594, -26438, 71015, -82317, 10756, -44480, -16913, 67411, -18620, 28353],
    [-87986, 15134, -64791, -72712, -71010, -61745, 10349, 13168, -56396, 33026, 90859],
    [-46804, 61097, 45454, -60237, 87879, -26699, -32358, 97856, 25755, 21955, 4670],
    [16788, 78713, -41461, 16924, 95319, 80494, 76954, 71910, 7124, 34320, -34427]
])

G2_star = np.array([
    [74345, 26055, -76371, -78971, 72373, 48837, -26227, 67624, 28439, 7537, -93504],
    [-57529, 48854, 40524, 51438, 1741, -15028, -16558, -21470, 90228, -34253, 27349],
    [-48093, 84636, 53923, -78052, -99859, 50447, 75976, -54483, -21822, -88486, -25668],
    [-31088, 55712, -9483, -80483, -63290, -4423, -38689, -87544, -66130, 74197, -82523],
    [80445, 28879, -31231, 42699, -51393, 86029, -72624, -18982, 35462, 36830, -95416],
    [-96653, 2959, 30114, -39748, 49998, -66558, 97849, -56467, -86419, 59184, 17732]
])

# Create a dictionary with keys "setting1" and "setting2"
matrices_dict = {
    "setting1": G1_star,
    "setting2": G2_star
}

# Save the dictionary to Task3.pkl
with open("Task3.pkl", "wb") as f:
    pickle.dump(matrices_dict, f)

print("Matrices saved successfully in 'Task3'.")

import pickle

# File path for Task3.pkl
file_path = "Task3"

# Load the file
try:
    with open(file_path, "rb") as f:
        loaded_data = pickle.load(f)

    # Verify the dictionary structure
    if isinstance(loaded_data, dict) and "setting1" in loaded_data and "setting2" in loaded_data:
        print("Task3 matrices loaded successfully and in proper format!")

        # Print G*1 matrix
        print("\nG*1 Matrix (setting1):\n", loaded_data["setting1"])

        # Print G*2 matrix
        print("\nG*2 Matrix (setting2):\n", loaded_data["setting2"])
    else:
        print("Error: The file does not contain the expected dictionary format with 'setting1' and 'setting2' keys.")
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found. Make sure the file exists and the path is correct.")
except Exception as e:
    print(f"An error occurred while loading the file: {e}")
