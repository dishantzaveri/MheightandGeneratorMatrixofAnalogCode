import pickle
import numpy as np

# Function to create and save Task3 file
def create_task3(file_path):
    # Define the matrices
    G_star_1 = np.array([
        [-2841,  2549, -1228,  3868,  4201,   -21,   460, -1794, -2376,   -71,  1716],
        [-3039,  -949,  4758, -2677,  2725, -1535,  4655, -2644, -4526,  5379,  3774],
        [  253,  4089,  4637, -3391,  -595,  1858,  -611,   118, -1454,  -824,  3229],
        [  875,   905, -1741,   431, -4373,  4503, -2534, -3653,   136, -1026,   224],
        [-2190, -3156, -4712,   112,  -685,  4833,   950,  1126,  4131, -2570, -1429]
    ])

    G_star_2 = np.array([
        [-4692, -2215, -1300, -3181, -2154,  2980, -4832, -1523,    87,   -24,  2246],
        [-1927,  4250, -4595,  3236, -1266,  2646,  3827,  -624,  3136, -4933, -4296],
        [ 4122,   621, -1771,  1097,   212,  4166,  1427,  3999,  2509, -1854,  4002],
        [ 5075,  4292,  4951, -4344,  -884,  3385, -1593,  2575, -3859,  1759,  1862],
        [ 4102, -3328,  4102, -3265,  3607, -4405, -2025, -4733,   964, -1588,  3477],
        [ 3776, -2287,  -888,  3944,  5311, -1105,  -628, -5227,  -671, -2441,   374]
    ])

    # Create dictionary
    matrices = {
        "setting1": G_star_1,
        "setting2": G_star_2
    }

    # Save to file
    with open(file_path, "wb") as file:
        pickle.dump(matrices, file)

    print(f"Task3 file created successfully with G*1 and G*2 matrices.")

# Function to read and display Task3 file in the desired format
def read_and_print_task3(file_path):
    try:
        with open(file_path, "rb") as file:
            data = pickle.load(file)

        # Extract matrices
        G_star_1 = data["setting1"]
        G_star_2 = data["setting2"]

        # Print matrices in the desired format
        print("G*1 Matrix (setting1):")
        for row in G_star_1:
            print(row)

        print("\nG*2 Matrix (setting2):")
        for row in G_star_2:
            print(row)

        # Verifying the shapes of the matrices
        print(f"\nShape of G*1: {G_star_1.shape}")
        print(f"Shape of G*2: {G_star_2.shape}")

    except Exception as e:
        print("An error occurred while reading the file:", e)

# Main Execution
file_path = "Task3"

# Create the Task3 file
create_task3(file_path)

# Read and print the Task3 file
read_and_print_task3(file_path)
