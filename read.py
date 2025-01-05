import pickle

# Load the .pkl file
with open("Task2GeneratorMatrices.pkl", "rb") as f:
    data = pickle.load(f)

# Open a text file to save the output
with open("output.txt", "w") as output_file:
    # Display and save information about each matrix
    for key, matrix_info in data.items():
        output_file.write(f"\nMatrix ID: {key}\n")
        output_file.write(f"n: {matrix_info['n']}\n")
        output_file.write(f"k: {matrix_info['k']}\n")
        output_file.write(f"m: {matrix_info['m']}\n")
        output_file.write("Generator Matrix:\n")
        output_file.write(f"{matrix_info['GeneratorMatrix']}\n")
        output_file.write("\n" + "="*40 + "\n")  # Separator for readability

print("Output saved to 'output.txt'")
