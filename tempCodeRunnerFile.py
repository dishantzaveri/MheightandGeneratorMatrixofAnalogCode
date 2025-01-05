# Check if loaded_data is a dictionary
if isinstance(loaded_data, dict):
    print("Data is saved as a dictionary.")
    print("Dictionary keys:", loaded_data.keys())
else:
    print("Data is not saved as a dictionary.")

# Optionally, print the contents to verify
print("Contents of the dictionary:")
for key, value in loaded_data.items():
    print(f"{key}: \n{value}\n")