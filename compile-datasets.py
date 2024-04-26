import os

# Directory containing the text files
directory = "./datasets/calling/"

# Output file to compile all text
output_file = "calling-training-dataset.txt"

# Open the output file in append mode
with open(output_file, "a", encoding="utf-8") as output:
    # Traverse the directory
    for root, _, files in os.walk(directory):
        for file in files:
            # Check if the file is a .txt file
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                # Read the content of the file
                with open(file_path, "r", encoding="utf-8") as input_file:
                    # Write the content to the output file
                    output.write(input_file.read())
                    output.write("\n")  # Add a newline between files

print("Compilation complete.")
