import subprocess

# Define the path to the bash script
script_path = "./data/coco.sh"

# Run the bash script and capture its output
process = subprocess.Popen(["bash", script_path], stdout=subprocess.PIPE)
output, error = process.communicate()

# Print the output
print(output.decode())