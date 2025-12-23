import kagglehub
import shutil
import os

print("Download starting...")

# Download latest version
path = kagglehub.dataset_download("ananthu017/emotion-detection-fer")

print("Path to dataset files:", path)

# Define target destination
target_dir = os.path.join(os.getcwd(), "dataset")

# Move the downloaded files to our local 'dataset' folder
print(f"Moving dataset to {target_dir}...")

# Remove existing poor dataset if it exists
if os.path.exists(target_dir):
    shutil.rmtree(target_dir)

# Copy the new dataset
shutil.copytree(path, target_dir)

print("Dataset setup complete!")
