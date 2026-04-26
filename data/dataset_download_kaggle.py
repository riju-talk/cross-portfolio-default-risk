import kagglehub
import os

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(current_dir, "dataset")

# Download latest version to the dataset folder
path = kagglehub.dataset_download("wordsforthewise/lending-club")

print("Path to dataset files:", path)