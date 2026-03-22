import os

# Define the folder paths
folders = [
    "src/data_ingestion",
    "tests"
]

# Define the file paths
files = [
    "src/__init__.py",
    "src/data_ingestion/__init__.py",
    "src/data_ingestion/statsbomb_fetcher.py",
    "tests/__init__.py",
    "tests/test_statsbomb_fetcher.py",
    ".env",
    "requirements.txt"
]
new_folder = "data/raw"

os.makedirs(new_folder, exist_ok=True)

print("Data folders created successfully!")

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create empty files
for file in files:
    with open(file, 'w') as f:
        pass

print("File structure created successfully!")