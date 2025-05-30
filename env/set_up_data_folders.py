import os

folders = [
    "data/raw",
    "data/processed",
    "data/logs",
    "data/backtests",
    "data/exports",
]

def create_data_directory():
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Created {folder}")

if __name__ == "__main__":
    create_data_directory()