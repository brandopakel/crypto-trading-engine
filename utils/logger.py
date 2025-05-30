import os
import pandas as pd

# Increment Folder Code:
#   existing = [f for f in os.listdir(folder) if f.endswith(".csv")]
#   counter = len(existing) + 1

def save_log(df=pd.DataFrame, folder="", base_name = ""):
    os.makedirs(folder, exist_ok=True)

    counter = 1
    while True:
        file_path = os.path.join(folder, f"{base_name}_{counter}.csv")
        if not os.path.exists(file_path):
            break
        counter += 1
    
    df.to_csv(file_path, index=False)
    print(f"\n✅ Saved log to: {file_path}\n")