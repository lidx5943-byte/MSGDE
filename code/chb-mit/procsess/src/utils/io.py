import numpy as np
from pathlib import Path

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_numpy(arr, path):
    np.save(path, arr)
    print(f"Saved: {path}")

def load_numpy(path):
    return np.load(path, allow_pickle=True)
