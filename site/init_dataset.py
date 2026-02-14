
import os
import sys

# Ensure project root is importable (.. from site/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print("Starting dataset initialization...")

try:
    from src.models.dataset_loader import download_dataset_from_hf
    download_dataset_from_hf(
        filename="normalized_dataset_2024_2025.npz",
        filepath="data/dataset"
    )
    print("Dataset initialization complete.")
except Exception as e:
    print(f"Error initializing dataset: {e}")
    sys.exit(1)
