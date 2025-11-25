# src/utils.py

# Import os so we can work with directories and file paths
import os


def ensure_dirs():
    """
    Make sure standard output directories exist:
    - models/  (for saving trained models later)
    - reports/ (for saving figures or evaluation outputs)
    """
    # List of directories we want to guarantee exist
    dirs = ["models", "reports"]

    # Iterate over required directory names
    for d in dirs:
        # Use os.makedirs with exist_ok=True so it doesn't error if the folder already exists
        os.makedirs(d, exist_ok=True)
