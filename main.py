# main.py
"""
Main entry point for the Fake Profile Detection pipeline.

This script:
1. Selects a dataset inside data/raw/
2. Loads & cleans the dataset
3. Runs cross-validated evaluation (NO leakage)
4. Prints accuracy, precision, recall, and F1 (mean ± std)
5. Optionally plots ROC curves and PCA projection
"""

# ------------------------------- Imports ----------------------------------

from pprint import pprint              # Useful for debugging (optional)
import glob                            # Used to auto-detect CSV datasets inside data/raw/

# Local project imports
from src.data_preprocessing import load_raw_data, clean_data   # Loading + cleaning data
from src.model_training import (                               # Core ML + visualisations
    cv_evaluate,
    plot_roc_curves,
    plot_pca_projection,
)
from src.utils import ensure_dirs                              # Ensures required folders exist

# Pipeline Banner
print("== fake-profile pipeline v2 (no-leakage, cross-validated) ==")


# ------------------------------- Dataset Auto-Detector -------------------------------

def pick_dataset() -> str:
    """
    Automatically picks a dataset CSV from data/raw/.

    Returns:
        str: Path to the *first* CSV found in sorted alphabetical order.

    Raises:
        FileNotFoundError: If no dataset is found.
    """

    # Find every CSV file inside data/raw/
    candidates = sorted(glob.glob("data/raw/*.csv"))

    # If no CSVs found, stop the program with a human-friendly message
    if not candidates:
        raise FileNotFoundError(
            "No CSV found in data/raw/. "
            "Add a dataset such as cresci_merged.csv or social_honeypot_clean.csv."
        )

    # Return the first CSV in alphabetical order (this keeps behaviour consistent)
    return candidates[0]


# ------------------------------- Main Pipeline -------------------------------

def main() -> None:
    """
    Controls the full Fake Profile Detection pipeline:

        - Ensures required directories exist
        - Chooses dataset (auto or manual)
        - Loads the dataset
        - Cleans the dataset (NaN, dedupe)
        - Runs cross-validation
        - Prints results
        - Optionally plots ROC curves & PCA projection
    """

    # Ensure required output directories exist
    ensure_dirs()

    # -------- Dataset selection --------
    # Manual: currently using the combined Cresci + Honeypot dataset
    csv_path = "data/raw/combined_cresci_honeypot.csv"

    # If you ever want auto-pick again, just replace with:
    # csv_path = pick_dataset()

    # Or choose a specific one:
    # csv_path = "data/raw/cresci_merged.csv"
    # csv_path = "data/raw/social_honeypot_clean.csv"

    print(f"Using dataset: {csv_path}")

    # -------- Load dataset --------
    df = load_raw_data(csv_path)      # Read CSV into DataFrame
    df = clean_data(df)               # Deduplicate, fix missing values, enforce typing

    print(f"Loaded rows: {len(df):,} | Columns: {len(df.columns)}")

    # -------- Cross-validated evaluation (NO leakage) --------
    summaries = cv_evaluate(df, label_col="label")

    # -------- Print final evaluation results --------
    print("\n==== 5-fold CV (mean ± std) ====")

    for model_name, metrics in summaries.items():
        # Format: "accuracy: 0.900 ± 0.004, precision: 0.920 ± 0.005, ..."
        line = ", ".join(
            f"{metric}: {mean:.3f} ± {std:.3f}"
            for metric, (mean, std) in metrics.items()
        )
        print(f"{model_name}: {line}")

    # -------- Optional Hold-out Evaluation --------
    # from src.model_training import holdout_evaluate
    # holdout_evaluate(df, label_col="label")

    # -------- ROC curves (LogReg, RF, XGB) --------
    # This will open a matplotlib window with ROC curves for all three models.
    plot_roc_curves(df, label_col="label")

    # -------- PCA projection (numeric features only) --------
    # This will open a matplotlib window with a 2D PCA scatter plot coloured by label.
    plot_pca_projection(df, label_col="label")


# ------------------------------- Program Entry Point -------------------------------

if __name__ == "__main__":
    main()
