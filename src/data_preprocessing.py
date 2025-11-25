# src/data_preprocessing.py

# Import pandas for working with tabular data (CSV files, DataFrames, etc.)
import pandas as pd


def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load the raw dataset from a CSV file.
    """
    # Try to read the CSV file located at the given path
    try:
        df = pd.read_csv(path)  # read_csv loads the file into a pandas DataFrame
    except FileNotFoundError:   # if the file path is wrong or missing
        # Raise a clearer error message so the user knows where to put the file
        raise FileNotFoundError(f"Dataset not found at {path}. Place your CSV in data/raw/.")
    # Return the loaded DataFrame to the caller
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic cleaning steps:
    - drop duplicate rows
    - reset index
    - handle obvious missing values
    """
    # Make a copy of the input DataFrame so we don't modify it in-place unexpectedly
    df = df.copy()

    # Drop exact duplicate rows to avoid skewing training/testing
    df = df.drop_duplicates()

    # Reset the index after dropping rows so indices go from 0..n-1
    df = df.reset_index(drop=True)

    # If a 'label'-type column exists, drop rows where label is missing
    for label_col in ["label", "is_bot", "bot", "target", "is_fake"]:
        # Check if this potential label column exists in the DataFrame
        if label_col in df.columns:
            # Drop rows with missing labels, because we can't train on unlabeled data
            df = df.dropna(subset=[label_col])
            # Only apply this once, then break out of the loop
            break

    # Return the cleaned DataFrame
    return df
