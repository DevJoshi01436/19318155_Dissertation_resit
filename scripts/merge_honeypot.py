"""
merge_honeypot.py

This script converts the Social Honeypot (ICWSM2011) dataset into a clean CSV
that is compatible with the Cresci 2017 merged dataset and your ML pipeline.
"""

import pandas as pd          # pandas for dataframes and CSV loading
import os                    # os for path joins
import glob                  # glob for flexible file matching


# Path where your honeypot dataset files live.
# This should be the folder that contains:
#   content_polluters
#   content_polluters_followings
#   legitimate_users
#   legitimate_users_followings
#   ...
HONEYPOT_PATH = r"C:\Users\devj0\Desktop\dataset\social_honeypot_icwsm_2011\social_honeypot_icwsm_2011"

# Output CSV file path (we save into data/raw/ inside your project)
OUTPUT_PATH = r"data/raw/social_honeypot_clean.csv"


def find_file(base_name: str) -> str:
    """
    Find a file in HONEYPOT_PATH whose name starts with base_name.
    This handles cases like:
      - content_polluters
      - content_polluters.txt
      - content_polluters.csv
    """
    # Build a glob pattern like ".../content_polluters*"
    pattern = os.path.join(HONEYPOT_PATH, base_name + "*")
    # Find all matching paths
    candidates = glob.glob(pattern)

    # If nothing found, raise a clear error
    if not candidates:
        raise FileNotFoundError(f"No file found matching: {pattern}")

    # Use the first match (there should normally be just one)
    return candidates[0]


def load_honeypot_file(base_name: str, label: int) -> pd.DataFrame:
    """
    Loads a honeypot text file (e.g., 'content_polluters' or 'legitimate_users')
    and converts it to a dataframe with a label column.

    Args:
        base_name (str): base file name without extension
        label (int): 1 for bots, 0 for legitimate

    Returns:
        pd.DataFrame
    """
    # Find the actual file path (with or without extension)
    file_path = find_file(base_name)

    # Read the file, letting pandas guess the delimiter (tab, comma, etc.)
    df = pd.read_csv(file_path, sep=None, engine="python")

    # Add label column
    df["label"] = label

    print(f"Loaded {os.path.basename(file_path)}: {len(df)} rows, label={label}")
    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames Social Honeypot columns to match the Cresci dataset.
    Some columns may not exist, so we create placeholders where needed.
    """
    # Map various possible original column names to your unified ones
    column_map = {
        "id": "id",
        "user_id": "id",
        "userid": "id",
        "username": "screen_name",
        "screen_name": "screen_name",
        "description": "description",
        "bio": "description",
        "followers": "followers_count",
        "followers_count": "followers_count",
        "friends": "friends_count",
        "friends_count": "friends_count",
        "statuses": "status_count",
        "status_count": "status_count",
        "tweets": "status_count",
    }

    # Apply the rename where names match
    df = df.rename(columns=column_map)

    # For any expected column that is missing, create an empty one
    for col in ["id", "screen_name", "followers_count", "friends_count", "status_count", "description"]:
        if col not in df.columns:
            df[col] = None

    # Ensure description is string and replace NaN with empty string
    df["description"] = df["description"].fillna("").astype(str)

    # Return only the columns your pipeline uses (plus label)
    return df[["id", "screen_name", "followers_count", "friends_count", "status_count", "description", "label"]]


def main():
    """Main function to load, clean, merge and save the honeypot dataset."""
    print("\n=== Processing Social Honeypot Dataset ===\n")

    # Load bot (content polluters) and legitimate user files
    bots = load_honeypot_file("content_polluters", label=1)
    real = load_honeypot_file("legitimate_users", label=0)

    # Standardize both to common column names
    bots_clean = standardize_columns(bots)
    real_clean = standardize_columns(real)

    # Combine bots + real users into a single dataframe
    full = pd.concat([bots_clean, real_clean], ignore_index=True)

    # Deduplicate by id (if available), otherwise by screen_name
    before = len(full)
    if "id" in full.columns:
        full = full.drop_duplicates(subset=["id"])
    elif "screen_name" in full.columns:
        full = full.drop_duplicates(subset=["screen_name"])
    after = len(full)

    print(f"\nDeduplicated: {before} -> {after} rows")

    # Save final CSV into your project
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    full.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"\n✅ Saved cleaned dataset: {OUTPUT_PATH}")
    print(f"Rows: {len(full)} | Columns: {list(full.columns)}\n")


if __name__ == "__main__":
    main()
