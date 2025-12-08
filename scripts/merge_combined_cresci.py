"""
Merge Cresci (cresci_merged.csv) + Social Honeypot (social_honeypot_clean.csv)
into a single combined dataset with a unified schema.

Input (must exist in data/raw/):
    - cresci_merged.csv
    - social_honeypot_clean.csv

Output:
    - combined_cresci_honeypot.csv

Final columns:
    id, screen_name, followers_count, friends_count, status_count, description, label
"""

import os          # for file path handling
import pandas as pd  # for CSV loading and DataFrame operations

# Folder where your raw CSVs live
RAW_PATH = "data/raw"

# Path of the final combined CSV
OUTPUT_FILE = os.path.join(RAW_PATH, "combined_cresci_honeypot.csv")


def load_dataset(file_name: str) -> pd.DataFrame:
    """
    Load a CSV from data/raw safely, with a helpful error if it is missing.
    """
    # Build full path: data/raw/<file_name>
    path = os.path.join(RAW_PATH, file_name)

    # Check if the file actually exists
    if not os.path.exists(path):
        # Raise a clear error instead of a vague pandas error
        raise FileNotFoundError(f"❌ Missing file: {path}")

    # Log what we’re loading
    print(f"✓ Loaded: {file_name}")

    # Read the CSV into a DataFrame and return it
    return pd.read_csv(path)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize Cresci + Honeypot columns into a unified schema.

    Required output columns:
        id, screen_name, followers_count, friends_count, status_count,
        description, label
    """
    # Map alternative column names to our standard names
    rename_map = {
        "user_id": "id",
        "username": "screen_name",
        "statuses_count": "status_count",
        "tweets": "status_count",
        "bio": "description",
        "profile_description": "description",
        "is_bot": "label",
        "target": "label",
    }

    # Apply renaming where those columns exist
    df = df.rename(columns=rename_map)

    # Define the columns we want in the final combined dataset
    required = [
        "id",
        "screen_name",
        "followers_count",
        "friends_count",
        "status_count",
        "description",
        "label",
    ]

    # Ensure every required column exists; if missing, create it filled with None
    for col in required:
        if col not in df.columns:
            df[col] = None

    # Keep only the required columns in the correct order
    df = df[required].copy()

    # Make sure label is numeric (0 or 1) if possible
    df["label"] = pd.to_numeric(df["label"], errors="coerce")

    return df


def main() -> None:
    """Main routine: load both datasets, standardise, combine, deduplicate, save."""
    print("\n=== Merging Cresci + Social Honeypot datasets ===\n")

    # 1. Load both datasets from data/raw/
    cresci = load_dataset("cresci_merged.csv")
    honeypot = load_dataset("social_honeypot_clean.csv")

    # 2. Standardise column names & ensure same schema
    cresci_std = standardize_columns(cresci)
    honeypot_std = standardize_columns(honeypot)

    print("✓ Standardized both datasets to common schema")

    # 3. Concatenate them into one big DataFrame
    combined = pd.concat([cresci_std, honeypot_std], ignore_index=True)

    print(f"✓ Combined rows (before dedupe): {len(combined):,}")

    # 4. Drop duplicate users based on 'id' (if same user appears in both datasets)
    before = len(combined)
    if "id" in combined.columns:
        combined = combined.drop_duplicates(subset="id")
    after = len(combined)

    print(f"✓ Deduplicated on 'id': {before:,} → {after:,} unique users")

    # 5. Ensure data/raw exists (should already, but safe to check)
    os.makedirs(RAW_PATH, exist_ok=True)

    # 6. Save final combined CSV
    combined.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✅ Saved combined dataset: {OUTPUT_FILE}")
    print(f"   Total rows: {len(combined):,}\n")


if __name__ == "__main__":
    main()
