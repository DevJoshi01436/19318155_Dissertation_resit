"""
Merge Cresci + Social Honeypot datasets into one combined CSV.

Input files (must exist in data/raw/):
    - cresci_merged.csv
    - social_honeypot_clean.csv

Output file:
    - combined_cresci_honeypot.csv
"""

import pandas as pd
import os

RAW_PATH = "data/raw"
OUTPUT_FILE = os.path.join(RAW_PATH, "combined_cresci_honeypot.csv")


def load_dataset(file_name: str) -> pd.DataFrame:
    """Load a CSV from data/raw safely."""
    path = os.path.join(RAW_PATH, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Missing file: {path}")
    print(f"✓ Loaded: {file_name}")
    return pd.read_csv(path)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize Cresci + Honeypot columns into a unified schema.

    Required output columns:
        id, screen_name, followers_count, friends_count, status_count,
        description, label
    """
    # Rename known variations
    rename_map = {
        "user_id": "id",
        "username": "screen_name",
        "statuses_count": "status_count",
        "tweets": "status_count",
        "bio": "description",
        "profile_description": "description",
        "is_bot": "label",
        "target": "label"
    }

    df = df.rename(columns=rename_map)

    # Ensure all required columns exist
    required = [
        "id", "screen_name", "followers_count", "friends_count",
        "status_count", "description", "label"
    ]

    for col in required:
        if col not in df.columns:
            df[col] = None  # fill missing columns with None

    # Keep only the required columns
    return df[required]


def main():
    print("\n=== Merging Cresci + Honeypot datasets ===\n")

    # 1. Load both datasets
    cresci = load_dataset("cresci_merged.csv")
    honeypot = load_dataset("social_honeypot_clean.csv")

    # 2. Standardize column names
    cresci_std = standardize_columns(cresci)
    honeypot_std = standardize_columns(honeypot)

    print("✓ Standardized both datasets to common schema")

    # 3. Concatenate
    combined = pd.concat([cresci_std, honeypot_std], ignore_index=True)

    print(f"✓ Combined rows: {len(combined):,}")

    # 4. Drop duplicates based on "id"
    before = len(combined)
    combined = combined.drop_duplicates(subset="id")
    after = len(combined)

    print(f"✓ Deduplicated: {before:,} → {after:,} unique users")

    # 5. Save to data/raw
    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Saved combined dataset: {OUTPUT_FILE}")
    print(f"   Total rows: {len(combined):,}\n")


if __name__ == "__main__":
    main()
