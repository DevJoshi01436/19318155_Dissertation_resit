r"""
Convert the original Social Honeypot TXT files into a clean CSV
that is compatible with the Cresci 2017 merged dataset.

Input (original dataset folder – change this path if needed):
    C:\Users\devj0\Desktop\dataset\social_honeypot_icwsm_2011\social_honeypot_icwsm_2011

We only use:
    - content_polluters.txt      → label = 1 (fake/bot)
    - legitimate_users.txt       → label = 0 (genuine)

TXT format from the dataset README:
    UserID \t CreatedAt \t CollectedAt \t NumberOfFollowings
           \t NumberOfFollowers \t NumberOfTweets
           \t LengthOfScreenName \t LengthOfDescriptionInUserProfile

Output:
    data/raw/social_honeypot_clean.csv
    with columns: id, screen_name, followers_count,
                  friends_count, status_count, description, label
"""

import os                     # for path handling
import pandas as pd           # for tables / CSV reading & writing

# ------------------------------------------------------------------
# 1) POINT THIS TO YOUR HONEYPOT FOLDER
# ------------------------------------------------------------------
# The folder that contains content_polluters.txt and legitimate_users.txt.
# >>> If you move the dataset somewhere else, UPDATE THIS PATH. <<<
HONEYPOT_DIR = (
    r"C:\Users\devj0\Desktop\dataset\social_honeypot_icwsm_2011"
    r"\social_honeypot_icwsm_2011"
)

# Where to save the cleaned CSV (inside your project)
RAW_DIR = "data/raw"
OUTPUT_CSV = os.path.join(RAW_DIR, "social_honeypot_clean.csv")


def load_honeypot_txt(file_name: str, label: int) -> pd.DataFrame:
    """
    Load one of the TXT files (content_polluters or legitimate_users)
    and convert it into our standard schema.

    Args:
        file_name (str): base file name, e.g. "content_polluters.txt"
        label (int): 1 for bots, 0 for genuine

    Returns:
        pd.DataFrame with columns:
            id, screen_name, followers_count, friends_count,
            status_count, description, label
    """
    # Build the full path to the TXT file
    path = os.path.join(HONEYPOT_DIR, file_name)

    if not os.path.exists(path):
        # Fail fast if the file isn’t there
        raise FileNotFoundError(f"❌ Could not find file: {path}")

    print(f"✓ Loading honeypot file: {path}")

    # Column names from the official README
    col_names = [
        "UserID",
        "CreatedAt",
        "CollectedAt",
        "NumberOfFollowings",
        "NumberOfFollowers",
        "NumberOfTweets",
        "LengthOfScreenName",
        "LengthOfDescriptionInUserProfile",
    ]

    # Read a tab-separated file with NO header row
    df_raw = pd.read_csv(
        path,
        sep="\t",          # values are separated by tab characters
        header=None,       # there is no header line in the file
        names=col_names,   # assign our own column names
        encoding="utf-8",  # safe default
        engine="python",   # robust parser for messy text
    )

    # Map the columns into our unified schema.
    # We do not have the real screen_name or description text here,
    # only their lengths, so we leave those as empty / None.
    df_std = pd.DataFrame(
        {
            "id": df_raw["UserID"],
            "screen_name": None,                         # unknown in this dataset
            "followers_count": df_raw["NumberOfFollowers"],
            "friends_count": df_raw["NumberOfFollowings"],
            "status_count": df_raw["NumberOfTweets"],
            "description": None,                         # only length is provided
            "label": label,                              # 1 = bot, 0 = genuine
        }
    )

    print(
        f"  -> rows loaded: {len(df_std):,} | "
        f"label = {label} ({'bots' if label == 1 else 'genuine'})"
    )

    return df_std


def main() -> None:
    """Main function: load both TXT files, combine, deduplicate, save CSV."""
    print("\n=== Building social_honeypot_clean.csv from TXT files ===\n")

    # 1) Load bots (content polluters)
    bots = load_honeypot_txt("content_polluters.txt", label=1)

    # 2) Load genuine users
    legit = load_honeypot_txt("legitimate_users.txt", label=0)

    # 3) Combine them into one table
    combined = pd.concat([bots, legit], ignore_index=True)
    print(f"\n✓ Combined rows (before dedupe): {len(combined):,}")

    # 4) Drop duplicate user IDs if any
    before = len(combined)
    combined = combined.drop_duplicates(subset="id")
    after = len(combined)
    print(f"✓ Deduplicated on 'id': {before:,} → {after:,} unique users")

    # 5) Ensure output folder exists
    os.makedirs(RAW_DIR, exist_ok=True)

    # 6) Save to CSV
    combined.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Saved cleaned honeypot CSV to: {OUTPUT_CSV}")
    print(f"   Final row count: {len(combined):,}\n")


if __name__ == "__main__":
    main()
