# scripts/merge_cresci.py

# Import OS/path utilities, ZIP handling, globbing, and a text buffer wrapper
import os, io, glob, zipfile
# Import pandas for CSV I/O and dataframe manipulation
import pandas as pd

# ---------------------- USER CONFIG: EDIT THESE IF YOUR PATHS DIFFER ----------------------

DATA_PATH = r"C:\Users\devj0\Desktop\dataset\cresci-2017.csv\datasets_full.csv"  # <- folder that contains items like 'genuine_accounts.csv' (zipped folder) etc.
OUT_PATH  = r"C:\Users\devj0\Desktop\AI_FakeProfile_Detection_scaffold\AI_FakeProfile_Detection\data\raw\cresci_merged.csv"  # <- where to save merged file in your project

# Define the standard Cresci-2017 group names we want to ingest (humans + several bot families)
WANTED = {
    "genuine_accounts",         # humans (label 0)
    "fake_followers",           # bots (label 1)
    "social_spambots_1",        # bots
    "social_spambots_2",        # bots
    "social_spambots_3",        # bots
    "traditional_spambots_1",   # bots
    "traditional_spambots_2",   # bots
    "traditional_spambots_3",   # bots
    "traditional_spambots_4",   # bots
}

# ---------------------- LOW-LEVEL HELPERS ----------------------

def _read_csv_plain(path: str) -> pd.DataFrame:
    """Read a regular CSV file on disk (UTF-8, skip bad lines)."""
    return pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")  # load CSV, ignore malformed lines

def _read_csv_from_zip(zip_path: str) -> pd.DataFrame:
    """Open a .zip (Windows may display as '.csv'), then read users*.csv or the first *.csv inside."""
    with zipfile.ZipFile(zip_path, "r") as zf:                                    # open the compressed file
        names = [n for n in zf.namelist() if not n.startswith("__MACOSX/")]       # filter out MacOS junk
        cand = [n for n in names if n.lower().endswith(".csv") and "users" in n.lower()]  # prefer users*.csv
        if not cand:                                                               # if no users*.csv found
            cand = [n for n in names if n.lower().endswith(".csv")]               # pick first csv
        if not cand:                                                               # if no csv at all
            raise FileNotFoundError(f"No CSV inside zip: {zip_path}")             # fail explicitly
        with zf.open(cand[0]) as fh:                                              # open the chosen csv entry
            wrapper = io.TextIOWrapper(fh, encoding="utf-8", errors="ignore")     # wrap as text stream
            return pd.read_csv(wrapper)                                           # read via pandas

def _read_csv_from_folder(folder: str) -> pd.DataFrame | None:
    """Search a folder (and one level deeper) for users*.csv or any *.csv; also handle zip files found inside."""
    # Build search patterns for direct csv, nested csv, and nested users*.csv
    patterns = [
        os.path.join(folder, "*users*.csv"),      # direct users*.csv
        os.path.join(folder, "*.csv"),            # any csv directly
        os.path.join(folder, "*", "*users*.csv"), # one-level-deep users*.csv
        os.path.join(folder, "*", "*.csv"),       # one-level-deep any csv
    ]
    for patt in patterns:                                                              # iterate over patterns
        for cand in sorted(glob.glob(patt)):                                           # iterate found paths
            if "__MACOSX" in cand:                                                     # skip MacOS artifact
                continue
            if os.path.isdir(cand):                                                    # if a directory (rare)
                inner = _read_csv_from_folder(cand)                                    # recurse one level
                if inner is not None:                                                  # if found a csv inside
                    return inner                                                       # return dataframe
            elif zipfile.is_zipfile(cand):                                             # if it's a zip file
                return _read_csv_from_zip(cand)                                        # read from zip
            else:                                                                       # otherwise plain csv file
                return _read_csv_plain(cand)                                           # read directly
    return None                                                                         # nothing found -> None

def load_block(entry_path: str) -> pd.DataFrame:
    """Load a single group (zip-like .csv or folder). Assign label 0 for 'genuine_accounts', else 1."""
    stem = os.path.splitext(os.path.basename(entry_path))[0]                           # base name without ext
    label = 0 if stem.lower() == "genuine_accounts" else 1                             # humans=0, bots=1

    df = None                                                                          # placeholder for data
    if os.path.isfile(entry_path):                                                     # if path is a file
        if zipfile.is_zipfile(entry_path):                                             # if file is actually a zip
            df = _read_csv_from_zip(entry_path)                                        # read from zip
            src = f"zip:{os.path.basename(entry_path)}"                                # for logging
        else:                                                                           # else plain csv on disk
            df = _read_csv_plain(entry_path)                                           # read directly
            src = f"plain:{os.path.basename(entry_path)}"                              # for logging
    elif os.path.isdir(entry_path):                                                    # if path is a folder
        df = _read_csv_from_folder(entry_path)                                         # search for a csv inside
        src = f"folder:{os.path.basename(entry_path)}" if df is not None else "folder:(empty)"  # log source

    if df is None:                                                                     # still nothing read?
        raise FileNotFoundError(f"No readable CSV found in {entry_path}")             # explicit error

    df["label"] = label                                                                # add binary label column
    print(f"✓ {src:30s} rows={len(df)} label={label}")                                 # log loaded group
    return df                                                                          # return dataframe

# ---------------------- MAIN MERGE LOGIC ----------------------

def main() -> None:
    """Merge all Cresci groups, dedupe, clean, and save one CSV for training."""
    if not os.path.isdir(DATA_PATH):                                                   # ensure DATA_PATH exists
        raise FileNotFoundError(f"DATA_PATH does not exist: {DATA_PATH}")             # clear error if missing

    # Collect both file-like entries (often 'zip named .csv') and subfolders in DATA_PATH
    top_files = sorted(glob.glob(os.path.join(DATA_PATH, "*.csv")))                    # e.g., fake_followers.csv (zip)
    top_dirs  = [p for p in glob.glob(os.path.join(DATA_PATH, "*")) if os.path.isdir(p)]  # e.g., fake_followers.csv/ (folder)
    entries   = top_files + top_dirs                                                   # combine candidates

    # Filter to the expected group names if present; otherwise use all entries (more robust across bundles)
    selected = []                                                                      # list to store picks
    for e in entries:                                                                  # iterate each entry
        stem = os.path.splitext(os.path.basename(e))[0]                                # name without extension
        if stem in WANTED:                                                             # only take wanted groups
            selected.append(e)
    if not selected:                                                                   # if none matched WANTED
        selected = entries                                                             # fall back to "all entries"

    # Load each selected block, returning a list of dataframes
    frames = [load_block(p) for p in selected]                                         # read every group
    df = pd.concat(frames, ignore_index=True, sort=False)                              # concatenate them all

    # Normalize expected column names where Cresci sometimes differs (e.g., statuses_count vs status_count)
    if "statuses_count" in df.columns and "status_count" not in df.columns:            # if alternative name exists
        df = df.rename(columns={"statuses_count": "status_count"})                      # unify to status_count

    # Keep a compact feature set if available (others will be dropped automatically)
    keep_cols = ["id", "screen_name", "followers_count", "friends_count", "status_count", "description", "label"]
    cols = [c for c in keep_cols if c in df.columns]                                   # only keep present columns
    if cols:                                                                           # if we have any of them
        df = df[cols]                                                                  # reduce to compact set

    # Remove duplicate users if possible (prefer unique 'id', else unique 'screen_name')
    if "id" in df.columns:                                                             # if id exists
        before = len(df)                                                                # remember row count
        df = df.drop_duplicates(subset=["id"])                                         # deduplicate on id
        after = len(df)                                                                 # new row count
        print(f"Deduplicated by id: {before} -> {after}")                               # log dedupe effect
    elif "screen_name" in df.columns:                                                  # else if screen_name exists
        before = len(df)                                                                # remember row count
        df = df.drop_duplicates(subset=["screen_name"])                                 # dedupe on screen_name
        after = len(df)                                                                 # new row count
        print(f"Deduplicated by screen_name: {before} -> {after}")                      # log dedupe effect

    # Light cleaning: coerce numeric fields to integers, fill missing with 0
    for col in ["followers_count", "friends_count", "status_count"]:                   # iterate numeric cols
        if col in df.columns:                                                          # if the column exists
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)    # numeric -> int

    # Ensure text column exists and is string-typed (helps TF-IDF later)
    if "description" in df.columns:                                                    # if description present
        df["description"] = df["description"].fillna("").astype(str)                   # normalize text

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)                              # create folder if missing
    # Save the merged CSV to your project
    df.to_csv(OUT_PATH, index=False, encoding="utf-8")                                 # write merged csv
    # Log final stats
    print(f"\n✅ Saved merged file: {OUT_PATH}")                                        # success message
    print(f"Rows: {len(df):,} | Columns: {list(df.columns)}")                          # dataset summary

# Standard Python entry point
if __name__ == "__main__":                                                             # if run as a script
    main()                                                                             # execute merge
