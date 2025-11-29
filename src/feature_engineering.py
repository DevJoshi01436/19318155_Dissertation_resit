import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix

def _find_label_col(df: pd.DataFrame):
    for c in ["label", "is_bot", "bot", "target", "is_fake"]:
        if c in df.columns:
            return c
    raise ValueError("Could not find a label column. Add one named 'label' or 'is_bot'.")

def _find_text_col(df: pd.DataFrame):
    for c in ["description", "bio", "text", "tweet_text"]:
        if c in df.columns:
            return c
    return None

def _numeric_frame(df: pd.DataFrame, exclude: list) -> pd.DataFrame:
    num_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return df[num_cols].copy(), num_cols

def _safe_split(X, y, base_test_size=0.2, random_state=42):
    y = np.asarray(y)
    n = len(y)
    classes, counts = np.unique(y, return_counts=True)
    n_classes = len(classes)

    # need ≥1 test sample/class if stratifying
    min_test_fraction = n_classes / max(n, 1)
    test_size = max(base_test_size, min_test_fraction)

    can_stratify = np.all(counts >= 2) and n >= 6
    stratify = y if can_stratify else None

    if n < 6 or not can_stratify:
        return train_test_split(X, y, test_size=0.5, random_state=random_state, stratify=None)
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)

def prepare_splits(df: pd.DataFrame):
    """
    Returns:
      X_train, X_test (scipy sparse)
      y_train, y_test (np arrays)
      feature_names (list[str])
    All transformers are fit ONLY on training data to avoid leakage.
    """
    df = df.copy()
    y_col = _find_label_col(df)
    text_col = _find_text_col(df)

    y = df[y_col].astype(int).values
    # assemble raw numeric/text before split
    exclude = [y_col] + ([text_col] if text_col else [])
    num_df, num_cols = _numeric_frame(df, exclude)

    # quick placeholder arrays to split; we will transform after split
    dummy_X = np.zeros((len(df), 1))
    X_train_idx, X_test_idx, y_train, y_test = train_test_split(
        np.arange(len(df)), y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y))>1 else None
    )

    # numeric
    scaler = StandardScaler(with_mean=False)
    num_train = csr_matrix(num_df.iloc[X_train_idx].fillna(0).values)
    num_test  = csr_matrix(num_df.iloc[X_test_idx].fillna(0).values)
    if num_train.shape[1] > 0:
        num_train = scaler.fit_transform(num_train)
        num_test  = scaler.transform(num_test)

    # text → TF-IDF
    if text_col:
        vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
        text_train = vectorizer.fit_transform(df.iloc[X_train_idx][text_col].astype(str).values)
        text_test  = vectorizer.transform(df.iloc[X_test_idx][text_col].astype(str).values)
        X_train = hstack([num_train, text_train]).tocsr()
        X_test  = hstack([num_test,  text_test]).tocsr()
        feature_names = list(num_cols) + [f"tfidf_{i}" for i in range(text_train.shape[1])]
    else:
        X_train, X_test = num_train, num_test
        feature_names = list(num_cols)

    return X_train, X_test, y_train, y_test, feature_names
