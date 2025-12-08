# src/model_training.py
"""
Model building, evaluation, and visualisation utilities for Fake Profile Detection.

This module:
- Detects which columns are label, text, and numeric features
- Builds a preprocessing pipeline (scaling + TF-IDF)
- Builds model pipelines (LogReg, RandomForest, XGBoost)
- Runs cross-validated evaluation (no data leakage)
- Optional hold-out evaluation with confusion matrix + classification report
- Plots ROC curves for all models
- Plots PCA 2D projection of numeric features
"""

# ------------------------- Imports -------------------------

# Pipelines and preprocessing
from sklearn.pipeline import Pipeline                         # To chain preprocessing + model
from sklearn.compose import ColumnTransformer                 # To apply transforms to specific columns
from sklearn.preprocessing import StandardScaler              # To scale numeric features
from sklearn.feature_extraction.text import TfidfVectorizer   # To turn text into numeric vectors

# Model selection and evaluation tools
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import (
    get_scorer,                      # factory for standard metric scorers
    classification_report,           # detailed per-class metrics
    confusion_matrix,                # confusion matrix (TP, FP, TN, FN)
    roc_curve,                       # compute ROC points
    auc,                             # compute area under ROC
)

# Machine learning models
from sklearn.linear_model import LogisticRegression           # Linear baseline model
from sklearn.ensemble import RandomForestClassifier           # Tree ensemble model
from xgboost import XGBClassifier                             # Gradient boosting tree model

# PCA for dimensionality reduction
from sklearn.decomposition import PCA

# Numeric + table utilities
import numpy as np                                            # For numeric operations
import pandas as pd                                           # For DataFrame handling

# Plotting (for ROC and PCA visualisations)
import matplotlib.pyplot as plt                               # For plotting curves and scatter plots


# ------------------------- Scorers -------------------------

# Build a dictionary of evaluation metrics using scikit-learn scorers.
# We use these names in cross_validate() so it returns test_accuracy, test_precision, etc.
SCORERS = {
    name: get_scorer(name)
    for name in ["accuracy", "precision", "recall", "f1"]     # Metrics we care about
}


# ------------------------- Column Detection -------------------------

def _detect_columns(df: pd.DataFrame, label_hint: str = "label"):
    """
    Detect label, text, and numeric feature columns in the DataFrame.

    Args:
        df (pd.DataFrame): Input dataset.
        label_hint (str): Preferred label column name (usually "label").

    Returns:
        y_col (str): Name of the label column.
        text_col (str or None): Name of the text column used for TF-IDF (e.g. description).
        num_cols (list[str]): List of numeric feature column names.
    """
    # Candidate label column names in priority order.
    # We pick the first one that actually exists.
    label_candidates = [label_hint, "is_bot", "bot", "target", "is_fake"]

    y_col = None  # will store the chosen label column name

    for c in label_candidates:
        if c in df.columns:
            y_col = c
            break

    # If no label column was found at all, stop with a clear error message.
    if y_col is None:
        raise ValueError(
            f"No label column found (expected one of: {', '.join(label_candidates)})."
        )

    # Candidate text columns to use for TF-IDF.
    text_candidates = ["description", "bio", "text", "tweet_text"]

    text_col = None  # will store the chosen text column name (if any)

    for c in text_candidates:
        if c in df.columns:
            text_col = c
            break

    # Build list of numeric columns:
    # - all numeric dtypes
    # - excluding the label column
    # - excluding the text column (if present)
    exclude = [y_col] + ([text_col] if text_col else [])
    num_cols = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]

    return y_col, text_col, num_cols


# ------------------------- Preprocessor -------------------------

def _preprocessor(num_cols, text_col):
    """
    Build a ColumnTransformer that:
      - scales numeric columns
      - applies TF-IDF to the text column (if present)

    Args:
        num_cols (list[str]): numeric feature columns
        text_col (str or None): text column name

    Returns:
        ColumnTransformer: combined preprocessing step
    """
    # List of (name, transformer, columns) tuples passed to ColumnTransformer.
    transformers = []

    # If we have numeric columns, scale them with StandardScaler.
    # with_mean=False keeps it compatible with sparse outputs from TF-IDF.
    if num_cols:
        transformers.append(
            ("num", StandardScaler(with_mean=False), num_cols)
        )

    # If we have a text column, apply TF-IDF on it.
    # ColumnTransformer will pass this column as a 1D array/Series to TfidfVectorizer.
    if text_col:
        transformers.append(
            ("txt", TfidfVectorizer(max_features=1000, stop_words="english"), text_col)
        )

    # Build the ColumnTransformer:
    # - apply specified transformers
    # - drop any other columns
    # - sparse_threshold controls when it returns sparse vs dense matrices
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
    )


# ------------------------- Pipeline Builder -------------------------

def build_cv_pipelines(num_cols, text_col):
    """
    Build pipelines for each classifier (LogReg, RandomForest, XGBoost)
    using a shared preprocessor.

    Args:
        num_cols (list[str]): numeric feature columns
        text_col (str or None): text column name

    Returns:
        dict[str, Pipeline]: dictionary of model_name -> full pipeline
    """
    # Shared preprocessing (scaling + TF-IDF).
    pre = _preprocessor(num_cols, text_col)

    # Define the classifiers we want to test.
    # NOTE: n_estimators and folds have been kept moderate for speed on a laptop.
    models = {
        # Logistic Regression: linear baseline, good as a reference model.
        "logreg": LogisticRegression(max_iter=500),

        # Random Forest: ensemble of decision trees, handles nonlinear patterns.
        # Reduced n_estimators and n_jobs=-1 to make it faster on your machine.
        "rf": RandomForestClassifier(
            n_estimators=100,    # was 300: reduce for faster training
            random_state=19318155,
            n_jobs=-1,           # use all available CPU cores
        ),

        # XGBoost: gradient boosting on decision trees, strong for tabular data.
        # Reduced n_estimators and using 'hist' tree_method for speed.
        "xgb": XGBClassifier(
            n_estimators=200,     # was 400: reduce for faster training
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",   # faster algorithm on CPU
            n_jobs=-1,            # use all cores
            random_state=19318155,
        ),
    }

    # Wrap each model with the preprocessor into a single sklearn Pipeline.
    pipelines = {
        name: Pipeline([("pre", pre), ("clf", model)])
        for name, model in models.items()
    }

    return pipelines


# ------------------------- Cross-Validated Evaluation -------------------------

def cv_evaluate(df: pd.DataFrame, label_col: str = "label"):
    """
    Perform cross-validation for all models (LogReg, RF, XGB)
    with no data leakage.

    Steps:
      1) Detect columns (label, text, numeric)
      2) Clean numeric + text features (handle NaNs)
      3) Build pipelines
      4) Run Stratified K-Fold CV
      5) Return mean ± std for each metric

    Args:
        df (pd.DataFrame): cleaned input dataset
        label_col (str): name of the label column to prefer

    Returns:
        dict: { model_name: { metric_name: (mean, std), ... }, ... }
    """
    # 1. Detect label, text, numeric columns.
    y_col, text_col, num_cols = _detect_columns(df, label_hint=label_col)

    # 2. Build feature matrix X from numeric + optional text column.
    # We explicitly choose only the numeric columns + the text column (if any).
    X = df[[*(num_cols), *( [text_col] if text_col else [] )]].copy()

    # --- Text cleaning: TF-IDF cannot handle NaN, so fill with "" and cast to str ---
    if text_col:
        X[text_col] = X[text_col].fillna("").astype(str)

    # --- Numeric cleaning: many models cannot handle NaN, so we coerce + fill ---
    if num_cols:
        # Convert values to numeric; any invalid strings become NaN.
        for col in num_cols:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        # Replace any remaining NaNs with 0.
        X[num_cols] = X[num_cols].fillna(0)

    # 3. Build label vector y as integer array.
    y = df[y_col].astype(int).values

    # 4. Configure Stratified K-Fold CV SAFELY.
    #
    # We must ensure:
    #   - There are at least 2 classes (0 and 1)
    #   - Each class has at least 2 samples if we want n_splits >= 2
    #   - n_splits never exceeds the smallest class size or total samples
    unique_classes, counts = np.unique(y, return_counts=True)

    # If there is only one class, classification is meaningless.
    if len(unique_classes) < 2:
        raise ValueError(
            "cv_evaluate: need at least 2 different classes for classification "
            f"but found {len(unique_classes)} unique value(s): {unique_classes.tolist()}."
        )

    # Smallest class size (e.g., min(#bots, #genuine)).
    min_class = int(counts.min())

    # Overall sample size.
    total_samples = len(y)

    # If the smallest class has < 2 samples, we cannot do stratified CV with >=2 splits.
    if min_class < 2 or total_samples < 2:
        raise ValueError(
            "cv_evaluate: not enough data to run stratified cross-validation.\n"
            f"Total samples: {total_samples}, smallest class size: {min_class}.\n"
            "StratifiedKFold requires at least 2 samples in EACH class. "
            "Please use a larger dataset or switch to a simple train/test split (e.g. holdout_evaluate)."
        )

    # Choose number of folds:
    # - Max 3 folds (kept smaller than 5 to improve runtime on large datasets)
    # - Cannot exceed the smallest class size.
    # - Cannot exceed total number of samples.
    n_splits = min(3, min_class, total_samples)

    # Extra safety: n_splits must be at least 2.
    if n_splits < 2:
        raise ValueError(
            f"cv_evaluate: computed n_splits={n_splits}, which is invalid. "
            "Need at least 2 folds. Check class distribution and dataset size."
        )

    # Build StratifiedKFold to preserve class balance in each fold.
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=19318155,
    )

    # 5. Build pipelines for each model.
    pipes = build_cv_pipelines(num_cols, text_col)

    # Dictionary to hold summary stats per model.
    summaries = {}

    # 6. Run cross-validation for each model pipeline.
    for name, pipe in pipes.items():
        # cross_validate returns a dict of arrays: test_accuracy, test_precision, etc.
        scores = cross_validate(
            pipe,                # pipeline (preprocessing + classifier)
            X,                   # feature matrix (cleaned)
            y,                   # labels
            cv=cv,               # CV splitter
            scoring=SCORERS,     # metrics to compute
            error_score=np.nan,  # if a fold crashes, store NaN instead of raising
            n_jobs=-1,           # run folds in parallel where possible
        )

        # For each "test_*" metric, compute mean and std across folds.
        summaries[name] = {
            metric.replace("test_", ""): (
                float(np.nanmean(scores[metric])),   # mean score across folds
                float(np.nanstd(scores[metric]))     # standard deviation across folds
            )
            for metric in scores
            if metric.startswith("test_")
        }

    # Return nested dict of metrics, ready for printing in main.py.
    return summaries


# ------------------------- Hold-out Evaluation (Optional) -------------------------

def holdout_evaluate(df: pd.DataFrame, label_col: str = "label"):
    """
    Optional helper:
    Perform a single train/test split (80/20) and print:

      - Confusion matrix
      - Classification report

    using Logistic Regression as a simple reference model.

    Args:
        df (pd.DataFrame): cleaned dataset
        label_col (str): preferred label column name
    """
    # Detect columns the same way as in cv_evaluate.
    y_col, text_col, num_cols = _detect_columns(df, label_hint=label_col)

    # Build feature matrix X with numeric + optional text column.
    X = df[[*(num_cols), *( [text_col] if text_col else [] )]].copy()

    # Clean text column (no NaN, enforce string).
    if text_col:
        X[text_col] = X[text_col].fillna("").astype(str)

    # Clean numeric columns (coerce to numeric and fill NaN with 0).
    if num_cols:
        for col in num_cols:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        X[num_cols] = X[num_cols].fillna(0)

    # Build label vector.
    y = df[y_col].astype(int).values

    # Build preprocessor and classifier for the hold-out pipeline.
    pre = _preprocessor(num_cols, text_col)
    clf = LogisticRegression(max_iter=500)
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    # Split dataset into train (80%) and test (20%).
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=19318155,
        stratify=y if len(set(y)) > 1 else None,  # preserve class balance if possible
    )

    # Fit pipeline on training data.
    pipe.fit(X_train, y_train)

    # Predict labels on the test set.
    y_pred = pipe.predict(X_test)

    # Print confusion matrix: shows TP, TN, FP, FN.
    print("\n=== Hold-out Confusion Matrix (Logistic Regression) ===")
    print(confusion_matrix(y_test, y_pred))

    # Print precision, recall, F1 per class.
    print("\n=== Hold-out Classification Report (Logistic Regression) ===")
    print(classification_report(y_test, y_pred, digits=3))


# ------------------------- ROC Curve Plotting -------------------------

def plot_roc_curves(df: pd.DataFrame, label_col: str = "label"):
    """
    Plot ROC curves for all three models (LogReg, RF, XGB)
    using a single 80/20 train/test split.

    This is mainly for visualisation in the dissertation.

    Args:
        df (pd.DataFrame): cleaned dataset
        label_col (str): name of label column
    """
    # Detect columns and build X/y with the same cleaning logic.
    y_col, text_col, num_cols = _detect_columns(df, label_hint=label_col)

    # Build feature matrix with numeric + optional text.
    X = df[[*(num_cols), *( [text_col] if text_col else [] )]].copy()

    # Clean text: fill NaN and cast to str.
    if text_col:
        X[text_col] = X[text_col].fillna("").astype(str)

    # Clean numeric: coerce to numeric and fill NaN.
    if num_cols:
        for col in num_cols:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        X[num_cols] = X[num_cols].fillna(0)

    # Build label vector.
    y = df[y_col].astype(int).values

    # Use the same preprocessing as in cv_evaluate.
    pre = _preprocessor(num_cols, text_col)

    # Define models again for ROC plotting (same lighter settings).
    models = {
        "LogReg": LogisticRegression(max_iter=500),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            random_state=19318155,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=19318155,
        ),
    }

    # Split once into train/test.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=19318155,
        stratify=y if len(set(y)) > 1 else None,
    )

    plt.figure()  # start a new figure

    # Loop over models, fit, and plot ROC.
    for name, model in models.items():
        # Build the full pipeline for each model.
        pipe = Pipeline([("pre", pre), ("clf", model)])
        pipe.fit(X_train, y_train)

        # Try to use predict_proba; if not available, fall back to decision_function.
        try:
            y_score = pipe.predict_proba(X_test)[:, 1]
        except Exception:
            y_score = pipe.decision_function(X_test)

        # Compute ROC curve and AUC.
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve (matplotlib will choose colours automatically).
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

    # Plot diagonal random-chance line.
    plt.plot([0, 1], [0, 1], linestyle="--")

    # Axis labels and title.
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Fake Profile Detection Models")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


# ------------------------- PCA Projection -------------------------

def plot_pca_projection(
    df: pd.DataFrame,
    label_col: str = "label",
    n_components: int = 2,
    max_points: int = 5000,
):
    """
    Plot a 2D PCA projection of the numeric features, coloured by label.

    Notes:
    - Only numeric columns are used for PCA (followers, friends, statuses, etc.)
    - Text features (TF-IDF) are high-dimensional and are ignored here for clarity

    Args:
        df (pd.DataFrame): cleaned dataset
        label_col (str): label column name
        n_components (int): number of PCA components (2 by default for 2D scatter)
        max_points (int): max number of points to plot (for readability)
    """
    # Detect columns and focus only on numeric ones for PCA.
    y_col, text_col, num_cols = _detect_columns(df, label_hint=label_col)

    if not num_cols:
        print("No numeric columns available for PCA projection.")
        return

    # Work on a copy of the numeric columns.
    X_num = df[num_cols].copy()

    # Clean numeric (coerce to numeric, fill NaNs).
    for col in num_cols:
        X_num[col] = pd.to_numeric(X_num[col], errors="coerce")
    X_num = X_num.fillna(0)

    # Scale numeric features before PCA (mean=0, std=1).
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = scaler.fit_transform(X_num)

    # Apply PCA to reduce to n_components dimensions.
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Prepare labels.
    y = df[y_col].astype(int).values

    # Optionally subsample for readability if there are too many points.
    if len(y) > max_points:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(len(y), size=max_points, replace=False)
        X_pca = X_pca[idx]
        y = y[idx]

    # Scatter plot for each class separately (matplotlib chooses colours).
    plt.figure()
    for label_value in np.unique(y):
        mask = (y == label_value)
        plt.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            alpha=0.6,
            label=f"Class {label_value}",
            s=10,
        )

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("PCA Projection of Numeric Features")
    plt.legend()
    plt.grid(True)
    plt.show()
