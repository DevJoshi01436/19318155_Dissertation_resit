# src/prediction.py

# Import typing for type hints
from typing import List

# Import joblib to save/load trained models and pipelines
import joblib

# Import pandas for DataFrame construction
import pandas as pd


def load_model(path: str):
    """
    Load a previously saved scikit-learn Pipeline or model from disk.
    """
    # joblib.load deserializes the object (Pipeline/model) from the given file path
    model = joblib.load(path)
    # Return the loaded object to the caller
    return model


def predict_accounts(model, records: List[dict]) -> pd.DataFrame:
    """
    Use a trained model to predict labels for a list of account feature dicts.

    Each dict in 'records' should match the training columns, e.g.:
        {
          "followers_count": 123,
          "friends_count": 456,
          "status_count": 789,
          "description": "some bio text..."
        }
    """
    # Convert the list of dictionaries into a pandas DataFrame
    df = pd.DataFrame(records)

    # Use the model's .predict() method to get predicted class labels
    preds = model.predict(df)

    # Try to get prediction probabilities if the model supports predict_proba
    try:
        # predict_proba returns an array of shape (n_samples, 2) for binary classification
        proba = model.predict_proba(df)[:, 1]  # probability of the positive class (label 1)
    except Exception:
        # If predict_proba is not available, set probabilities to None
        proba = None

    # Build a result DataFrame with the original input and the predictions
    result = df.copy()
    result["pred_label"] = preds  # add predicted label column
    # If we have probabilities, also add them as a column
    if proba is not None:
        result["pred_proba"] = proba

    # Return the result DataFrame to the caller
    return result
