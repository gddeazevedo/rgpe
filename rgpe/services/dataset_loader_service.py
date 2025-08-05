import pandas as pd
import numpy as np
from numpy.typing import NDArray


def load_gram_distance_dataset() -> tuple[NDArray[float], NDArray[float]]:
    """
    Load the Gram distance dataset from a CSV file.
    
    Returns:
        X (np.ndarray): Features of the dataset.
        y (np.ndarray): Target values of the dataset.
    """
    df = pd.read_csv("/app/dataset/gram_distance.csv")
    X = df[["gram_point"]].values
    y = df["distance_to_zero"].values
    X = np.concatenate((X, np.zeros((X.shape[0], 1))), axis=1)  # Ensure X is 2D
    return X, y


def load_40_features_dataset() -> tuple[NDArray[float], NDArray[float]]:
    df = pd.read_csv("/app/dataset/40_features.csv")
    X = df.drop(columns=["z_term1_1", "z_term1_2"]).values
    y = df["z_term1_1"].values
    return X, y
