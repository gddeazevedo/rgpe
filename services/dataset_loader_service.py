import pandas as pd
import numpy as np
from numpy.typing import NDArray


def load_gram_distance_dataset() -> tuple[NDArray, NDArray]:
    """
    Load the Gram distance dataset from a CSV file.
    
    Returns:
        X (np.ndarray): Features of the dataset.
        y (np.ndarray): Target values of the dataset.
    """
    df = pd.read_csv("/app/dataset/gram_distance.csv")
    X = df[["gram_point"]].values[:100]
    y = df["distance_to_zero"].values[:100]
    X = np.concatenate((X, np.zeros((100, 1))), axis=1)  # Ensure X is 2D
    return X, y
