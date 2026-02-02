import pandas as pd
import numpy as np
from numpy.typing import NDArray


def load_zeta_zeros() -> NDArray[np.float64]:
    df_zeros = pd.read_csv("./dataset/zeta_zeros.csv")
    return np.asarray(df_zeros["zeta_zero"].values, dtype=np.float64)


def load_gram_points(limit: int = 100_000) -> NDArray[np.float64]:
    gram_points_df = pd.read_csv("./dataset/gram_points.csv")
    return np.asarray(gram_points_df["gram_point"].values[:limit], dtype=np.float64)


def load_distances() -> NDArray[np.float64]:
    df_distances = pd.read_csv("./dataset/distances.csv")
    return np.asarray(df_distances.iloc[:, 0].values, dtype=np.float64)


def load_cogram_points() -> NDArray[np.float64]:
    cogram_points_df = pd.read_csv("./dataset/cogram_points.csv")
    return np.asarray(cogram_points_df["cogram_point"].values, dtype=np.float64)


def load_gram_distance_dataset() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    df = pd.read_csv("./dataset/gram_distance.csv")
    X = np.asarray(df[["gram_point"]].values, dtype=np.float64)
    y = np.asarray(df["distance_to_zero"].values, dtype=np.float64)
    X = np.concatenate((X, np.zeros((X.shape[0], 1))), axis=1)
    return X, y


def load_40_features_dataset() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    df_features = pd.read_csv("./dataset/40_features.csv")
    df_distances = pd.read_csv("./dataset/distances.csv")
    y = np.asarray(df_distances.iloc[:df_features.shape[0], 0].values, dtype=np.float64)
    X = np.asarray(df_features.values, dtype=np.float64)
    return X, y


def load_o_shank_dataset() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    df_features = pd.read_csv("./dataset/o_shank.csv")
    df_distances = pd.read_csv("./dataset/distances.csv")
    y = np.asarray(df_distances.iloc[:df_features.shape[0], 0].values, dtype=np.float64)
    X = np.asarray(df_features.values, dtype=np.float64)
    return X, y


def load_j_kampe_dataset() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    df_j_kampe = pd.read_csv("./dataset/j_kampe.csv")
    distances = load_distances()
    y = np.asarray(distances[:df_j_kampe.shape[0]], dtype=np.float64)
    X = np.asarray(df_j_kampe.values, dtype=np.float64)
    return X, y
