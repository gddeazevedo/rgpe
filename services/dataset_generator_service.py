import requests
import numpy as np
import pandas as pd
from lib.gram_points.GramPoints import write_gram_points


def generate_gram_points_dataset() -> None:
    write_gram_points('./dataset/gram_points.csv', 0, 100000)


def download_zeta_zeros() -> None:
    """
    Downloads the Riemann zeta function zeros from a specified URL and saves them as a NumPy array.
    """
    url = "https://www-users.cse.umn.edu/~odlyzko/zeta_tables/zeros1"
    r = requests.get(url)
    r.raise_for_status()
    zeros = np.fromstring(r.text, sep="\n")
    zeros = zeros.astype(float)
    np.save("./dataset/zeta_zeros.npy", zeros)


def write_gram_distance_dataset() -> None:
    """
    Gera o dataset de distâncias entre os pontos gramaticais e os zeros da função zeta.
    """
    zeros   = np.load("./dataset/zeta_zeros.npy")
    df_gram = pd.read_csv("./dataset/gram_points.csv")
    gram_points = df_gram["n-th gram point"].values

    if len(zeros) > len(gram_points) - 1:
        print("[AVISO] Cortando zeros para casar com g_{n-1}")
        zeros = zeros[:len(gram_points) - 1]

    X = gram_points[1:]
    g_prev = gram_points[:-1]
    y = zeros - g_prev

    df = pd.DataFrame({
        "gram_point": X,
        "distance_to_zero": y
    })

    df.to_csv("dataset/gram_distance.csv", index=False)
    print("[OK] Dataset gerado com shape:", df.shape)
