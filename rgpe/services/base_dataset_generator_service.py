import requests
import mpmath as mp
import numpy as np
import pandas as pd
from . import dataset_loader_service
from . import riemann_service
from ..utils.decorators import log_dataset_generation_execution


@log_dataset_generation_execution("Gram Points")
def generate_gram_points_dataset(start: int = 0, end: int = 100_000, seed: float = 7.0) -> None:
    """
    Gera um CSV com Gram points de start até end.
    Usa o Gram point anterior como chute inicial para o próximo.
    """
    t0 = mp.mpf(seed)
    gram_points = []

    for n in range(start, end + 1):
        gram_point = riemann_service.get_gram_point(n - 1, t0)
        gram_points.append((n, gram_point))
        t0 = gram_point

        if n % 1000 == 0:
            print(f"Gram Progress: {n / end * 100:.2f}%")

    df = pd.DataFrame(gram_points, columns=["n", "gram_point"])
    df.to_csv("./dataset/gram_points.csv", index=False)


@log_dataset_generation_execution("coGram Points")
def generate_cogram_points_dataset(start: int = 0, end: int = 100_000, seed: float = 7.0) -> None:
    t0 = mp.mpf(seed)
    gram_points = []

    for n in range(start, end + 1):
        gram_point = riemann_service.get_cogram_point(n - 1, t0)
        gram_points.append((n, gram_point))
        t0 = gram_point

        if n % 1000 == 0:
            print(f"coGram Progress: {n / end * 100:.2f}%")

    df = pd.DataFrame(gram_points, columns=["n", "cogram_point"])
    df.to_csv("./dataset/cogram_points.csv", index=False)


@log_dataset_generation_execution("Zeta zeros")
def _download_zeta_zeros() -> None:
    """
    Faz download dos zeros da função zeta de Riemann e salva em CSV.
    """
    url = "https://www-users.cse.umn.edu/~odlyzko/zeta_tables/zeros1"
    r = requests.get(url)
    r.raise_for_status()
    zeros = np.fromstring(r.text, sep="\n").astype(float)
    df = pd.DataFrame({"n": np.arange(1, len(zeros) + 1),"zeta_zero": zeros})
    df.to_csv("./dataset/zeta_zeros.csv", index=False)


def _write_distances_dataset(limit: int = 100_000) -> None:
    """
    Gera as distâncias entre o zero e o ponto de gram.
    """
    zeros = dataset_loader_service.load_zeta_zeros()
    gram_points = dataset_loader_service.load_gram_points(limit=limit + 1)

    print(zeros.shape, gram_points.shape)
    y = np.zeros((limit,), dtype=np.float64)

    for i in range(limit):
        y[i] = zeros[i] - gram_points[i]

    df = pd.DataFrame({"n": np.arange(1, limit + 1), "distance": y})
    df.to_csv("./dataset/distances.csv", index=False)


@log_dataset_generation_execution("Distances")
def generate_distances_dataset(limit: int = 100_000) -> None:
    _download_zeta_zeros()
    _write_distances_dataset(limit)
