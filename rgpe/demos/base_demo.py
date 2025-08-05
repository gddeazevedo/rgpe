from dataclasses import dataclass, field
from abc import abstractmethod, ABC
from sklearn.svm import SVR
from numpy.typing import NDArray
from sklearn.metrics import r2_score, mean_squared_error
import time
import pandas as pd


@dataclass
class BaseDemo(ABC):
    """Base class for demos."""

    results: list[dict[str, float]] = field(default_factory=list[dict[str, float]])

    @abstractmethod
    def run(self) -> None:
        pass

    def train_svr_model(
            self,
            X_train: NDArray,
            X_test: NDArray,
            y_train: NDArray,
            y_test: NDArray,
            kernel: str,
            epsilon: float,
            C: float
    ) -> None:
        svr = SVR(kernel=kernel, epsilon=epsilon, C=C)
        start_time = time.time()
        svr.fit(X_train, y_train)
        y_pred = svr.predict(X_test)
        end_time = time.time()
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        self.results.append({
            "C": C,
            "epsilon": epsilon,
            "mse": mse,
            "r2": r2,
            "execution_time": end_time - start_time,
        })

    def show_results(self) -> None:
        df_results = pd.DataFrame(self.results)
        print(df_results)

    def save_results(self, file_name: str) -> None:
        df_results = pd.DataFrame(self.results)
        path = f"/app/results/{file_name}"
        df_results.to_csv(path, index=False)
