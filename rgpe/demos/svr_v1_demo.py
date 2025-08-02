import pandas as pd
import time
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from numpy.typing import NDArray
from .base_demo import BaseDemo


class SVRV1Demo(BaseDemo):
    def __init__(self):
        super().__init__()

        df = pd.read_csv("/app/dataset/gram_distance.csv")
        self.X = df[["gram_point"]].values
        self.y = df["distance_to_zero"].values

    def run(self) -> None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, self.y, test_size=0.2, random_state=42
        )

        C_values = [0.1, 1.0, 10.0]
        epsilon_values = [0.01, 0.1, 0.2]
        results: list[dict] = []

        for C in C_values:
            for epsilon in epsilon_values:
                print(f"Training SVR with C={C}, epsilon={epsilon}")
                self.__train_model(X_train, y_train, X_test, y_test, C, epsilon, results)

        self.__save_results(results)

    def __train_model(
            self,
            X_train: NDArray,
            y_train: NDArray,
            X_test: NDArray,
            y_test: NDArray,
            C: float,
            epsilon: float,
            results: list[dict]
    ) -> None:
        svr = SVR(kernel="rbf", C=C, epsilon=epsilon)
        start_time = time.time()
        svr.fit(X_train, y_train)
        y_pred = svr.predict(X_test)
        end_time = time.time()
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results.append({
            "C": C,
            "epsilon": epsilon,
            "mse": mse,
            "r2": r2,
            "exec_time": end_time - start_time
        })

    def __save_results(self, results: list[dict]) -> None:
        df_results = pd.DataFrame(results)
        print(df_results)
        df_results.to_csv("/app/results/svr_v1.csv", index=False)
