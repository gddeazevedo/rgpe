from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .base_demo import BaseDemo
from ..services.dataset_loader_service import load_gram_distance_dataset


class SVRV1Demo(BaseDemo):
    def __init__(self):
        super().__init__()
        self.X, self.y = load_gram_distance_dataset()

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
                self.train_svr_model(
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    C=C,
                    epsilon=epsilon,
                    kernel="rbf",
                )

        self.show_results()
        self.save_results("svr_v1.csv")
