from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVR
from ..services.dataset_loader_service import load_gram_distance_dataset
from .base_demo import BaseDemo


class QSVRQiskitDemo(BaseDemo):
    def run(self):
        X, y = load_gram_distance_dataset()

        X = X[:100]
        y = y[:100]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X, y)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        feature_map = ZZFeatureMap(feature_dimension=2, reps=2)
        quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

        qsvr = QSVR(quantum_kernel=quantum_kernel)
        qsvr.fit(X_train, y_train)

        y_pred = qsvr.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"MSE: {mse:.4f}")
        print(f"R² Score: {r2:.4f}")
