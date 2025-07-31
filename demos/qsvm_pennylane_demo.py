import numpy as np
from numpy.typing import NDArray

from base_demo import BaseDemo

from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pennylane as qml
from pennylane.templates import AngleEmbedding

from typing import Callable, List


class QSVMPennylaneDemo(BaseDemo):
    def __init__(self) -> None:
        super().__init__()
        self.n_qubits: int | None = None
        self.dev_kernel: qml.Device | None = None
        self.projector: NDArray[np.float64] | None = None

    def _load_data(self) -> List[NDArray[np.float64]]:
        X, y = load_iris(return_X_y=True)
        X = X[:100]
        y = y[:100]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        y_scaled = 2 * (y - 0.5)

        return train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    def _create_kernel_device(self, n_qubits: int) -> Callable[
        [NDArray[np.float64], NDArray[np.float64]], float
    ]:
        self.n_qubits = n_qubits
        self.dev_kernel = qml.device("lightning.qubit", wires=n_qubits)

        self.projector = np.zeros((2 ** n_qubits, 2 ** n_qubits))
        self.projector[0, 0] = 1

        @qml.qnode(self.dev_kernel)
        def kernel(x1: NDArray[np.float64], x2: NDArray[np.float64]) -> float:
            wires = range(n_qubits)
            AngleEmbedding(x1, wires=wires)
            qml.adjoint(AngleEmbedding)(x2, wires=wires)
            return qml.expval(qml.Hermitian(self.projector, wires=wires))

        return kernel

    def _kernel_matrix(
        self,
        A: NDArray[np.float64],
        B: NDArray[np.float64],
        kernel_fn: Callable[[NDArray[np.float64], NDArray[np.float64]], float]
    ) -> NDArray[np.float64]:
        return np.array([[kernel_fn(a, b) for b in B] for a in A])

    def exec(self) -> None:
        np.random.seed(42)

        X_train, X_test, y_train, y_test = self._load_data()
        self.n_qubits = len(X_train[0])

        kernel_fn = self._create_kernel_device(self.n_qubits)

        K_train = self._kernel_matrix(X_train, X_train, kernel_fn)
        K_test = self._kernel_matrix(X_test, X_train, kernel_fn)

        svm = SVC(kernel="precomputed")
        svm.fit(K_train, y_train)

        if self.dev_kernel is not None:
            with self.dev_kernel.tracker:
                y_pred = svm.predict(K_test)
                accuracy = accuracy_score(y_test, y_pred)
                print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    demo = QSVMPennylaneDemo()
    demo.exec()
