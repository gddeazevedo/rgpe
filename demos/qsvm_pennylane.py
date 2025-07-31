import numpy as np

from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pennylane as qml
from pennylane.templates import AngleEmbedding


np.random.seed(42)

X, y = load_iris(return_X_y=True)

X = X[:100]
y = y[:100]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

y_scaled = 2 * (y - 0.5)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

n_qubits = len(X_train[0])


dev_kernel = qml.device("lightning.qubit", wires=n_qubits)

projector = np.zeros((2 ** n_qubits, 2 ** n_qubits))
projector[0, 0] = 1


@qml.qnode(dev_kernel)
def kernel(x1, x2):
    wires = range(n_qubits)
    AngleEmbedding(x1, wires=wires)
    qml.adjoint(AngleEmbedding)(x2, wires=wires)
    return qml.expval(qml.Hermetian(projector, wires=wires))                   



