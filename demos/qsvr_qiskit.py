import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVR


np.random.seed(42)

def generate_data(n_samples=100):
    X = np.random.uniform(-1, 1, size=(n_samples, 2))
    y = np.sin(np.pi * X[:, 0]) + np.cos(np.pi * X[:, 1])
    return X, y

X, y = generate_data(100)

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
