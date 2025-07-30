import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVR
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_aer import AerSimulator
from qiskit.primitives import BackendSamplerV2 as BackendSampler

df = pd.read_csv("dataset/gram_distance.csv")

X = df[["gram_point"]].values
y = df["distance_to_zero"].values

X = X[:50]
y = y[:50]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

sampler  = BackendSampler(backend=AerSimulator())
fidelity = ComputeUncompute(sampler=sampler)

feature_map    = ZZFeatureMap(feature_dimension=2, reps=2)
quantum_kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)

qsvr = QSVR(quantum_kernel=quantum_kernel)

print("Training QSVR model...")
qsvr.fit(X_train, y_train)

y_pred = qsvr.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"R² Score: {r2:.4f}")
