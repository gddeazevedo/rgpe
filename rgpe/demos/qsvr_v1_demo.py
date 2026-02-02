import time
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score, root_mean_squared_error
from qiskit.circuit.library import pauli_feature_map #, zz_feature_map
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from ..services.dataset_scaler_service import DatasetScalerService


class QSVRV1Demo:
    def run(self):
        gevrey_method_features = pd.read_csv("results/gevrey_method_feature_selection.csv")["feature"].tolist()
        features = gevrey_method_features[:10]
        C = 10
        epsilon = 0.001
        n_qubits = len(features)

        dataset_scaler_service = DatasetScalerService(features, StandardScaler)
        X_train, X_test, y_train, y_test = dataset_scaler_service.get_scaled_data(limit=1000)

        feature_map = pauli_feature_map(feature_dimension=n_qubits, reps=2, entanglement="linear")
        sampler = StatevectorSampler()
        fidelity = ComputeUncompute(sampler=sampler)

        print("Training Quantum SVR...")
        start = time.time()
        quantum_kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)
        svr = SVR(kernel=quantum_kernel.evaluate, C=C, epsilon=epsilon)
        svr.fit(X_train, y_train)
        y_pred = svr.predict(X_test)
        end = time.time()
        print(f"Training completed in {end - start} seconds.")
        r2 = r2_score(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        print(f"R2 score: {r2}")
        print(f"Root Mean Squared Error: {rmse}")

        result = {
            "r2_score": r2,
            "rmse": rmse,
            "training_time_seconds": end - start
        }

        pd.DataFrame([result]).to_csv("results/qsvr_v1_demo_results.csv", index=False)

'''
svr = SVR(kernel='precomputed', C=C, epsilon=epsilon)
K_train = quantum_kernel.evaluate(X_train)
K_test = quantum_kernel.evaluate(X_test, X_train)
svr.fit(K_train, y_train)
y_pred = svr.predict(K_test)
'''
