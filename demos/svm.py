import numpy as np
from itertools import combinations

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler

# ===============================
# CONFIGURAÇÃO - PARÂMETROS GLOBAIS
# ===============================
N_QUBITS = 2        # Número de qubits
N_FEATURES = 2      # Número de features do dataset
SAMPLE_SIZE = 130   # Número de amostras totais
RANDOM_SEED = 42

############################################################################
# ZZ FeatureMap
############################################################################
def ZZFeatureMap_qiskit(nqubits, data):
    qc = QuantumCircuit(nqubits)
    nload = min(len(data), nqubits)
    for i in range(nload):
        qc.h(i)
        qc.rz(2.0 * data[i], i)
    for q0, q1 in combinations(range(nload), 2):
        qc.cz(q0, q1)
        qc.rz(2.0 * (np.pi - data[q0])*(np.pi - data[q1]), q1)
        qc.cz(q0, q1)
    return qc

############################################################################
# Construção do circuito de kernel (U(a) U(b)^\dagger)
############################################################################
def kernel_circuit(a, b, nqubits):
    qc_a = ZZFeatureMap_qiskit(nqubits, a)
    qc_b = ZZFeatureMap_qiskit(nqubits, b)
    return qc_a.compose(qc_b.inverse())

############################################################################
# Overlap |0...0> para medir similaridade
############################################################################
def kernel_overlap(a, b, nqubits):
    qc_k = kernel_circuit(a, b, nqubits)
    sv = Statevector.from_label('0'*nqubits).evolve(qc_k)
    amp_0 = sv.data[0]
    return abs(amp_0)**2

############################################################################
# Matriz de Gram
############################################################################
def build_kernel_matrix(X1, X2, nqubits):
    n1, n2 = len(X1), len(X2)
    K = np.zeros((n1, n2))
    for i, a in enumerate(X1):
        for j, b in enumerate(X2):
            K[i, j] = kernel_overlap(a, b, nqubits)
    return K


def main():
    # 1) Carrega Iris (classes 0 e 1)
    X, y = load_iris(return_X_y=True)
    mask = (y == 0) | (y == 1)
    X, y = X[mask], y[mask]

    # 2) Seleciona features e amostras
    X = X[:, :N_FEATURES][:SAMPLE_SIZE]
    y = y[:SAMPLE_SIZE]

    # 3) Escalonamento [0,1]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # 4) Split treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, shuffle=True, random_state=RANDOM_SEED
    )

    # 5) Constroi matrizes de kernel
    nqubits = N_QUBITS
    K_train = build_kernel_matrix(X_train, X_train, nqubits)
    K_test  = build_kernel_matrix(X_test,  X_train, nqubits)

    # 6) Treina e avalia SVM
    svm = SVC(kernel='precomputed')
    svm.fit(K_train, y_train)
    acc = svm.score(K_test, y_test)

    print(f"Acurácia do QSVM (ZZ feature map): {acc:.3f}")

if __name__ == "__main__":
    main()