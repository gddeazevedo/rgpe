import pandas as pd
import time
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error


df = pd.read_csv("/app/dataset/gram_distance.csv")

X = df[["gram_point"]].values
y = df["distance_to_zero"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

C_values = [0.1, 1.0, 10.0]
epsilon_values = [0.01, 0.1, 0.2]
best_mse = float("inf")
results: list[dict] = []

for C in C_values:
    for epsilon in epsilon_values:
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

df_results = pd.DataFrame(results)
print(df_results)
df_results.to_csv("/app/results/svr_v1.csv", index=False)
