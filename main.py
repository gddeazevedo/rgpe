import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, r2_score, mean_squared_error


df = pd.read_csv("dataset/gram_distance.csv")

X = df[["gram_point"]].values
y = df["distance_to_zero"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

svr = SVR(kernel="rbf", C=1.0, epsilon=0.1)
print("Treinando modelo SVR...")
svr.fit(X_train, y_train)

y_pred = svr.predict(X_test)

# rmse = root_mean_squared_error(y_test[:10], y_pred)
r2 = r2_score(y_test, y_pred)

# print(f"RMSE: {rmse:.8f}")
print(f"R²  : {r2:.8f}")
