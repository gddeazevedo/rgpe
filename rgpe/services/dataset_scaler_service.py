import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class DatasetScalerService:
    MAX_LIMIT = 100_000
    OFFSET    = 1_000

    def __init__(self, features: list[str], scaler: type[MinMaxScaler | StandardScaler]):
        self.__scaler_X = scaler()
        self.__X_original = pd.read_csv("dataset/j_kampe.csv")
        self.__y_original = pd.read_csv("dataset/distances.csv")["distance"]
        self.__features = features

    def get_scaled_data(self, limit: int = 11_000):
        if limit + self.OFFSET > self.MAX_LIMIT:
            limit = self.MAX_LIMIT

        X = self.__X_original[self.__features].values
        y = self.__y_original.values.ravel()

        X = X[self.OFFSET:limit + self.OFFSET]
        y = y[self.OFFSET:limit + self.OFFSET]

        X_train = X[:int(0.8 * X.shape[0])]
        X_test  = X[int(0.8 * X.shape[0]):]
        y_train = y[:int(0.8 * y.shape[0])]
        y_test  = y[int(0.8 * y.shape[0]):]

        X_train_scaled = self.__scaler_X.fit_transform(X_train)
        X_test_scaled  = self.__scaler_X.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test
