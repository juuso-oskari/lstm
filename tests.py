import cupy as cp
import time
import pandas as pd
import numpy as np
from sklearn import preprocessing




if __name__ == "__main__":

    data = pd.read_csv('USD_INR.csv')

    test_data = data.truncate(after=1984, axis="index").loc[:, "Price"]
    train_data = data.truncate(before=1985, axis="index").loc[:, "Price"]

    x = test_data.values.reshape(-1, 1)  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    print(df)





