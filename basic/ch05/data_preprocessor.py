import numpy as np
from sklearn import preprocessing
# 深度学习 tensorflow pytorch

input_data = np.array([[5.1, -2.9, 3.3],
                       [-1.2, 7.8, -6.1],
                       [3.9, 0.4, 2.1],
                       [7.3, -9.9, -4.5]])

data_binarized = preprocessing.Binarizer(threshold=2.1).transform(input_data)
print(data_binarized)

print("\nAfter")
print("Mean = ", input_data.mean(axis=0))
print("std deviation = ", input_data.std(axis=0))

data_scaled = preprocessing.scale(input_data)
print("\nAfter")
print("Mean = ", data_scaled.mean(axis=0))
print("std deviation = ", data_scaled.std(axis=0))

data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaler_minmax = data_scaler_minmax.fit_transform(input_data)
print(data_scaler_minmax)

data_normalized_l1 = preprocessing.normalize(input_data, norm='l1')
data_normalized_l2 = preprocessing.normalize(input_data, norm='l2')
print(data_normalized_l1)
print(data_normalized_l2)
