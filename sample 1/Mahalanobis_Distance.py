import pandas as pd
import numpy as np
from scipy.spatial import distance

df = pd.read_csv("mahalanobis_distance_file.csv").dropna()
num_data = df.select_dtypes(include='number')
cov_matrix = np.cov(num_data.T)
inv_cov = np.linalg.inv(cov_matrix)

rows = len(num_data)
matrix = np.zeros((rows, rows))

for i in range(rows):
    for j in range(rows):
        matrix[i, j] = distance.mahalanobis(num_data.iloc[i], num_data.iloc[j], inv_cov)

print(matrix)
