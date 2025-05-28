import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import hamming

df = pd.read_csv("hamming_distance_file.csv").dropna()
cat_data = df.astype(str).apply(LabelEncoder().fit_transform)

rows = len(cat_data)
matrix = [[hamming(cat_data.iloc[i], cat_data.iloc[j]) for j in range(rows)] for i in range(rows)]
print(matrix)
