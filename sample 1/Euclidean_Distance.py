import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("euclidean_distance_file.csv").dropna()
num_data = df.select_dtypes(include='number')
scaled = StandardScaler().fit_transform(num_data)
distance_matrix = euclidean_distances(scaled)

print(distance_matrix)
