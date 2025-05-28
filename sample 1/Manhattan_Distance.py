import pandas as pd
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("manhattan_distance_file.csv").dropna()
num_data = df.select_dtypes(include='number')
scaled = StandardScaler().fit_transform(num_data)
distance_matrix = manhattan_distances(scaled)

print(distance_matrix)
