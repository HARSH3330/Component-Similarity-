import pandas as pd
import numpy as np

df = pd.read_csv("dot_product_similarity_file.csv").dropna()
num_data = df.select_dtypes(include='number').values
dot_product_matrix = np.dot(num_data, num_data.T)

print(dot_product_matrix)
