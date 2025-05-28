import pandas as pd
from scipy.stats import pearsonr

df = pd.read_csv("pearson_similarity_file.csv").dropna()
num_data = df.select_dtypes(include='number')
rows = len(num_data)

matrix = [[pearsonr(num_data.iloc[i], num_data.iloc[j])[0] for j in range(rows)] for i in range(rows)]
print(matrix)
