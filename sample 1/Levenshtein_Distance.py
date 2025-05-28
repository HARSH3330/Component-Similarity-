import pandas as pd
from Levenshtein import distance as levenshtein_distance

df = pd.read_csv("levenshtein_distance_file.csv").dropna()
text_data = df.astype(str).agg(' '.join, axis=1)

rows = len(text_data)
matrix = [[levenshtein_distance(text_data[i], text_data[j]) for j in range(rows)] for i in range(rows)]

print(matrix)
