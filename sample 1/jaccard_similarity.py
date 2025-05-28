import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import jaccard_score

df = pd.read_csv("jaccard_similarity_file.csv").dropna()
data = df.apply(lambda x: x.astype(str).str.lower().str.split(','), axis=1).agg(' '.join, axis=1).str.split()

mlb = MultiLabelBinarizer()
binary = mlb.fit_transform(data)
rows = len(binary)

similarity_matrix = [[jaccard_score(binary[i], binary[j]) for j in range(rows)] for i in range(rows)]
print(similarity_matrix)
