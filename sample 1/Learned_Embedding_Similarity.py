import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')
df = pd.read_csv("learned_embedding_similarity_file.csv").dropna()
text_data = df.select_dtypes(include='object').astype(str).agg(' '.join, axis=1)
embeddings = model.encode(text_data)
similarity_matrix = cosine_similarity(embeddings)

print(similarity_matrix)
