import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("cosine_similarity_file.csv").dropna()
text_data = df.select_dtypes(include='object').astype(str).agg(' '.join, axis=1)
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(text_data)
similarity_matrix = cosine_similarity(vectors)

print(similarity_matrix)
