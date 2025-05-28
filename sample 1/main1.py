from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import pandas as pd
import io
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from sklearn.preprocessing import StandardScaler, LabelEncoder, MultiLabelBinarizer
from scipy.spatial import distance
from scipy.stats import pearsonr
from Levenshtein import distance as levenshtein_distance
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Allow CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load SBERT model once
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Utility to read uploaded file into a DataFrame
def read_file_to_df(file: UploadFile) -> pd.DataFrame:
    contents = file.file.read()
    extension = file.filename.split(".")[-1].lower()
    try:
        if extension == "csv":
            df = pd.read_csv(io.BytesIO(contents))
        elif extension in ("xlsx", "xls"):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        return df.dropna()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")

# Similarity/Distance Functions
def compute_cosine_similarity(df: pd.DataFrame, attributes: List[str]) -> np.ndarray:
    text_data = df[attributes].astype(str).agg(' '.join, axis=1)
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(text_data)
    return cosine_similarity(vectors)

def compute_jaccard_similarity(df: pd.DataFrame, attributes: List[str]) -> np.ndarray:
    data = df[attributes].astype(str).agg(' '.join, axis=1).str.lower().str.split()
    mlb = MultiLabelBinarizer()
    binary = mlb.fit_transform(data)
    rows = len(binary)
    matrix = np.zeros((rows, rows))
    for i in range(rows):
        for j in range(rows):
            matrix[i, j] = np.sum(np.minimum(binary[i], binary[j])) / np.sum(np.maximum(binary[i], binary[j]))
    return matrix

def compute_euclidean_distance(df: pd.DataFrame, attributes: List[str]) -> np.ndarray:
    num_data = df[attributes].select_dtypes(include='number')
    scaled = StandardScaler().fit_transform(num_data)
    return euclidean_distances(scaled)

def compute_manhattan_distance(df: pd.DataFrame, attributes: List[str]) -> np.ndarray:
    num_data = df[attributes].select_dtypes(include='number')
    scaled = StandardScaler().fit_transform(num_data)
    return manhattan_distances(scaled)

def compute_mahalanobis_distance(df: pd.DataFrame, attributes: List[str]) -> np.ndarray:
    num_data = df[attributes].select_dtypes(include='number')
    cov_matrix = np.cov(num_data.T)
    inv_cov = np.linalg.pinv(cov_matrix)
    rows = len(num_data)
    matrix = np.zeros((rows, rows))
    for i in range(rows):
        for j in range(rows):
            matrix[i, j] = distance.mahalanobis(num_data.iloc[i], num_data.iloc[j], inv_cov)
    return matrix

def compute_hamming_distance(df: pd.DataFrame, attributes: List[str]) -> np.ndarray:
    cat_data = df[attributes].astype(str).apply(LabelEncoder().fit_transform)
    rows = len(cat_data)
    matrix = np.zeros((rows, rows))
    for i in range(rows):
        for j in range(rows):
            matrix[i, j] = distance.hamming(cat_data.iloc[i], cat_data.iloc[j])
    return matrix

def compute_pearson_similarity(df: pd.DataFrame, attributes: List[str]) -> np.ndarray:
    num_data = df[attributes].select_dtypes(include='number')
    rows = len(num_data)
    matrix = np.zeros((rows, rows))
    for i in range(rows):
        for j in range(rows):
            corr, _ = pearsonr(num_data.iloc[i], num_data.iloc[j])
            matrix[i, j] = corr if not np.isnan(corr) else 0
    return matrix

def compute_levenshtein_distance(df: pd.DataFrame, attributes: List[str]) -> np.ndarray:
    text_data = df[attributes].astype(str).agg(' '.join, axis=1)
    rows = len(text_data)
    matrix = np.zeros((rows, rows))
    for i in range(rows):
        for j in range(rows):
            matrix[i, j] = levenshtein_distance(text_data.iat[i], text_data.iat[j])
    return matrix

def compute_dot_product_similarity(df: pd.DataFrame, attributes: List[str]) -> np.ndarray:
    num_data = df[attributes].select_dtypes(include='number').values
    return np.dot(num_data, num_data.T)

def compute_learned_embedding_similarity(df: pd.DataFrame, attributes: List[str]) -> np.ndarray:
    text_data = df[attributes].astype(str).agg(' '.join, axis=1).tolist()
    embeddings = sbert_model.encode(text_data)
    return cosine_similarity(embeddings)

# GET route for friendly browser access
@app.get("/compute_similarity/")
def get_info():
    return {"message": "Please use POST method with file upload and form data: selected_attributes and selected_metrics."}

# Main POST route
@app.post("/compute_similarity/")
async def compute_similarity(
    file: UploadFile = File(...),
    selected_attributes: str = Form(...),  # comma-separated attribute names
    selected_metrics: str = Form(...),     # comma-separated metric names
):
    df = read_file_to_df(file)
    attributes = [a.strip() for a in selected_attributes.split(",")]
    metrics = [m.strip().lower() for m in selected_metrics.split(",")]

    if not set(attributes).issubset(df.columns):
        raise HTTPException(status_code=400, detail="One or more selected attributes are not in the dataset")

    results = {}

    try:
        if 'cosine' in metrics:
            results['cosine'] = compute_cosine_similarity(df, attributes).tolist()
        if 'jaccard' in metrics:
            results['jaccard'] = compute_jaccard_similarity(df, attributes).tolist()
        if 'euclidean' in metrics:
            results['euclidean'] = compute_euclidean_distance(df, attributes).tolist()
        if 'manhattan' in metrics:
            results['manhattan'] = compute_manhattan_distance(df, attributes).tolist()
        if 'mahalanobis' in metrics:
            results['mahalanobis'] = compute_mahalanobis_distance(df, attributes).tolist()
        if 'hamming' in metrics:
            results['hamming'] = compute_hamming_distance(df, attributes).tolist()
        if 'pearson' in metrics:
            results['pearson'] = compute_pearson_similarity(df, attributes).tolist()
        if 'levenshtein' in metrics:
            results['levenshtein'] = compute_levenshtein_distance(df, attributes).tolist()
        if 'dotproduct' in metrics:
            results['dotproduct'] = compute_dot_product_similarity(df, attributes).tolist()
        if 'learnedembedding' in metrics:
            results['learnedembedding'] = compute_learned_embedding_similarity(df, attributes).tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing similarity: {str(e)}")

    return results
