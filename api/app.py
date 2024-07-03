import os
import gdown
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.security.api_key import APIKeyHeader
from starlette.status import HTTP_400_BAD_REQUEST
from utils.feature_extraction import extract_features
from utils.similarity_calculation import calculate_similarity
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import normalize
from joblib import load
from typing import Dict
import tensorflow as tf
import uvicorn
import h5py
import logging

# Google Drive file IDs (replace with actual IDs)
model_file_id = '1kIc8v0JUZOulf8g889k3aAFOBYJThq86'
features_file_id = '1oS4xjJFgh7HCUdiByAqCqPGnmR5L-vJY'
labels_file_id = '11b4egRBs3HlMVc15_FRNKnhc5OrMmE21'
image_paths_file_id = '1wMiw7nwaMcksi2gSJyyo4KGYkYj_NMTz'
image_urls_file_id = '1PXetHEkm1XwjWCpFTcjH2HUXp51L4JWm'
pca_file_id = '1omkcIVUPndwFKQyqAj4QlURX8xGjkW8s'
kmeans_file_id = '11YPpfe9mnQsszZypuqemZwkbPMMVvzDP'

# File paths where the files will be saved
model_path = 'models/fine_tuned_resnet_model.h5'
features_path = 'models/features.npy'
labels_path = 'models/labels.npy'
image_paths_path = 'models/image_paths.npy'
image_urls_path = 'models/image_urls.npy'
pca_path = 'models/pca_model.joblib'
kmeans_path = 'models/kmeans_model.joblib'

# Function to download files from Google Drive
def download_file(file_id, output_path):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output_path, quiet=False)

# Download the necessary files
download_file(model_file_id, model_path)
download_file(features_file_id, features_path)
download_file(labels_file_id, labels_path)
download_file(image_paths_file_id, image_paths_path)
download_file(image_urls_file_id, image_urls_path)
download_file(pca_file_id, pca_path)
download_file(kmeans_file_id, kmeans_path)


model = load_model(model_path)
pca = load(pca_path)
kmeans = load(kmeans_path)
features = np.load(features_path)
labels = np.load(labels_path)
valid_image_paths = np.load(image_paths_path)
image_urls = np.load(image_urls_path)


# Initialize FastAPI app
app = FastAPI()
auth = HTTPBearer()

# Replace this with your actual API keys
API_KEYS = {
    "NAMANNEGI30576": "user1",
    "EBISU123": "user2",
}

API_KEY_NAME = "Authorization"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def verify_token(token: str = Depends(api_key_header)):
    if token in API_KEYS:
        return API_KEYS[token]
    else:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST, detail="Invalid token"
        )

@app.get('/')
async def root():
    return {"message": "Welcome to the FastAPI app"}

@app.post('/similar')
async def find_similar_images(url: str):
    if not url:
        raise HTTPException(status_code=400, detail="No URL provided")
    
    input_features = extract_features(url, model)
    if input_features is None:
        raise HTTPException(status_code=400, detail="Could not process the provided URL")
    
    input_features = normalize(input_features.reshape(1, -1))
    input_features = pca.transform(input_features)

    cluster_label = kmeans.predict(input_features)[0]
 
    cluster_indices = np.where(labels == cluster_label)[0]
    cluster_features = features[cluster_indices]
    cluster_image_paths = np.array(valid_image_paths)[cluster_indices]
    cluster_image_urls = np.array(image_urls)[cluster_indices]
    
    similar_indices = calculate_similarity(input_features, cluster_features, top_n=3)
    similar_images = [cluster_image_paths[idx] for idx in similar_indices]
    similar_urls = [cluster_image_urls[idx] for idx in similar_indices]
    
    return {'similar_images': similar_urls}

if __name__ == '__main__':
    uvicorn.run(app, port=8000)
