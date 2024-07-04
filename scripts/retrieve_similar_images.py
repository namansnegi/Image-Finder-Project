import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from joblib import load
import requests
from io import BytesIO
import sys
from pathlib import Path
import logging

def extract_features(img_path, model, target_size=(224, 224)):
    try:
        # Read and preprocess image
        img = tf.io.read_file(img_path)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.resize(img, target_size)
        img = preprocess_input(img)  # Preprocess input to match ResNet50 expectations
        
        # Expand dimensions to match model input
        img = tf.expand_dims(img, axis=0)  # Shape: (1, height, width, channels)
        
        # Extract features using the model
        features = model.predict(img)
        
        return features.flatten()
    except tf.errors.InvalidArgumentError:
        print("Warning: Could not process image {}. Skipping.".format(img_path))
        return None

def calculate_similarity(input_features, dataset_features, top_n=3, metric='cosine'):
    """
    Calculate the similarity between input features and dataset features, and return the top N similar items.
    
    Parameters:
    - input_features: Feature vector of the input image.
    - dataset_features: Feature matrix of the dataset images.
    - top_n: Number of top similar items to return.
    - metric: Similarity metric to use ('cosine' or 'euclidean').
    
    Returns:
    - Indices of the top N similar items in the dataset.
    """
    if len(input_features.shape) == 1:
        input_features = input_features.reshape(1, -1)
    
    if metric == 'cosine':
        similarity_scores = cosine_similarity(input_features, dataset_features)
        indices = np.argsort(similarity_scores[0])[::-1]
    elif metric == 'euclidean':
        similarity_scores = euclidean_distances(input_features, dataset_features)
        indices = np.argsort(similarity_scores[0])
    else:
        raise ValueError("Unsupported metric: {}. Choose 'cosine' or 'euclidean'.".format(metric))
    
    return indices[1:top_n+1]

def download_image_from_url(url, target_path):
    """Download image from URL and save it to target path."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(target_path, 'wb') as handle:
            for block in response.iter_content(1024):
                if not block:
                    break
                handle.write(block)
        return target_path
    except requests.exceptions.RequestException as e:
        print("Failed to download {}: {}".format(url, e))
        return None


def display_similar_images(input_image_url, image_paths,image_urls, features, labels, model, pca, kmeans, top_n=3, target_size=(224, 224), metric='cosine'):

        # Download the input image from URL
    input_image_path = download_image_from_url(input_image_url, "input_image.jpg")
    
    if not input_image_path:
        print("Failed to download the input image.")
        return

    input_features = extract_features(input_image_path, model)
    input_features= normalize(input_features.reshape(1, -1))
    input_features = pca.transform(input_features)  # Encode input features
    
    # Find the cluster of the input image
    cluster_label = kmeans.predict(input_features)[0]
    
    # Get the indices of images in the same cluster
    cluster_indices = np.where(labels == cluster_label)[0]
    
    # Get the features of images in the same cluster
    cluster_features = features[cluster_indices]
    cluster_image_paths = np.array(image_paths)[cluster_indices]
    cluster_image_urls = np.array(image_urls)[cluster_indices]
    
    # Calculate similarity
    similar_indices = calculate_similarity(input_features, cluster_features, top_n=top_n)
    
    # Display the input image and the most similar images
    plt.figure(figsize=(15, 5))
    plt.subplot(1, top_n + 1, 1)
    img = Image.open(input_image_path).resize(target_size, Image.LANCZOS)
    plt.imshow(img)
    plt.title("Input Image")
    plt.axis('off')
    
    # Verify and display similar images
    similar_image_urls = []
    for i, idx in enumerate(similar_indices):
        similar_image_path = cluster_image_paths[idx]
        similar_image_url = cluster_image_urls[idx]
        similar_image_urls.append(similar_image_url)
        
        plt.subplot(1, top_n + 1, i + 2)
        img = Image.open(similar_image_path).resize(target_size, Image.LANCZOS)
        plt.imshow(img)
        plt.title("Similar Image {}".format(i + 1))
        plt.axis('off')
    
    plt.show()
    return similar_image_urls

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python retrieve_similar_images.py <model_data_folder> <input_image_url>")
        sys.exit(1)

    model_data_folder = sys.argv[1]
    input_image_url = sys.argv[2]

    # Convert to absolute paths
    model_data_path = Path(model_data_folder).resolve()

    logging.info("Model data folder: {}".format(model_data_path))
    logging.info("Input image URL: {}".format(input_image_url))

    # Load models and data
    encoder_model = load_model(model_data_path / 'fine_tuned_resnet_model.h5')
    pca = load(model_data_path / 'pca_model.joblib')
    kmeans = load(model_data_path / 'kmeans_model.joblib')
    features = np.load(model_data_path / 'features.npy')
    labels = np.load(model_data_path / 'labels.npy')
    image_paths = np.load(model_data_path / 'image_paths.npy')
    image_urls = np.load(model_data_path / 'image_urls.npy')

    # Get similar images
    similar_image_urls = display_similar_images(input_image_url, image_paths, image_urls, features, labels, encoder_model, pca, kmeans, top_n=3)

    # Print similar image URLs
    for url in similar_image_urls:
        print(url)
