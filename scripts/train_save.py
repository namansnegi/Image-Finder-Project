import argparse
import concurrent.futures
import csv
import time
from pathlib import Path
import requests
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, UpSampling2D
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from scipy.spatial.distance import mahalanobis
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Reshape
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')
from joblib import dump, load
import os

def download_image(url):
    """Download and save images from urls"""
    filename = url.split("/")[-1]
    file = Path("./images").joinpath(filename)
    file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with file.open("wb") as handle:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            for block in response.iter_content(1024):
                if not block:
                    break
                handle.write(block)
        return file
    except requests.exceptions.RequestException as e:
        print("Failed to download {}: {}".format(url, e))
        return None

def is_valid_image(file_path):
    """Check if a file is a valid image"""
    try:
        img = tf.io.read_file(str(file_path))
        img = tf.image.decode_image(img, channels=3)
        return True
    except:
        return False

def process_image(data):
    url, filename = data
    file_path = download_image(url)
    if file_path and is_valid_image(file_path):
        return (str(file_path), url)
    return None

def initialize_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    encoder_output = base_model.output
    x = Conv2DTranspose(256, (3, 3), padding='same', activation='relu')(encoder_output)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(128, (3, 3), padding='same', activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(64, (3, 3), padding='same', activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(32, (3, 3), padding='same', activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(16, (3, 3), padding='same', activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded_output = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    autoencoder = Model(inputs=base_model.input, outputs=decoded_output)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, base_model

def load_and_augment(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_image(img, channels=3)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, image_size)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    img = tf.cast(img, tf.float32) / 255.0
    return img, img 

def load_and_preprocess(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_image(img, channels=3)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, image_size)
    img = tf.cast(img, tf.float32) / 255.0
    return img, img  

def extract_features(img_path, model, target_size=(224, 224)):
    try:
        # Read and preprocess image
        img = tf.io.read_file(img_path)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.resize(img, target_size)
        img = preprocess_input(img)
        
        # Expand dimensions to match model input
        img = tf.expand_dims(img, axis=0) 
        
        # Extract features using the model
        features = model.predict(img)
        
        return features.flatten()
    except tf.errors.InvalidArgumentError:
        print("Warning: Could not process image {}. Skipping.".format(img_path))
        return None

# Parse arguments for CSV file
parser = argparse.ArgumentParser()
parser.add_argument("--csv_file", help="CSV file with image URLs")
args = parser.parse_args(args=["--csv_file", "./images.csv"])

# Read image URLs from CSV file
with open(args.csv_file, "r") as handle:
    reader = csv.DictReader(handle, delimiter=';')
    image_data = [(row['url'], row['url'].split("/")[-1]) for row in reader] 

t = time.perf_counter()

valid_images = []
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    results = executor.map(process_image, image_data)
    for result in results:
        if result is not None:
            valid_images.append(result)

image_paths, image_urls = zip(*valid_images)
image_paths, image_urls = list(image_paths), list(image_urls)

print("Downloaded and validated {} images in {:.2f} seconds".format(len(image_paths), time.perf_counter() - t))

# Declare Constants
image_size = (224, 224) 
batch_size = 32  
split_ratio = 0.8 

combined = list(zip(image_paths, image_urls))
np.random.shuffle(combined)
image_paths, image_urls = map(list, zip(*combined))

# Split into training and validation file lists
split_index = int(len(image_paths) * split_ratio)
train_files = image_paths[:split_index]
val_files = image_paths[split_index:]

# Create datasets for training and validation
train_dataset = tf.data.Dataset.from_tensor_slices(train_files)
train_dataset = train_dataset.map(load_and_augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices(val_files)
val_dataset = val_dataset.map(load_and_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

for batch in train_dataset.take(1):
    print("Shape of batch of images with augmentation:", batch[0].shape)
for batch in val_dataset.take(1):
    print("Shape of batch of images without augmentation:", batch[0].shape)

# Initialize model
autoencoder, base_model = initialize_model()
autoencoder.summary()

# Train the autoencoder
history = autoencoder.fit(train_dataset, epochs=1, validation_data=val_dataset)

# Extract encoder model from the autoencoder
encoder_model = Model(inputs=autoencoder.input, outputs=base_model.output)
encoder_model.compile(optimizer='adam', loss='mse')
encoder_model.summary()


# Extract features from the images
features = []
valid_image_paths = []
valid_image_urls = []
for img_path, img_url in zip(image_paths, image_urls):
    feature = extract_features(img_path, encoder_model) 
    if feature is not None:
        features.append(feature)
        valid_image_paths.append(img_path)
        valid_image_urls.append(img_url)
features = np.array(features)

# Normalize features
features = normalize(features, axis=1)

# Dimension Reduction with PCA
n_components = min(features.shape[0], features.shape[1]) - 1
pca = PCA(n_components=n_components)
features = pca.fit_transform(features)

# Cluster the images
n_clusters = 15
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(features.astype(np.float32))

# Save PCA, KMeans models, and other data

output_folder = "./output"
output_path = Path(output_folder)
output_path.mkdir(parents=True, exist_ok=True)

dump(pca, output_path / 'pca_model.joblib')
dump(kmeans, output_path / 'kmeans_model.joblib')
encoder_model.save(str(output_path / 'fine_tuned_resnet_model.h5'))
np.save(output_path / 'features.npy', features)
np.save(output_path / 'labels.npy', labels)
np.save(output_path / 'image_paths.npy', valid_image_paths)
np.save(output_path / 'image_urls.npy', valid_image_urls)

print("Model and data saved in {}".format(output_folder))
