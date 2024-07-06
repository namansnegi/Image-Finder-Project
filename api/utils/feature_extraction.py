import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import tensorflow as tf
import requests
from io import BytesIO

def download_image_from_url(url, target_path):
    """"Download and save images from url"""
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

def extract_features(img_url, model, target_size=(224, 224)):
    try:
        # Download the input image from URL
        img_path = download_image_from_url(img_url, "input_image.jpg")
    
        if not img_path:
            print("Failed to download the input image.")
            return None
            
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
        print(f"Warning: Could not process image {img_path}. Skipping.")
        return None
