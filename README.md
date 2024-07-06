# ImageFinder

## Repository Structure
This repository contains the code for the ImageFinder project, an image retrieval system using machine learning.

### Parts of the Repository
1. **api**: Contains all the files for the FastAPI application
2. **app**: Contains all the files for the Flask front-end application
3. **scripts**: Contains command-line scripts and Jupyter notebooks for training the model and finding similar images.

### Folder Structure
```
ImageFinder/
├── api/
│   ├── app.py
│   ├── models/
│   │   └── model_files
│   ├── utils/
│   │   └── feature_extraction.py
│   │   └── similarity_calculation.py
│   └── requirements.txt
├── app/
│   ├── static/
│   ├── templates/
│   │   └── index.html
│   └── app.py
├── scripts/
│   ├── train_save.py
│   ├── retrieve_similar_images.py
│   └── Image_Search.ipynb
└── README.md
```

## Usage Instructions

### Using Deployed Versions

I have deployed the API and the APP on the cloud. You can access it and test using the following urls:  
- **API**: Access the deployed API at `[ImageFinderAPI](https://web-646-1deb7c86-0qdgf9mi.onporter.run/docs)`.
- **App**: Access the deployed app at `[ImageFinderAPP](https://web1-649-1deb7c86-f6459y89.onporter.run/)`.

### Installation and Setup on localhost

1. Git clone the repository or downlaod it. Open terminal or your code editor and go to the repository location on your laptop/computer

2. **Install Miniconda**:
   - Download and install from [Miniconda download page](https://docs.conda.io/en/latest/miniconda.html).

3. **Create and Activate a Conda Environment**:
   ```bash
   conda create -n imagefinder python=3.11
   conda activate imagefinder
   ```

4. **Install Dependencies**:
   ```bash
   conda install numpy matplotlib pillow requests scipy joblib tensorflow scikit-learn
   pip install fastapi uvicorn jupyter flask
   ```
### Using the Components

1. **Train and Save Model**:
   Ruuning this file does the follwing main tasks:
     - Downloads all the images from the images csv file
     - Data preprcoessing and Machine learning model initialisation
     - Model Training
     - Saving all the model files and other artefacts
     
     I have already executed this file and the models and the artefacts have been saved. Only run this file if you want to retrain the machine learning model.
   
     ```bash
     cd scripts
     python train_save.py
     ```


     
     I have already executed this file and the models and the artefacts have been saved. Only run this file if you want to retrain the machine learning model.

3. **Retrieve Similar Images**:
   Running this file will use the saved model to retrieve similar images for the input image url 
     ```bash
     cd scripts
     python retrieve_similar_images.py /path/to/model_data "https://example.com/path/to/input_image.jpg"
     ```
     In the current folder structure the command will be:
     ```bash
     python retrieve_similar_images.py output "https://example.com/path/to/input_image.jpg"
     ```
      

4. **Run Jupyter Notebook**:
   You can run the whole code interactively using Jupyter Notebook
   - Start Jupyter Notebook:
     ```bash
     jupyter notebook
     ```
   - Open `scripts/Image_Search.ipynb`.

6. **Run FastAPI Application**:
   - Start the FastAPI server:
     ```bash
     cd api
     uvicorn app:app --reload
     ```
   - Access the API at `http://127.0.0.1:8000/docs`.

7. **Run Flask Front-End**:
   I have created a simple front end to use the API 
   - Start the Flask application:
     ```bash
     cd app
     python app.py
     ```
   - Access the front-end at `http://127.0.0.1:5000`. Make sure that the FastAPI server is running on `http://127.0.0.1:8000`





## Technical Choices

### Convolutional Neural Networks (CNNs)
I chose CNNs for this project because they are great at working with images. They have two main advantages:
1. **Spatial Locality**: CNNs look at small parts of the image at a time, making it easier to detect objects.
2. **Translational Invariance**: CNNs can recognize objects no matter where they are in the image because they share weights and use pooling layers.

### Transfer Learning vs. Training a New Model
Training deep learning models from scratch needs a lot of data and resources, which can be expensive and time-consuming. Transfer Learning helps solve this problem.

Transfer Learning uses a pre-trained model that has already learned useful features from a previous task (the **pretext task**). This model is then adapted to the new task (the **target task**).

1. **Reduced Training Time**: Pre-trained models like ResNet50 already know useful features, so we only need to fine-tune them, saving a lot of time.
2. **Improved Performance with Limited Data**: Since the model is already trained on a lot of data, it performs well even with less new data.

For this project, I used the pre-trained ResNet50 model because it captures detailed features well due to its deep architecture and residual connections.

### Fine-Tuning with Autoencoder
To fine-tune the ResNet50 model for finding similar images, I used an Autoencoder. Autoencoders learn efficient representations of the input data. Here’s how:

1. **Encoder**: The encoder part of the autoencoder uses the ResNet50 model up to the second last layer. It compresses input images into smaller feature vectors.
2. **Decoder**: The decoder rebuilds the input image from these feature vectors. This ensures that the feature vectors capture all important information.
3. **Training**: The autoencoder is trained to minimize the difference between the input and reconstructed images. This fine-tunes the ResNet50 encoder to produce optimal feature vectors.
4. **Feature Extraction**: After training, the ResNet50 encoder is used to extract feature vectors from all images in the database and the query image.

### Data Preprocessing and Augmentation
Before training, preprocessing and augmenting the data is crucial. Here’s what I did:

1. **Resizing**: I resized the images to 224x224 pixels because ResNet50 was trained on this size, ensuring compatibility and leveraging pre-trained weights.
2. **Normalization**: I scaled image pixel values to a range of 0 to 1 for faster training.
3. **Augmentation Techniques**:
    - **Random Flipping**
    - **Random Brightness Adjustment**
    - **Random Contrast Adjustment**
These techniques increase the diversity of the training data, helping the model generalize better to new, unseen images.

### Principal Component Analysis (PCA)
I used PCA to reduce the dimensionality of the feature vectors. PCA transforms the feature space into orthogonal components, ordered by the amount of variance they explain. This makes similarity calculations faster and more efficient without losing significant information.

### K-Means Clustering
After PCA, K-Means clustering groups similar images together. Clustering organizes the feature space into distinct groups, enhancing the retrieval system's efficiency by narrowing down the search space for similar images.

### Cosine Similarity and Euclidean Distance
- **Cosine Similarity**: Measures the cosine of the angle between two vectors. It’s useful in high-dimensional spaces where vector magnitudes can vary.
- **Euclidean Distance**: Measures the straight-line distance between two points in the feature space, providing a traditional measure of similarity.

### FastAPI Framework
I chose FastAPI for developing the API because it’s fast, easy to use, and efficient for building APIs, making it a great choice for small projects where quick development and high performance are important.

## Overall Process 
                            Training Process                                      Finding Similar Images
                                 ┌─────────────┐                                        ┌─────────────┐
                                 │ Input Images│                                        │ Input Image │
                                 └──────┬──────┘                                        └──────┬──────┘
                                        │                                                      │
                                        ▼                                                      ▼
                            ┌────────────────────┐                                  ┌────────────────────┐
                            │ Preprocessing      │                                  │ Load Artifacts     │
                            │ (Resize, Normalize,│                                  │ (Models, Features, │
                            │ Augment)           │                                  │  Clusters)         │
                            └──────┬─────────────┘                                  └──────┬─────────────┘
                                   │                                                      │
                                   ▼                                                      ▼
                            ┌────────────────────┐                                  ┌────────────────────┐
                            │    Base Model      │                                  │ Preprocessing      │
                            │    (ResNet50)      │                                  │ (Resize, Normalize)│
                            └──────┬─────────────┘                                  └──────┬─────────────┘
                                   │                                                      │
                                   ▼                                                      ▼
                            ┌────────────────────┐                                  ┌────────────────────┐
                            │   Autoencoder      │                                  │ Encoder (ResNet50) │
                            │(Fine-Tuning)       │                                  │ (Feature Extraction)│
                            └──────┬─────────────┘                                  └──────┬─────────────┘
                                   │                                                      │
                                   ▼                                                      ▼
                            ┌────────────────────┐                                  ┌────────────────────┐
                            │ Encoder (ResNet50) │                                  │       PCA          │
                            │(Feature Extraction)│                                  │(Dimensionality Red.)│
                            └──────┬─────────────┘                                  └──────┬─────────────┘
                                   │                                                      │
                                   ▼                                                      ▼
                            ┌────────────────────┐                                  ┌────────────────────┐
                            │      PCA           │                                  │     Load K-Means   │
                            │(Dimensionality Red.)│                                 │      Clusters      │
                            └──────┬─────────────┘                                  └──────┬─────────────┘
                                   │                                                      │
                                   ▼                                                      ▼
                            ┌────────────────────┐                                  ┌────────────────────┐
                            │    K-Means         │                                  │ Similarity Calc.   │
                            │   Clustering       │                                  │ (Cosine/Euclidean) │
                            └──────┬─────────────┘                                  └──────┬─────────────┘
                                   │                                                      │
                                   ▼                                                      ▼
                            ┌────────────────────┐                                  ┌────────────────────┐
                            │  Save Artifacts    │                                  │ Retrieve Similar   │
                            │(Models, Features,  │                                  │      Images        │
                            │ Clusters)          │                                  └────────────────────┘
                            └────────────────────┘


## Improvements


### Improving the Machine Learning Process
1. **Try Different Models**: Test other pre-trained models like EfficientNet, VGG16, or InceptionV3. This helps find out which model works best for extracting features and improving accuracy.
2. **Combine Multiple Models**: Use several models together to create an ensemble. This takes advantage of the strengths of each model, making the system more robust and accurate.

### Similarity Measure Enhancements
1. **Combine Similarity Measures**: Use multiple similarity measures like cosine similarity and Euclidean distance together. This can be done using a voting mechanism to get more reliable and accurate results, reducing the weaknesses of any single measure.

### Data Improvements
1. **Increase Dataset Size**: Gather more and diverse data to train and fine-tune the models. More data helps the model learn better features and improves overall performance.
