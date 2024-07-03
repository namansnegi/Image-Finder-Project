# ImageFinder

## Repository Structure
This repository contains the code for the ImageFinder project, an image retrieval system using machine learning.

### Parts of the Repository
1. **api**: Contains all the files for the FastAPI application, including the code for serving the API.
2. **app**: Contains all the files for the Flask front-end application, including HTML templates and static files.
3. **scripts**: Contains command-line scripts and Jupyter notebooks for training the model and finding similar images.

### Folder Structure
```
Image-Finder-Project/
├── api/
│   ├── app.py
│   ├── models/
│   │   └── model_files
│   ├── utils/
│   │   └── feature_extraction.py
│   │   └── similarity_calculation.py
│   └── requirements.txt
├── app/
│   ├── static
│   ├── templates/
│   │   └── index.html
│   └── app.py
├── scripts/
│   ├── train_save.py
│   ├── retrieve_similar_images.py
│   └── Image_Search.ipynb
│   ├── output/
│   │   └── model_files
│   ├── images/
│   │   └── downloaded_images
└── README.md
```

## Usage Instructions

### Installation and Setup

1. **Install Miniconda**:
   - Download and install from [Miniconda download page](https://docs.conda.io/en/latest/miniconda.html).

2. **Create and Activate a Conda Environment**:
   ```bash
   conda create -n imagefinder python=3.11
   conda activate imagefinder
   ```

3. **Install Dependencies**:
   ```bash
   conda install numpy matplotlib pillow requests scipy joblib tensorflow scikit-learn
   pip install fastapi uvicorn jupyter flask
   ```

### Using the Components

1. **Train and Save Model**:
   - Train and save the model:
     ```bash
     python scripts/train_save.py
     ```

2. **Retrieve Similar Images**:
   - Retrieve similar images based on an input image URL:
     ```bash
     python scripts/retrieve_similar_images.py /path/to/model_data "https://example.com/path/to/input_image.jpg"
     ```

3. **Run Jupyter Notebook**:
   - Start Jupyter Notebook:
     ```bash
     jupyter notebook
     ```
   - Open `scripts/Image_Search.ipynb`.

4. **Run FastAPI Application**:
   - Start the FastAPI server:
     ```bash
     uvicorn api.app:app --reload
     ```
   - Access the API at `http://127.0.0.1:8000/docs`.

5. **Run Flask Front-End**:
   - Start the Flask application:
     ```bash
     python app/app.py
     ```
   - Access the front-end at `http://127.0.0.1:5000`.

### Using Deployed Versions

- **API**: Access the deployed API at `https://web-646-1deb7c86-0qdgf9mi.onporter.run/similar`.
- **App**: Access the deployed app at `[provide-app-url]`.

## Technical Choices

### Transfer Learning with ResNet
**Reasoning**: ResNet50, a widely used convolutional neural network, is effective in capturing detailed features due to its deep architecture and residual connections, which help in learning intricate patterns in images. Using a pre-trained model leverages existing learned features from large datasets like ImageNet, significantly reducing the time required for training and improving accuracy even with limited data.
**Trade-Offs**: While using a pre-trained model speeds up development and boosts initial accuracy, it may limit the customization of the model to specific nuances of the new dataset. Fine-tuning the later layers of the pre-trained model can mitigate this, but it still requires careful adjustment.

### Fine-Tuning with Autoencoder
**Reasoning**: Autoencoders are used to fine-tune the feature extraction process, focusing on learning compact and efficient representations of the input data. By reconstructing the input from these learned representations, autoencoders can highlight and preserve important features while reducing noise.
**Trade-Offs**: Training an autoencoder adds complexity and requires additional computational resources. However, the benefit of improved feature quality and more meaningful embeddings outweighs the cost, especially for tasks involving similarity measures.

### Alternative Fine-Tuning with Contrastive Learning (Siamese Network)
**Reasoning**: Contrastive learning with a Siamese network is another effective approach for learning similarity measures. By training on pairs of similar and dissimilar images, the network learns a more nuanced embedding space where similar images are closer together, and dissimilar images are further apart. This method is particularly effective for tasks that require high precision in similarity comparisons.
**Trade-Offs**: While Siamese networks can provide highly accurate similarity measures, they require a carefully constructed training set of image pairs and more complex training procedures compared to autoencoders. The additional complexity can be justified by the improved performance in similarity tasks.

### Principal Component Analysis (PCA)
**Reasoning**: PCA is a dimensionality reduction technique that transforms the feature space into a set of orthogonal components, ordered by the amount of variance they explain in the data. Reducing the dimensionality of feature vectors using PCA enhances computational efficiency, making similarity calculations faster and more manageable without sacrificing significant information.
**Trade-Offs**: Although PCA reduces computational load, it involves some information loss. The challenge is to select an optimal number of components that balance efficiency with the preservation of essential features. This trade-off is acceptable given the significant improvements in speed and reduced computational resource requirements.

### Cosine Similarity and Euclidean Distance
**Reasoning**: Cosine similarity measures the cosine of the angle between two vectors, providing a measure of orientation similarity, which is useful in high-dimensional spaces where the magnitude of vectors might vary. Euclidean distance measures the straight-line distance between two points in the feature space, offering a more traditional measure of similarity based on distance.
**Trade-Offs**: Using both metrics allows for a comprehensive analysis of similarity, capturing both orientation and magnitude differences. However, the choice between cosine similarity and Euclidean distance can affect the results depending on the nature of the dataset and the specific application requirements. In practice, it might be necessary to experiment with both to determine the most effective measure for the given task.

### FastAPI Framework
**Reasoning**: FastAPI is chosen for developing the API due to its high performance, ease of use, and modern features such as automatic interactive API documentation, asynchronous capabilities, and Pydantic-based data validation. It allows for rapid development and efficient request handling, making it suitable for building scalable web APIs.
**Trade-Offs**: While FastAPI offers numerous advantages, it requires familiarity with asynchronous programming and Pydantic models. Additionally, it may not be as widely adopted as some older frameworks, potentially limiting the availability of certain community-contributed packages or resources. However, its performance benefits and modern features make it an excellent choice for this application.

## Improvements

### Improving the Machine Learning Process
1. **Experiment with Different Models**: Explore other pre-trained models like EfficientNet, VGG16, or InceptionV3 to compare feature extraction performance and potentially enhance accuracy.
2. **Enhance Feature Representation**: Incorporate additional layers or use advanced fine-tuning techniques such as attention mechanisms or transformer-based models to improve the quality of feature embeddings.
3. **Continuous Learning**: Implement a system for continuous learning to update the model with new data, improving accuracy over time. Set up a retraining pipeline for periodic updates to keep the model current and relevant.
4. **Ensemble Methods**: Combine multiple models to create an ensemble that leverages the strengths of different architectures, improving robustness and accuracy of feature extraction.

### Similarity Measure Enhancements
1. **Combining Similarity Measures**: Use a voting mechanism that combines multiple similarity measures (e.g., cosine similarity, Euclidean distance, and others) to improve the robustness and accuracy of similarity scoring. This approach can help mitigate the weaknesses of individual measures and provide more reliable results.

### Data Improvements
1. **Increasing Dataset Size**: Collect more diverse and extensive datasets to train and fine-tune the models. More data can help the model learn better feature representations and improve overall performance.
2. **Data Augmentation**: Apply data augmentation techniques to artificially increase the diversity of the training data, making the model more robust to variations and improving its generalization capabilities.

### Model Monitoring and Maintenance
1. **Monitor Model Performance**: Set up monitoring tools to track the model's performance in production, ensuring it maintains accuracy and efficiency over time. Use metrics such as response time, accuracy, and similarity scores to detect and address issues promptly.
2. **Scheduled Retraining**: Schedule regular retraining sessions using updated data to maintain the model's accuracy and relevance. Automated retraining pipelines can help ensure that the model evolves with the changing data landscape and continues to perform well.
