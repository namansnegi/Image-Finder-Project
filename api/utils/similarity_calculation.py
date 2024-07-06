import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

def calculate_similarity(input_features, dataset_features, top_n=3, metric='cosine'):

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
