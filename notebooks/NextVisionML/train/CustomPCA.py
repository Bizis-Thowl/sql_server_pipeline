from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
import numpy as np

class CustomPCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2, filter_value='low'):
        self.n_components = n_components
        self.filter_value = filter_value
        self.pca = PCA(n_components=self.n_components)
        
    def fit(self, X, y=None):
        # Filter X based on y value
        if y is not None:
            filtered_X = X[y == self.filter_value]
        else:
            filtered_X = X
        
        # Fit PCA on filtered data
        self.pca.fit(filtered_X)
        return self
    
    def predict(self, X, base_threshold=0.1):
        # Calculate reconstruction error for each data point
        reconstruction_errors = np.square(X - self.inverse_transform(self.transform(X)))
        error_per_point = np.mean(reconstruction_errors, axis=1)
        
        # Classify data points based on error thresholds
        predictions = np.empty(error_per_point.shape, dtype=object)  # Use dtype=object for string labels
        
        # Apply threshold multipliers to categorize errors
        predictions[error_per_point <= base_threshold] = 'low'
        predictions[(error_per_point > base_threshold) & (error_per_point <= base_threshold * 2)] = 'low-med'
        predictions[(error_per_point > base_threshold * 2) & (error_per_point <= base_threshold * 3)] = 'medium'
        predictions[(error_per_point > base_threshold * 3) & (error_per_point <= base_threshold * 4)] = 'med-high'
        predictions[error_per_point > base_threshold * 4] = 'high'
        
        return predictions
    
    def transform(self, X, y=None):
        # Transform data using the fitted PCA
        transformed_X = self.pca.transform(X)
        return transformed_X
    
    def inverse_transform(self, X):
        # Reconstruct data back to original space
        reconstructed_X = self.pca.inverse_transform(X)
        return reconstructed_X
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
    
    def reconstruction_error(self, X):
        # Calculate and return the reconstruction error
        transformed_X = self.transform(X)
        reconstructed_X = self.inverse_transform(transformed_X)
        return np.mean(np.square(X - reconstructed_X))
