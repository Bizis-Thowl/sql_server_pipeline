import pandas as pd
import numpy as np
from pytest import fixture
from sklearn.decomposition import PCA
from custom_pca import CustomPCA

@fixture
def mock_data():
    np.random.seed(0)  # Ensure reproducibility
    X = pd.DataFrame(np.random.randn(100, 4), columns=list('ABCD'))
    y = pd.Series(np.random.choice([0, 1], size=100))
    return X, y

@fixture
def your_class_instance():
    instance = CustomPCA()
    instance.pca = PCA(n_components=2)  # Mock PCA with 2 components
    instance.filter_value = 0  # Set filter_value for testing
    return instance

def test_fit_with_y_none(your_class_instance, mock_data):
    X, _ = mock_data
    instance = your_class_instance
    result = instance.fit(X)
    assert instance.pca.n_components_ == 2  # Assumes PCA is fitted if this passes
    assert result == instance  # Ensure fit method returns self

def test_fit_with_y_provided(your_class_instance, mock_data):
    X, y = mock_data
    instance = your_class_instance
    result = instance.fit(X, y)
    assert instance.pca.n_components_ == 2  # PCA should be fitted on filtered data
    assert result == instance  # Ensure fit method returns self

def test_fit_with_no_matching_y(your_class_instance, mock_data):
    X, y = mock_data
    instance = your_class_instance
    instance.filter_value = 999  # Value not present in y
    # Attempt to fit, this should not raise an error but won't fit PCA due to no data
    try:
        instance.fit(X, y)
        no_error_occurred = True
    except AttributeError:
        no_error_occurred = False
    assert no_error_occurred  # Simply checks that no error occurs

def test_fit_with_empty_X(your_class_instance):
    X = pd.DataFrame()
    y = pd.Series()
    instance = your_class_instance
    # Attempt to fit, this should not raise an error but won't fit PCA due to no data
    try:
        instance.fit(X, y)
        no_error_occurred = True
    except AttributeError:
        no_error_occurred = False
    assert no_error_occurred  # Simply checks that no error occurs

def test_fit_with_missing_values(your_class_instance, mock_data):
    X, y = mock_data
    # Introduce NaN values into X
    X.iloc[0, 0] = np.nan
    
    instance = your_class_instance
    # Attempt to fit, expecting it to handle NaN values (e.g., by dropping them)
    try:
        instance.fit(X, y)
        no_error_occurred = True
    except ValueError as e:
        no_error_occurred = False
        assert "Input X contains NaN" not in str(e)  # Ensure the specific NaN error is not thrown
    
    assert no_error_occurred  # Simply checks that no error occurs and NaN handling is in place
