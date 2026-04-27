"""
Custom classes for Cluster 2 predictor unpickling.

These classes are required to load the saved joblib files:
- cluster_2_predictor.joblib
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnDropper(BaseEstimator, TransformerMixin):
    """Drops specified columns from the dataframe."""
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.drop(columns=[c for c in self.columns_to_drop if c in X.columns])
        return X


class PreprocessorWrapper(BaseEstimator, TransformerMixin):
    """
    Wrapper that correctly applies the preprocessing pipeline.
    
    Handles:
    - Column dropping
    - Feature scaling (MinMaxScaler on specific columns)
    - Variance threshold filtering
    - Dimensionality reduction (PCA)
    """
    def __init__(self, column_dropper, scaler, pca, variance_selector=None):
        self.column_dropper = column_dropper
        self.scaler = scaler
        self.pca = pca
        self.variance_selector = variance_selector
        self.scaler_columns = list(scaler.feature_names_in_)
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Step 1: Drop columns
        X_dropped = self.column_dropper.transform(X)
        
        # Step 2: Apply scaler only to the columns it was fitted on
        X_scaled = X_dropped.copy()
        X_scaled[self.scaler_columns] = self.scaler.transform(X_dropped[self.scaler_columns])
        
        # Step 3: Apply variance threshold (if present)
        if self.variance_selector is not None:
            X_scaled = X_scaled.loc[:, self.variance_selector.get_support()]
        
        # Step 4: Apply PCA
        return self.pca.transform(X_scaled)


class Predictor(BaseEstimator):
    """
    Combined predictor that handles preprocessing and model prediction in one call.
    
    Usage:
        from cluster_2_classes import Predictor
        import joblib
        
        predictor = joblib.load('cluster_2_predictor.joblib')
        predictions = predictor.predict(new_data)
        probabilities = predictor.predict_proba(new_data)
    """
    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model
    
    def predict(self, X):
        """Preprocess data and make predictions."""
        X_preprocessed = self.preprocessor.transform(X)
        return self.model.predict(X_preprocessed)
    
    def predict_proba(self, X):
        """Preprocess data and return probability estimates."""
        X_preprocessed = self.preprocessor.transform(X)
        return self.model.predict_proba(X_preprocessed)
    
    def __repr__(self):
        return f"Predictor(preprocessor={type(self.preprocessor).__name__}, model={type(self.model).__name__})"
