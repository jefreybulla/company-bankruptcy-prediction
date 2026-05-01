"""
cluster_6_classes.py
Custom predictor class for Cluster 6 stacking model.
Required to load and run the c6_stacking_model.joblib in one line.

Dependencies: scikit-learn, joblib, pandas, numpy
"""

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


class ClusterPredictor:
    """
    Loads the saved C6 stacking model and predicts bankruptcy.

    Usage (one line):
        predictions = ClusterPredictor('Data/c6_stacking_model.joblib').predict(X_test)
    """

    def __init__(self, model_path: str):
        data                = joblib.load(model_path)
        self.cluster_id     = data['cluster_id']
        self.feature_cols   = data['feature_cols']
        self.model          = data['model']
        self.threshold      = data.get('threshold', 0.5)
        self.n_train        = data.get('n_train')
        self.n_bankrupt     = data.get('n_bankrupt')
        print(f"[C{self.cluster_id}] Model loaded | "
              f"features={len(self.feature_cols)} | "
              f"threshold={self.threshold}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict bankruptcy for companies in Cluster 6.

        Parameters
        ----------
        X : pd.DataFrame
            Raw cluster data (must contain the required feature columns).

        Returns
        -------
        np.ndarray of int (0 = healthy, 1 = bankrupt)
        """
        X_sel = X[self.feature_cols]
        probs = self.model.predict_proba(X_sel)[:, 1]
        return (probs >= self.threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns raw bankruptcy probability scores."""
        X_sel = X[self.feature_cols]
        return self.model.predict_proba(X_sel)[:, 1]
