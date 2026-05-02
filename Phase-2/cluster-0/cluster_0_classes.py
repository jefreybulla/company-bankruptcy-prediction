"""
cluster_0_classes.py
Custom predictor class for Cluster 0 stacking model.
Required to load and run the c0_stacking_model.joblib in one line.

Dependencies: scikit-learn, imbalanced-learn, joblib, pandas, numpy
"""

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


class ClusterPredictor:
    """
    Loads the saved C0 stacking model and predicts bankruptcy.

    Usage (one line):
        predictions = ClusterPredictor('c0_stacking_model.joblib').predict(X_test)
    """

    def __init__(self, model_path: str):
        data                = joblib.load(model_path)
        self.cluster_id     = data['cluster_id']
        self.feature_cols   = data['feature_cols']
        self.model          = data['model']
        self.threshold      = data.get('threshold', 0.5)
        self.n_train        = data.get('n_train')
        self.n_bankrupt     = data.get('n_bankrupt')
        print(f"[C{self.cluster_id}] Model loaded | features={len(self.feature_cols)} | threshold={self.threshold}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_sel = X[self.feature_cols]
        probs = self.model.predict_proba(X_sel)[:, 1]
        return (probs >= self.threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_sel = X[self.feature_cols]
        return self.model.predict_proba(X_sel)[:, 1]
