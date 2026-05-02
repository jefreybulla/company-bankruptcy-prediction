from sklearn.base import BaseEstimator, TransformerMixin

class CleanColumns(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, z):
        z = z.copy()
        z.columns = z.columns.str.strip()
        return z
