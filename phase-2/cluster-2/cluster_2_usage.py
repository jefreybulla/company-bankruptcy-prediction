# %% [markdown]
# # Cluster 2 Usage: Testing the Exported Predictor
# 
# This notebook demonstrates how to load and use the combined predictor (preprocessing + model) that was exported from the Cluster 2 analysis.
# 
# The predictor handles both data preprocessing and model predictions in a single call.

# %% [markdown]
# ## Setup

# %%
import pandas as pd
import joblib

# Import custom classes
from cluster_2_classes import Predictor, PreprocessorWrapper, ColumnDropper

# Load the combined predictor
predictor = joblib.load('cluster_2_predictor.joblib')

print("✓ Cluster 2 Predictor loaded successfully")
print(f"  {predictor}")

# %% [markdown]
# ## Load Data

# %%
# Load and preprocess the data
df = pd.read_csv('../../Clusters/cluster_2.csv')
X = df.drop(columns=['Bankrupt?'])
y = df['Bankrupt?']

# %% [markdown]
# ## Make Predictions

# %%
# Make predictions (preprocessing is handled automatically by the predictor)
y_pred = predictor.predict(X)

print(f"Predictions made for {len(y_pred)} samples")
print(f"\nPrediction distribution:")
print(f"  Non-Bankrupt (0): {(y_pred == 0).sum()}")
print(f"  Bankrupt (1): {(y_pred == 1).sum()}")


