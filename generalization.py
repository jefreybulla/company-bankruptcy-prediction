# %% [markdown]
# # Generalization
# This notebook contains the inference for the test data set

# %%
import sys
import pandas as pd
import numpy as np

sys.path.append('..')
from modules.clustering import load_cluster_classifier
from modules.preprocessing import process_data

# Load the test data
print("Loading test data...")
test_df = pd.read_csv('./data/test_data.csv', skipinitialspace=True)
# Remove leading spaces from column names
test_df.columns = test_df.columns.str.strip()
print(f"Test data shape: {test_df.shape}")

# Process the test data (same preprocessing as clustering)
print("\nProcessing test data...")
test_processed = process_data(test_df, './EDA/preprocessing_artifacts.pkl')

# Load the saved cluster classifier
print("\nLoading cluster classifier...")
classifier_path = './clustering/cluster_classifier.pkl'
classifier = load_cluster_classifier(classifier_path)

# Make predictions on test data to get cluster IDs
print("\nMaking cluster predictions on test data...")
cluster_predictions = classifier.predict(test_processed.values)

print(f"\n{'='*50}")
print(f"CLUSTER PREDICTIONS")
print(f"{'='*50}")
print(f"Total samples predicted: {len(cluster_predictions)}")
print(f"Unique cluster IDs predicted: {np.unique(cluster_predictions)}")
print(f"Cluster distribution:")
unique, counts = np.unique(cluster_predictions, return_counts=True)
for cluster_id, count in zip(unique, counts):
    percentage = (count / len(cluster_predictions)) * 100
    print(f"  Cluster {cluster_id}: {count} samples ({percentage:.2f}%)")

# %%
test_processed.shape


