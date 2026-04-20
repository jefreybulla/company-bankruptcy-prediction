# %% [markdown]
# # Clustering

# %%
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '../modules')
from preprocessing import process_data

# %%
# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
import random
random.seed(RANDOM_SEED)

# %%
# Load and process test data
train_df_raw = pd.read_csv('../data/train_data.csv')

# Remove leading/trailing spaces from column names
train_df_raw.columns = train_df_raw.columns.str.strip()

print("Raw train data loaded:")
print(f"Shape: {train_df_raw.shape}")
print(f"Columns: {len(train_df_raw.columns)}")

# %%
# Process test data using our function
train_df_processed = process_data(train_df_raw, artifacts_path='../EDA/preprocessing_artifacts.pkl')

print(f"\n{'='*70}")
print("Train data after processing:")
print(f"Shape: {train_df_processed.shape}")
print(f"Columns: {len(train_df_processed.columns)}")

# %%
train_df_processed.head()

# %%
train_df_processed.describe()

# %%
attributes = train_df_processed.drop(columns=["Bankrupt?"])
attributes.shape

# %%
attributes.head()

# %% [markdown]
# # K-means Clustering

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Data is already scaled during preprocessing
print(f"Using preprocessed data with shape: {attributes.shape}")
print(f"Data range: min={attributes.min().min():.4f}, max={attributes.max().max():.4f}")

# Implement K-means with different k values
k_values = range(2, 11)
inertias = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
    kmeans.fit(attributes.values)
    inertias.append(kmeans.inertia_)
    print(f"K={k}: Inertia = {kmeans.inertia_:.4f}")

# Plot inertia versus k (Elbow method)
plt.figure(figsize=(10, 6))
plt.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (k)', fontsize=12)
plt.ylabel('Inertia', fontsize=12)
plt.title('Elbow Method - Inertia vs Number of Clusters', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xticks(k_values)
plt.tight_layout()
plt.show()

# %% [markdown]
# We will choose the elbow at k = 8

# %%
optimal_k = 8

# %%
# Fit K-means with optimal k
final_kmeans = KMeans(n_clusters=optimal_k, random_state=RANDOM_SEED, n_init=10)
cluster_labels = final_kmeans.fit_predict(attributes.values)

# Create a dataframe with cluster labels and bankruptcy status
clustering_results = pd.DataFrame({
    'Cluster': cluster_labels,
    'Bankrupt?': train_df_processed['Bankrupt?'].values
})

# Analyze each cluster
print(f"{'='*70}")
print(f"Cluster Analysis (k={optimal_k})")
print(f"{'='*70}\n")

cluster_stats = []

for cluster in range(optimal_k):
    cluster_data = clustering_results[clustering_results['Cluster'] == cluster]
    total_samples = len(cluster_data)
    bankrupt_count = (cluster_data['Bankrupt?'] == 1).sum()
    non_bankrupt_count = total_samples - bankrupt_count
    bankrupt_pct = (bankrupt_count / total_samples) * 100
    
    cluster_stats.append({
        'Cluster': cluster,
        'Total Samples': total_samples,
        'Bankrupt (1)': bankrupt_count,
        'Non-Bankrupt (0)': non_bankrupt_count,
        'Bankrupt %': f"{bankrupt_pct:.2f}%"
    })
    
    print(f"Cluster {cluster}:")
    print(f"  Total samples: {total_samples}")
    print(f"  Bankrupt (1): {bankrupt_count}")
    print(f"  Non-Bankrupt (0): {non_bankrupt_count}")
    print(f"  Bankruptcy rate: {bankrupt_pct:.2f}%\n")

# Create a summary dataframe
stats_df = pd.DataFrame(cluster_stats)
print(f"{'='*70}")
print("Summary Table:")
print(f"{'='*70}")
print(stats_df.to_string(index=False))

print(f"\n{'='*70}")
print(f"Total samples: {len(clustering_results)}")
total_bankrupt = (train_df_processed['Bankrupt?'] == 1).sum()
print(f"Overall bankruptcy count: {total_bankrupt}")
print(f"Overall bankruptcy rate: {(total_bankrupt/len(clustering_results))*100:.2f}%")

# %% [markdown]
# # Gaussian Mixture Model (GMM) Clustering

# %%
from sklearn.mixture import GaussianMixture

# Implement Gaussian Mixture Model with different numbers of components
print("Gaussian Mixture Model (GMM) Clustering")
print(f"{'='*70}")

n_components_range = range(2, 11)
bic_scores = []
aic_scores = []
neg_log_likelihood = []

for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, random_state=RANDOM_SEED, n_init=10)
    gmm.fit(attributes.values)
    bic_scores.append(gmm.bic(attributes.values))
    aic_scores.append(gmm.aic(attributes.values))
    neg_log_likelihood.append(-gmm.score(attributes.values) * len(attributes))
    print(f"Components={n_components}: BIC = {bic_scores[-1]:.2f}, AIC = {aic_scores[-1]:.2f}")

# Plot BIC and AIC for model selection (Elbow method for GMM)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# BIC plot
ax1.plot(n_components_range, bic_scores, 'mo-', linewidth=2, markersize=8)
ax1.set_xlabel('Number of Components', fontsize=12)
ax1.set_ylabel('BIC Score', fontsize=12)
ax1.set_title('GMM Model Selection - BIC vs Number of Components', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(n_components_range)

# AIC plot
ax2.plot(n_components_range, aic_scores, 'co-', linewidth=2, markersize=8)
ax2.set_xlabel('Number of Components', fontsize=12)
ax2.set_ylabel('AIC Score', fontsize=12)
ax2.set_title('GMM Model Selection - AIC vs Number of Components', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(n_components_range)

plt.tight_layout()
plt.show()

print(f"\n{'='*70}")
print("GMM Elbow Analysis Complete")
print(f"Lower BIC/AIC indicates better fit (optimal balance between model fit and complexity)")
print(f"Note: GMM uses BIC/AIC for model selection, not inertia like K-means")

# %% [markdown]
# # GMM Cluster Analysis (8 Components)

# %%
# Fit GMM with 8 components for comparison with K-means
optimal_components_gmm = 8
gmm_final = GaussianMixture(n_components=optimal_components_gmm, random_state=RANDOM_SEED, n_init=10)
gmm_cluster_labels = gmm_final.fit_predict(attributes.values)

# Get soft assignments (probabilities)
gmm_probabilities = gmm_final.predict_proba(attributes.values)

# Create a dataframe with cluster labels and bankruptcy status
gmm_clustering_results = pd.DataFrame({
    'Component': gmm_cluster_labels,
    'Bankrupt?': train_df_processed['Bankrupt?'].values
})

# Analyze each component
print(f"{'='*70}")
print(f"GMM Cluster Analysis (components={optimal_components_gmm})")
print(f"{'='*70}\n")

gmm_cluster_stats = []

for component in range(optimal_components_gmm):
    component_data = gmm_clustering_results[gmm_clustering_results['Component'] == component]
    total_samples = len(component_data)
    bankrupt_count = (component_data['Bankrupt?'] == 1).sum()
    non_bankrupt_count = total_samples - bankrupt_count
    bankrupt_pct = (bankrupt_count / total_samples) * 100
    
    gmm_cluster_stats.append({
        'Component': component,
        'Total Samples': total_samples,
        'Bankrupt (1)': bankrupt_count,
        'Non-Bankrupt (0)': non_bankrupt_count,
        'Bankrupt %': f"{bankrupt_pct:.2f}%"
    })
    
    print(f"Component {component}:")
    print(f"  Total samples: {total_samples}")
    print(f"  Bankrupt (1): {bankrupt_count}")
    print(f"  Non-Bankrupt (0): {non_bankrupt_count}")
    print(f"  Bankruptcy rate: {bankrupt_pct:.2f}%\n")

# Create a summary dataframe
gmm_stats_df = pd.DataFrame(gmm_cluster_stats)
print(f"{'='*70}")
print("Summary Table:")
print(f"{'='*70}")
print(gmm_stats_df.to_string(index=False))

print(f"\n{'='*70}")
print(f"Total samples: {len(gmm_clustering_results)}")
total_bankrupt = (train_df_processed['Bankrupt?'] == 1).sum()
print(f"Overall bankruptcy count: {total_bankrupt}")
print(f"Overall bankruptcy rate: {(total_bankrupt/len(gmm_clustering_results))*100:.2f}%")

print(f"\n{'='*70}")
print("GMM vs K-means Comparison:")
print(f"{'='*70}")
print(f"K-means: Highest bankruptcy rate = Cluster 3 (6.47%)")
print(f"GMM: Highest bankruptcy rate = Component {gmm_stats_df.loc[gmm_stats_df['Bankrupt %'].str.rstrip('%').astype(float).idxmax(), 'Component']} ({gmm_stats_df['Bankrupt %'].iloc[gmm_stats_df['Bankrupt %'].str.rstrip('%').astype(float).idxmax()]})")
print(f"K-means: Lowest bankruptcy rate = Cluster 6 (2.07%)")
print(f"GMM: Lowest bankruptcy rate = Component {gmm_stats_df.loc[gmm_stats_df['Bankrupt %'].str.rstrip('%').astype(float).idxmin(), 'Component']} ({gmm_stats_df['Bankrupt %'].iloc[gmm_stats_df['Bankrupt %'].str.rstrip('%').astype(float).idxmin()]})")

# %% [markdown]
# # Clustering Comparison: K-means vs GMM

# %%
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, normalized_mutual_info_score
import warnings
warnings.filterwarnings('ignore')

print(f"{'='*80}")
print("COMPREHENSIVE CLUSTERING EVALUATION")
print(f"{'='*80}\n")

# Get the true labels
true_labels = train_df_processed['Bankrupt?'].values

# 1. Silhouette Score (higher is better, range -1 to 1)
print("1. SILHOUETTE SCORE (higher is better, range: -1 to 1). Measures how similar samples are to their own cluster vs. other clusters")
print("-" * 80)
silhouette_kmeans = silhouette_score(attributes.values, cluster_labels)
silhouette_gmm = silhouette_score(attributes.values, gmm_cluster_labels)
print(f"K-means (k=8):        {silhouette_kmeans:.4f}")
print(f"GMM (components=8):   {silhouette_gmm:.4f}")
print(f"Winner: {'K-means' if silhouette_kmeans > silhouette_gmm else 'GMM'} (+{abs(silhouette_kmeans - silhouette_gmm):.4f})")

# 2. Davies-Bouldin Index (lower is better)
print(f"\n2. DAVIES-BOULDIN INDEX (lower is better). Measures average similarity between clusters")
print("-" * 80)
davies_bouldin_kmeans = davies_bouldin_score(attributes.values, cluster_labels)
davies_bouldin_gmm = davies_bouldin_score(attributes.values, gmm_cluster_labels)
print(f"K-means (k=8):        {davies_bouldin_kmeans:.4f}")
print(f"GMM (components=8):   {davies_bouldin_gmm:.4f}")
print(f"Winner: {'K-means' if davies_bouldin_kmeans < davies_bouldin_gmm else 'GMM'}")

# 3. Calinski-Harabasz Index (higher is better)
print(f"\n3. CALINSKI-HARABASZ INDEX (higher is better). Ratio of between-cluster to within-cluster distances")
print("-" * 80)
calinski_kmeans = calinski_harabasz_score(attributes.values, cluster_labels)
calinski_gmm = calinski_harabasz_score(attributes.values, gmm_cluster_labels)
print(f"K-means (k=8):        {calinski_kmeans:.2f}")
print(f"GMM (components=8):   {calinski_gmm:.2f}")
print(f"Winner: {'K-means' if calinski_kmeans > calinski_gmm else 'GMM'}")

# 4. Normalized Mutual Information (higher is better, measures alignment with true labels)
print(f"\n4. NORMALIZED MUTUAL INFORMATION (higher is better). How much information clustering has about bankruptcy")
print("-" * 80)
nmi_kmeans = normalized_mutual_info_score(true_labels, cluster_labels)
nmi_gmm = normalized_mutual_info_score(true_labels, gmm_cluster_labels)
print(f"K-means (k=8):        {nmi_kmeans:.4f}")
print(f"GMM (components=8):   {nmi_gmm:.4f}")
print(f"Winner: {'K-means' if nmi_kmeans > nmi_gmm else 'GMM'}")
print(f"\nInterpretation: How well does the clustering align with bankruptcy labels?")

# 5. Bankruptcy Risk Separation
print(f"\n5. BANKRUPTCY RISK STRATIFICATION")
print("-" * 80)

# Extract bankruptcy percentages
kmeans_bankruptcy_rates = [float(x.rstrip('%')) for x in stats_df['Bankrupt %']]
gmm_bankruptcy_rates = [float(x.rstrip('%')) for x in gmm_stats_df['Bankrupt %']]

kmeans_spread = max(kmeans_bankruptcy_rates) - min(kmeans_bankruptcy_rates)
gmm_spread = max(gmm_bankruptcy_rates) - min(gmm_bankruptcy_rates)

print(f"K-means bankruptcy rate range:  {min(kmeans_bankruptcy_rates):.2f}% - {max(kmeans_bankruptcy_rates):.2f}%")
print(f"GMM bankruptcy rate range:      {min(gmm_bankruptcy_rates):.2f}% - {max(gmm_bankruptcy_rates):.2f}%")
print(f"K-means spread (max - min):     {kmeans_spread:.2f}%")
print(f"GMM spread (max - min):         {gmm_spread:.2f}%")
print(f"Winner: {'K-means' if kmeans_spread > gmm_spread else 'GMM'} (better risk stratification)")

# Summary table
print(f"\n{'='*80}")
print("SUMMARY SCORECARD")
print(f"{'='*80}\n")

comparison_df = pd.DataFrame({
    'Metric': [
        'Silhouette Score',
        'Davies-Bouldin Index',
        'Calinski-Harabasz Index',
        'Normalized Mutual Info',
        'Bankruptcy Separation'
    ],
    'K-means': [
        f"{silhouette_kmeans:.4f}",
        f"{davies_bouldin_kmeans:.4f}",
        f"{calinski_kmeans:.2f}",
        f"{nmi_kmeans:.4f}",
        f"{kmeans_spread:.2f}%"
    ],
    'GMM': [
        f"{silhouette_gmm:.4f}",
        f"{davies_bouldin_gmm:.4f}",
        f"{calinski_gmm:.2f}",
        f"{nmi_gmm:.4f}",
        f"{gmm_spread:.2f}%"
    ],
    'Better (Higher/Lower)': [
        'Higher',
        'Lower',
        'Higher',
        'Higher',
        'Higher'
    ]
})

print(comparison_df.to_string(index=False))


# %% [markdown]
# ### Findings
# K-means Wins (3/5 metrics):
# 
# - Silhouette Score: 0.1324 vs 0.0301 - K-means clusters are more cohesive and separated
# - Davies-Bouldin Index: 1.7671 vs 3.8407 - K-means clusters are more distinct
# - Calinski-Harabasz Index: 587.13 vs 233.37 - K-means has better overall cluster quality
# 
# GMM Wins (2/5 metrics):
# 
# - Normalized Mutual Information: 0.0086 vs 0.0023 - GMM aligns better with bankruptcy labels
# - Bankruptcy Risk Stratification: 11.38% spread vs 4.40% spread - GMM creates more distinct risk groups (1.54%-12.92% vs 2.07%-6.47%)

# %% [markdown]
# ## Clustering Choice
# Given the experiement results, K-means overall offers better quality and separation so we will choose K-means cluster. 

# %% [markdown]
# # Characteristics of each Cluster

# %%
# Cluster Characteristics Analysis
print(f"{'='*80}")
print("CLUSTER CHARACTERISTICS ANALYSIS")
print(f"{'='*80}\n")

# Get cluster centers
cluster_centers = final_kmeans.cluster_centers_
feature_names = attributes.columns.tolist()

# Create a dataframe of cluster centers for easier analysis
centers_df = pd.DataFrame(cluster_centers, columns=feature_names)

# For each cluster, identify the most distinctive features
print("Top 5 Most Distinctive Features Per Cluster (highest absolute deviation from global mean):\n")

global_mean = attributes.values.mean(axis=0)

for cluster_id in range(optimal_k):
    print(f"{'='*80}")
    print(f"CLUSTER {cluster_id} - {len(clustering_results[clustering_results['Cluster'] == cluster_id])} samples (Bankruptcy rate: {stats_df[stats_df['Cluster'] == cluster_id]['Bankrupt %'].values[0]})")
    print(f"{'='*80}")
    
    # Calculate deviation from global mean for this cluster
    cluster_center = cluster_centers[cluster_id]
    deviations = np.abs(cluster_center - global_mean)
    
    # Get top 5 features with largest deviations
    top_5_idx = np.argsort(deviations)[::-1][:5]
    
    print("\nMost Distinctive Features:")
    for rank, feat_idx in enumerate(top_5_idx, 1):
        feat_name = feature_names[feat_idx]
        cluster_val = cluster_center[feat_idx]
        global_val = global_mean[feat_idx]
        deviation = deviations[feat_idx]
        direction = "HIGHER" if cluster_val > global_val else "LOWER"
        
        print(f"  {rank}. {feat_name}")
        print(f"     Cluster value: {cluster_val:.4f} | Global mean: {global_val:.4f}")
        print(f"     {direction} by {deviation:.4f}\n")

# Create a heatmap of cluster centers
print(f"\n{'='*80}")
print("Cluster Center Heatmap (All Features)")
print(f"{'='*80}\n")

fig, ax = plt.subplots(figsize=(16, 8))
im = ax.imshow(cluster_centers, cmap='RdBu_r', aspect='auto')

# Set ticks and labels
ax.set_xticks(np.arange(len(feature_names)))
ax.set_yticks(np.arange(optimal_k))
ax.set_xticklabels(feature_names, rotation=90, fontsize=8)
ax.set_yticklabels([f'Cluster {i}' for i in range(optimal_k)], fontsize=10)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Feature Value', rotation=270, labelpad=20)

ax.set_title('K-means Cluster Centers (All Features)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Statistical Summary by Cluster
print(f"\n{'='*80}")
print("Feature Variance Across Clusters (identifying discriminative features)")
print(f"{'='*80}\n")

# Calculate which features have high variance across clusters
feature_variances = []
for feat_idx in range(len(feature_names)):
    cluster_vals = cluster_centers[:, feat_idx]
    variance = np.var(cluster_vals)
    feature_variances.append((feature_names[feat_idx], variance))

# Sort by variance (descending)
feature_variances.sort(key=lambda x: x[1], reverse=True)

print("Top 10 Features with Highest Variance Across Clusters:")
print("(These features are most important for distinguishing clusters)\n")
for rank, (feat_name, variance) in enumerate(feature_variances[:10], 1):
    print(f"{rank:2d}. {feat_name:<40s} Variance: {variance:.6f}")


# %% [markdown]
# # Save each clusters to distinc CSV files

# %%
# Create comprehensive dataframe with all information
data_with_clusters = attributes.copy()
data_with_clusters['Cluster_ID'] = cluster_labels
data_with_clusters['Bankrupt?'] = train_df_processed['Bankrupt?'].values

print(f"{'='*80}")
print("SAVING CLUSTERS TO CSV FILES")
print(f"{'='*80}\n")

# Create output directory if it doesn't exist
output_dir = '../data/clusters'
import os
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}\n")

# Save each cluster to a separate CSV file
for cluster_id in range(optimal_k):
    cluster_data = data_with_clusters[data_with_clusters['Cluster_ID'] == cluster_id]
    
    # Generate filename
    filename = f"{output_dir}/cluster_{cluster_id}.csv"
    
    # Save to CSV
    cluster_data.to_csv(filename, index=False)
    
    # Print summary
    num_samples = len(cluster_data)
    num_bankrupt = (cluster_data['Bankrupt?'] == 1).sum()
    bankruptcy_rate = (num_bankrupt / num_samples) * 100
    
    print(f"✓ Cluster {cluster_id}: {filename}")
    print(f"  Samples: {num_samples} | Bankrupt: {num_bankrupt} ({bankruptcy_rate:.2f}%)")

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}\n")
print(f"Total clusters saved: {optimal_k}")
print(f"Total samples across all clusters: {len(data_with_clusters)}")
print(f"Each CSV file contains:")
print(f"  - {len(feature_names)} feature columns")
print(f"  - 1 'Cluster_ID' column (cluster membership)")
print(f"  - 1 'Bankrupt?' column (target label)")
print(f"  - Total: {len(feature_names) + 2} columns per file")


# %% [markdown]
# # Classification model to predict cluster label

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

print(f"{'='*80}")
print("LOGISTIC REGRESSION CLASSIFIER FOR CLUSTER PREDICTION")
print(f"{'='*80}\n")

# Prepare data
X = attributes.values  # Features
y = cluster_labels     # Target: cluster labels from K-means

print(f"Dataset shape: {X.shape}")
print(f"Number of clusters to predict: {len(np.unique(y))}")
print(f"Cluster distribution: {np.bincount(y)}\n")

# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=RANDOM_SEED, 
    stratify=y  # Ensure equal distribution across clusters
)

print(f"Training set size: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"Test set size: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)\n")

# Train Logistic Regression classifier
print("Training Logistic Regression classifier...")
log_reg = LogisticRegression(
    max_iter=1000,
    random_state=RANDOM_SEED,
    n_jobs=-1
)
log_reg.fit(X_train, y_train)
print("✓ Training completed\n")

# Predictions
y_train_pred = log_reg.predict(X_train)
y_test_pred = log_reg.predict(X_test)

# Accuracy scores
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"{'='*80}")
print("CLASSIFICATION ACCURACY")
print(f"{'='*80}\n")
print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"Test Accuracy:     {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Difference:        {(train_accuracy - test_accuracy):.4f} ({(train_accuracy - test_accuracy)*100:.2f}%)")

if train_accuracy - test_accuracy > 0.1:
    print("\n⚠ Note: Large gap between train and test accuracy may indicate overfitting")
else:
    print("\n✓ Good generalization: minimal gap between train and test accuracy")

# Detailed classification report
print(f"\n{'='*80}")
print("DETAILED CLASSIFICATION REPORT (Test Set)")
print(f"{'='*80}\n")
print(classification_report(y_test, y_test_pred, 
                          target_names=[f'Cluster {i}' for i in range(optimal_k)]))

# Confusion matrix
cm = confusion_matrix(y_test, y_test_pred)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cm, cmap='Blues', aspect='auto')

# Set ticks and labels
ax.set_xticks(np.arange(optimal_k))
ax.set_yticks(np.arange(optimal_k))
ax.set_xticklabels([f'C{i}' for i in range(optimal_k)])
ax.set_yticklabels([f'C{i}' for i in range(optimal_k)])

# Add text annotations
for i in range(optimal_k):
    for j in range(optimal_k):
        text = ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=11)

ax.set_xlabel('Predicted Cluster', fontsize=12)
ax.set_ylabel('True Cluster', fontsize=12)
ax.set_title('Confusion Matrix - Logistic Regression Cluster Prediction (Test Set)', 
             fontsize=13, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Count', rotation=270, labelpad=20)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Findings
# 
# - The logistic regression classifier achieves 97.68% accuracy on the test set
# - This means the cluster characteristics are well-captured by the features
# - The model can effectively distinguish between the 8 clusters
# - Model is suitable for predicting cluster membership for new companies

# %% [markdown]
# # Classifier function for Generalization

# %%
import sys
sys.path.append('..')

from modules.clustering import save_cluster_classifier

# Save the fitted logistic regression classifier
classifier_path = 'cluster_classifier.pkl'
save_cluster_classifier(log_reg, classifier_path)

print(f"\n✓ Logistic regression classifier saved successfully!")
print(f"  This model can now be used to predict cluster labels for new samples.")


