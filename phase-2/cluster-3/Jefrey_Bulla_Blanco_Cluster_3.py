# %% [markdown]
# # Cluster 3

# %%
import pandas as pd
import numpy as np
import random
import os

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

# %%
df = pd.read_csv('../../Clusters/cluster_3.csv')

# %%
df.head()

# %%
print(df.shape)
df.describe()

# %% [markdown]
# Notice than some features are normalized between 0 and 1. Some features such as "Total assets to GNP price" are not normalized with some samples having larges values. All features seem to be numerical

# %%
df.info()

# %%
# Find number and percentage of companies with Bankrupt? = 1
bankrupt_count = (df["Bankrupt?"] == 1).sum()
total_count = len(df)
bankrupt_percentage = (bankrupt_count / total_count) * 100

print("Bankruptcy Distribution:")
print("="*60)
print(f"Companies with Bankrupt? = 1: {bankrupt_count}")
print(f"Companies with Bankrupt? = 0: {total_count - bankrupt_count}")
print(f"Total companies: {total_count}")
print(f"\nBankruptcy Rate: {bankrupt_percentage:.2f}%")
print(f"Non-bankruptcy Rate: {100 - bankrupt_percentage:.2f}%")

# %% [markdown]
# ## Boolean features

# %%
df["Net Income Flag"].describe()

# %%
# Check unique values for Net Income Flag
print("Unique values in Net Income Flag:")
print(df["Net Income Flag"].unique())
print(f"\nValue counts for Net Income Flag:")
print(df["Net Income Flag"].value_counts())
print(f"\nTotal samples: {len(df)}")
print(f"Samples with Net Income Flag = 1: {(df['Net Income Flag'] == 1).sum()}")
print(f"All values are 1: {(df['Net Income Flag'] == 1).all()}")

# %%
df["Liability-Assets Flag"].describe()

# %%
# Check unique values for Net Income Flag
print("Unique values in Liability-Assets Flag:")
print(df["Liability-Assets Flag"].unique())
print(f"\nValue counts for Liability-Assets Flag:")
print(df["Liability-Assets Flag"].value_counts())
print(f"\nTotal samples: {len(df)}")
print(f"Samples with Liability-Assets Flag = 1: {(df['Liability-Assets Flag'] == 1).sum()}")
print(f"All values are 0: {(df['Liability-Assets Flag'] == 0).all()}")

# %% [markdown]
# We have two categorical (boolean) features Net Income Flag and Liability-Assets Flag. Both features are heavily skewed.

# %% [markdown]
# ## Identify Constant features

# %%
# Identify constant features (features with only one unique value)
constant_features = []
for col in df.columns:
    if df[col].nunique() == 1:
        constant_features.append(col)

print(f"Constant features found: {len(constant_features)}\n")
if constant_features:
    for feature in constant_features:
        unique_val = df[feature].unique()[0]
        print(f"  - {feature}: {unique_val}")
else:
    print("No constant features found.")

# %% [markdown]
# ## Drop constant features and index

# %%
df = df.drop(columns=["Index", "Liability-Assets Flag", "Net Income Flag", "Cluster"])
df.shape

# %%
df.head()

# %% [markdown]
# ## Missing Values Analysis

# %%
# Check for missing values
print("Missing values summary:")
print("="*50)
missing_values = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100

# Create summary dataframe
missing_summary = pd.DataFrame({
    'Column': df.columns,
    'Missing Count': missing_values.values,
    'Missing %': missing_percent.values
})

# Filter to show only columns with missing values
missing_summary = missing_summary[missing_summary['Missing Count'] > 0].sort_values('Missing Count', ascending=False)

if len(missing_summary) == 0:
    print("\n✓ No missing values found in the dataset!")
else:
    print(f"\nFound {len(missing_summary)} columns with missing values:\n")
    print(missing_summary.to_string(index=False))

print(f"\n{'='*50}")
print(f"Total samples: {len(df)}")
print(f"Total features: {len(df.columns)}")

# %% [markdown]
# ## Feature Normalization Analysis

# %%
# Identify features NOT normalized between 0 and 1
non_normalized = []
normalization_summary = []

for col in df.columns:
    if col == "Bankrupt?":  # Skip target variable
        continue
    
    min_val = df[col].min()
    max_val = df[col].max()
    is_normalized = (min_val >= 0) and (max_val <= 1)
    
    normalization_summary.append({
        'Feature': col,
        'Min': min_val,
        'Max': max_val,
        'Normalized': 'Yes' if is_normalized else 'No'
    })
    
    if not is_normalized:
        non_normalized.append(col)

# Create summary dataframe
norm_df = pd.DataFrame(normalization_summary)
non_norm_df = norm_df[norm_df['Normalized'] == 'No'].sort_values('Min')

print(f"Features NOT normalized between 0 and 1: {len(non_normalized)}\n")
print(non_norm_df.to_string(index=False))
print(f"\n{'='*80}")
print(f"Features normalized between 0 and 1: {len(norm_df) - len(non_normalized)}")
print(f"Total features: {len(norm_df)}")

# %%
attributes_df = df.drop(columns=["Bankrupt?"])
target_df = df["Bankrupt?"]

print(attributes_df.shape)
print(target_df.shape)

# %%
from sklearn.preprocessing import MinMaxScaler

# Rebuild non_normalized list for attributes_df (with cleaned column names)
non_normalized_in_attrs = []
for col in attributes_df.columns:
    min_val = attributes_df[col].min()
    max_val = attributes_df[col].max()
    is_normalized = (min_val >= 0) and (max_val <= 1)
    
    if not is_normalized:
        non_normalized_in_attrs.append(col)

print(f"Features to scale: {len(non_normalized_in_attrs)}")

attributes_df_raw = attributes_df.copy()

# Initialize MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Apply scaling to non-normalized features in attributes_df
if len(non_normalized_in_attrs) > 0:
    attributes_df[non_normalized_in_attrs] = scaler.fit_transform(attributes_df[non_normalized_in_attrs])
    print(f"Scaled {len(non_normalized_in_attrs)} features using MinMaxScaler")
else:
    print("No features to scale")

print(f"\nAttributes dataframe shape: {attributes_df.shape}")

# %%
# Verify all features in attributes_df are now normalized between 0 and 1
non_normalized = []
normalization_summary = []

tolerance = 1e-10

for col in attributes_df.columns:    
    min_val = attributes_df[col].min()
    max_val = attributes_df[col].max()
    is_normalized = (min_val >= -tolerance) and (max_val <= 1 + tolerance)
    
    normalization_summary.append({
        'Feature': col,
        'Min': min_val,
        'Max': max_val,
        'Normalized': 'Yes' if is_normalized else 'No'
    })
    
    if not is_normalized:
        non_normalized.append(col)

# Create summary dataframe
norm_df = pd.DataFrame(normalization_summary)
non_norm_df = norm_df[norm_df['Normalized'] == 'No'].sort_values('Min')

print(f"Features NOT normalized between 0 and 1: {len(non_normalized)}\n")
if len(non_normalized) > 0:
    print(non_norm_df.to_string(index=False))
else:
    print("✓ All features are now normalized!")
print(f"\n{'='*80}")
print(f"Features normalized between 0 and 1: {len(norm_df) - len(non_normalized)}")
print(f"Total features: {len(norm_df)}")

# %% [markdown]
# ## Feature Variance Analysis

# %%

# Varriance filter was considered and tested but eventually discarded
'''
from sklearn.feature_selection import VarianceThreshold

# Keep features with variance above threshold
# Low variance = feature is nearly constant across all samples (not informative)
# High variance = feature changes significantly (more informative)
#selector = VarianceThreshold(threshold=0.000000)   # yields all 93 features, no filter
#selector = VarianceThreshold(threshold=0.0001)   # yields 52 features
selector = VarianceThreshold(threshold=0.001)   # yields 41 features
#selector = VarianceThreshold(threshold=0.01)    # yields 14 features
#selector = VarianceThreshold(threshold=0.1)    # yields 4 features
#selector = VarianceThreshold(threshold=0.12)    # yields ? features


selector.fit(attributes_df)
attributes_df_var = attributes_df.loc[:, selector.get_support()]

# Show dropped features
support_mask = selector.get_support()
dropped_features = attributes_df.columns[~support_mask].tolist()

print(f"Features remaining: {attributes_df_var.shape[1]}")
print(f"Features dropped: {len(dropped_features)}\n")
print("Dropped features (low variance):")
print("="*60)
for i, feature in enumerate(dropped_features, 1):
    variance = attributes_df[feature].var()
    print(f"{i}. {feature} (variance: {variance:.6f})")

'''


# %%
# Variance filter not used
#print(attributes_df_var.shape)
#attributes_df_var.head()

# %% [markdown]
# ## PCA dimentionality reduction

# %%

from sklearn.decomposition import PCA

n_components = 15

# Apply sklearn PCA
sklearn_pca = PCA(n_components=n_components, random_state=42)
#X_pca_sklearn = sklearn_pca.fit_transform(attributes_df_var)
X_pca_sklearn = sklearn_pca.fit_transform(attributes_df)

print(f"\nScikit-learn PCA Results:")
print(f"  Transformed data shape: {X_pca_sklearn.shape}")



# %%
attributes_final = X_pca_sklearn

# %% [markdown]
# ## Number of features used to train the model

# %%
attributes_final.shape

# %% [markdown]
# We are using **8 features** to train the model

# %% [markdown]
# ## Stacking model

# %%
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

X = attributes_final
y = target_df


print(f"Class distribution - Negative: {(y == 0).sum()}, Positive: {(y == 1).sum()}")

# Base learners optimized for imbalanced data (sklearn only)
base_learners = [
    ('gb', GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )),
    ('svc', SVC(
        kernel='rbf',
        C=10,
        class_weight='balanced',
        probability=True,
        random_state=42
    )),
    ('rf', RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    )),
    ('lr', LogisticRegression(
        C=1.0,
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    ))
]

# Meta-learner: Logistic Regression with balanced class weights
meta_learner = LogisticRegression(
    C=1.0,
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)

clf_cv = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=6
)
result_cv = clf_cv.fit(X, y).score(X, y)
#print(f"Stacking Model (with CV) - Training Accuracy: {round(result_cv, 3)}")

# %%
from sklearn.metrics import recall_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

y_pred_cv = clf_cv.predict(X)

print("="*80)
print("MINORITY CLASS (Bankrupt? = 1) RECALL EVALUATION")
print("="*80)

# Recall with cross-validation
recall_cv = recall_score(y, y_pred_cv, pos_label=1)
tn, fp, fn, tp = confusion_matrix(y, y_pred_cv).ravel()
recall_formula_cv = tp / (fn + tp)

print(f"\nModel WITH Cross-Validation:")
print(f"  Recall (sklearn): {recall_cv:.4f}")
print(f"  Recall Formula TP/(FN+TP): {recall_formula_cv:.4f}")
print(f"  Confusion Matrix: TP={tp}, FN={fn}, FP={fp}, TN={tn}")

# Create confusion matrix for cross-validation
cm_cv = confusion_matrix(y, y_pred_cv)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm_cv, annot=True, fmt='g', ax=ax, cmap='Blues', cbar_kws={'label': 'Count'})
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix - Stacking Model with Cross-Validation')
ax.xaxis.set_ticklabels(['Non-Bankrupt', 'Bankrupt'])
ax.yaxis.set_ticklabels(['Non-Bankrupt', 'Bankrupt'])

plt.tight_layout()
plt.show()

print(f"\n{'='*80}")
print("Detailed Classification Report (CV Model):")
print(f"{'='*80}")
print(classification_report(y, y_pred_cv, target_names=['Non-Bankrupt', 'Bankrupt']))

# %% [markdown]
# ## Summary of dimensionality experiments
# 
# | Dimentions | Method used for reduction | Recall of minority class |
# |--|--|--|
# | 93 |  None | 0.17 | 
# | 52 | Variance filter | 1 |
# | 41 | Variance filter | 1 |
# | 14 | Variance filter | 0 |
# | 4 | Variance filter | 0 |
# | 24 | Variance -> 41 & PCA -> 24 | 1 |
# | 12 | Variance -> 41 & PCA -> 12 | 1 |
# | 9 | Variance -> 41 & PCA -> 9 | 1 |
# | 8 | Variance -> 41 & PCA -> 8 | 1 |
# | 15 | PCA | 1 |
# | 8 | PCA | 1 |
# | 7 | PCA | 0.3 |
# | 7 | Variance -> 41 & PCA -> 7 | 0 |
# | 6 | Variance -> 41 & PCA -> 6 | 0 |

# %% [markdown]
# We obtained a recall of the minority class of **1 (100%)** using **15** features as the input of the stacking model

# %% [markdown]
# # Final results
# 
# | Subgroup ID | Name | Companies | Bankrupt | TT | TF | N_features |
# |---|---|---|---|---|---|---|
# | 3 | Jefrey | 1024 | 6 | 6 | 0 | 15 |

# %% [markdown]
# ## Export Preprocessing Pipeline and Stacking Model
# 
# Save the preprocessing pipeline and the fitted stacking model so they can be loaded and used from any other notebook to make predictions.

# %%
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# Define ColumnDropper transformer
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


# Define PreprocessorWrapper to fix the pipeline structure issue
class PreprocessorWrapper(BaseEstimator, TransformerMixin):
    """
    Wrapper that correctly applies the preprocessing pipeline.
    
    The original pipeline has a structural issue: the scaler was fitted on only 14 
    non-normalized columns, but the pipeline tries to feed it all 93 columns.
    
    This wrapper handles the proper column ordering and selective scaling.
    """
    def __init__(self, column_dropper, scaler, pca):
        self.column_dropper = column_dropper
        self.scaler = scaler
        self.pca = pca
        self.scaler_columns = list(scaler.feature_names_in_)
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Step 1: Drop columns
        X_dropped = self.column_dropper.transform(X)
        
        # Step 2: Apply scaler only to the columns it was fitted on (preserves column order)
        X_scaled = X_dropped.copy()
        X_scaled[self.scaler_columns] = self.scaler.transform(X_dropped[self.scaler_columns])
        
        # Step 3: Apply PCA
        return self.pca.transform(X_scaled)

# %%
# Create the fixed preprocessing pipeline using the wrapper
columns_to_drop = ["Index", "Liability-Assets Flag", "Net Income Flag", "Cluster"]

column_dropper = ColumnDropper(columns_to_drop=columns_to_drop)
preprocessor_wrapper = PreprocessorWrapper(
    column_dropper=column_dropper,
    scaler=scaler,
    pca=sklearn_pca
)

print(f"Preprocessor wrapper created:")
print(f"  Input: Raw cluster data with {len(df.columns)} columns")
print(f"  Output: {n_components} PCA components")
print(f"  Fixes: Properly handles selective column scaling while preserving order")

# %%
# Create a combined Predictor that includes both preprocessing and prediction
class Predictor(BaseEstimator):
    """
    Combined predictor that handles preprocessing and model prediction in one call.
    
    Usage:
        predictor = joblib.load('cluster_3_predictor.joblib')
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


# Create the combined predictor
predictor = Predictor(
    preprocessor=preprocessor_wrapper,
    model=clf_cv
)

print(f"Combined predictor created:")
print(f"  Preprocessor: {type(preprocessor_wrapper).__name__}")
print(f"  Model: {type(clf_cv).__name__}")
print(f"  Ready for single-file deployment")

# Save only the combined predictor
joblib.dump(predictor, 'cluster_3_predictor.joblib')
print("\n✓ Saved: cluster_3_predictor.joblib")

# Verify it can be loaded and used
loaded_predictor = joblib.load('cluster_3_predictor.joblib')
print(f"✓ Reload test passed: {loaded_predictor}")

# Test with a sample
test_predictions = loaded_predictor.predict(df.drop(columns=['Bankrupt?']).head(5))
test_probas = loaded_predictor.predict_proba(df.drop(columns=['Bankrupt?']).head(5))
print(f"✓ Functionality test passed")
print(f"  Sample predictions: {test_predictions}")
print(f"  Sample probabilities shape: {test_probas.shape}")

# %%
# (Old approach - no longer needed, using single predictor file instead)
# joblib.dump(preprocessing_pipeline, 'cluster_3_preprocessing_pipeline.joblib')
# joblib.dump(clf_cv, 'cluster_3_stacking_model.joblib')
print("Note: Using single 'cluster_3_predictor.joblib' file for both preprocessing and prediction")
print("      No separate pipeline and model files needed.")

# %%
# Example: How to use the saved predictor from any other notebook
print("\n" + "="*70)
print("USAGE EXAMPLE: Making predictions on new data")
print("="*70)
print("""
# Load the combined predictor (preprocessing + model in one file)
import joblib
import pandas as pd

predictor = joblib.load('cluster_3_predictor.joblib')

# Load your new company data as a DataFrame
# (must have the same features as the original cluster_3.csv)
new_data = pd.read_csv('new_companies.csv')

# Make predictions (preprocessing is handled automatically)
predictions = predictor.predict(new_data)
probabilities = predictor.predict_proba(new_data)

# Display results
print("Predictions (0=Not Bankrupt, 1=Bankrupt):", predictions)
print("Probabilities:", probabilities)

# That's it! No need to handle preprocessing separately.
""")



