# %% [markdown]
# # EDA

# %%
import pandas as pd

# %%
df = pd.read_csv('../data/train_data.csv')

# Remove leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# %%
df.shape

# %%
df.head()

# %%
df.describe()

# %% [markdown]
# Notice than some features are normalized between 0 and 1. Some features such as "Total assets to GNP price" are not normalized and some samples have larges values. All features seem to be numerical

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
# Most feature are stored as float. A few features are stores as int and may represent categorical features. Let's investigate. 

# %%
# See exact names of all features
df.columns.tolist()

# %% [markdown]
# ## Boolean features

# %%
df["Net Income Flag"].describe()

# %%
df["Liability-Assets Flag"].describe()

# %% [markdown]
# We have two categorical (boolean) features Net Income Flag and Liability-Assets Flag. Both features are heavily skewed. 

# %%
# Check unique values for Net Income Flag
print("Unique values in Net Income Flag:")
print(df["Net Income Flag"].unique())
print(f"\nValue counts for Net Income Flag:")
print(df["Net Income Flag"].value_counts())
print(f"\nTotal samples: {len(df)}")
print(f"Samples with Net Income Flag = 1: {(df['Net Income Flag'] == 1).sum()}")
print(f"All values are 1: {(df['Net Income Flag'] == 1).all()}")

# %% [markdown]
# Since all sample have `Net Income Flag = 1` this feature is not informative for classificaiotn and therefore can be dropped. 

# %%
#df = df.drop(columns=[" Net Income Flag"])
#df.shape

# %%
# Check correlation of Liability-Assets Flag and Bankrupcy
target_col = "Bankrupt?"

# Relationship to bankruptcy
print("Bankruptcy rate by Liability-Assets Flag:")
liability_flag_bankruptcy = df.groupby("Liability-Assets Flag")[target_col].agg(['sum', 'count', 'mean'])
liability_flag_bankruptcy.columns = ['Bankruptcies', 'Total Companies', 'Bankruptcy Rate']
print(liability_flag_bankruptcy)
print(f"\nBankruptcy rate when flag = 0: {liability_flag_bankruptcy.loc[0, 'Bankruptcy Rate']*100:.2f}%")
print(f"Bankruptcy rate when flag = 1: {liability_flag_bankruptcy.loc[1, 'Bankruptcy Rate']*100:.2f}%")

# %% [markdown]
# We observed that:
# - Flag = 0: 3.33% bankruptcy rate (193 bankruptcies out of 5,800 companies)
# - Flag = 1: 71.43% bankruptcy rate (5 bankruptcies out of 7 companies)
# 
# Thus `Liability-Assets Flag` is a higly predictive feature

# %% [markdown]
# ## Identify Constant features
# 

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

# %% [markdown]
# # Create attributes Data frame

# %%
attributes_df = df.drop(columns=["Index", "Bankrupt?", "Net Income Flag"])

attributes_df.shape

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

# IMPORTANT: Save a copy of raw attributes_df BEFORE scaling (needed for fitting final scaler later)
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

for col in attributes_df.columns:    
    min_val = attributes_df[col].min()
    max_val = attributes_df[col].max()
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
if len(non_normalized) > 0:
    print(non_norm_df.to_string(index=False))
else:
    print("✓ All features are now normalized!")
print(f"\n{'='*80}")
print(f"Features normalized between 0 and 1: {len(norm_df) - len(non_normalized)}")
print(f"Total features: {len(norm_df)}")

# %% [markdown]
# # Feature Variance Analysis

# %%
from sklearn.feature_selection import VarianceThreshold

# Keep features with variance above threshold
# Low variance = feature is nearly constant across all samples (not informative)
# High variance = feature changes significantly (more informative)
selector = VarianceThreshold(threshold=0.001)
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

# %%
attributes_df_var.describe()

# %%
attributes_df_var.info()

# %%
# plot histograms
import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=10, ncols=5, figsize=(20, 24))
axes = axes.flatten()

for idx, col in enumerate(attributes_df_var.columns):
    axes[idx].hist(attributes_df_var[col], bins=50)
    axes[idx].set_title(col[:20], fontsize=10)
    axes[idx].set_xlabel('Value')
    axes[idx].set_ylabel('Frequency')

# Hide unused subplots
for idx in range(len(attributes_df_var.columns), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.show()

# %%
# Create boxplots for all features
fig, axes = plt.subplots(nrows=10, ncols=5, figsize=(20, 24))
axes = axes.flatten()

for idx, col in enumerate(attributes_df_var.columns):
    axes[idx].boxplot(attributes_df_var[col])
    axes[idx].set_title(col[:20], fontsize=14)
    axes[idx].set_ylabel('Value')

# Hide unused subplots
for idx in range(len(attributes_df_var.columns), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.show()

# %%
# Save processed training set to CSV

# Add target column to attributes_df_var
#final_df = attributes_df_var.copy()
# Add target column to final dataframe.
#final_df['Bankrupt?'] = df['Bankrupt?'].values

# Save to CSV
#final_df.to_csv('train_data_processed.csv', index=False)

#print(f"Final dataset shape: {final_df.shape}")
#print(f"Saved to train_data_processed.csv")
#print(f"\nFirst few rows:")
#print(final_df.head())

# %% [markdown]
# # Function for generalization

# %%
# Check which non_normalized features survived variance filtering
print("Non-normalized features that survived variance filtering:")
print("="*70)
survived = [col for col in non_normalized_in_attrs if col in attributes_df_var.columns]
dropped_from_non_norm = [col for col in non_normalized_in_attrs if col not in attributes_df_var.columns]

print(f"\nSurvived ({len(survived)}):")
for col in survived:
    print(f"  ✓ {col}")

print(f"\nDropped due to low variance ({len(dropped_from_non_norm)}):")
for col in dropped_from_non_norm:
    print(f"  ✗ {col}")

print(f"\n{'='*70}")
print(f"Total non-normalized features: {len(non_normalized_in_attrs)}")
print(f"Survived: {len(survived)} ({len(survived)/len(non_normalized_in_attrs)*100:.1f}%)")
print(f"Dropped: {len(dropped_from_non_norm)} ({len(dropped_from_non_norm)/len(non_normalized_in_attrs)*100:.1f}%)")

# %%
import joblib

# Create a NEW scaler fitted ONLY on the survived non-normalized features
# (not all non-normalized features - only those that made it through variance filtering)
# Important: Fit on RAW data (attributes_df_raw) NOT already-scaled data (attributes_df or attributes_df_var)
scaler_final = MinMaxScaler(feature_range=(0, 1))
scaler_final.fit(attributes_df_raw[survived])

# Save preprocessing artifacts for later use on test data
preprocessing_artifacts = {
    'scaler': scaler_final,
    'final_features': list(attributes_df_var.columns),
    'survived_non_normalized': survived,
}

joblib.dump(preprocessing_artifacts, 'preprocessing_artifacts.pkl')
print("Preprocessing artifacts saved to preprocessing_artifacts.pkl")
print(f"Saved features: {len(preprocessing_artifacts['final_features'])}")
print(f"Saved non-normalized features to scale: {len(preprocessing_artifacts['survived_non_normalized'])}")
print(f"\nScaler features: {survived}")

# Define function to process test data
def process_data(test_df, artifacts_path='preprocessing_artifacts.pkl'):
    """
    Process test data with the same transformations as training data.
    
    Parameters:
    -----------
    test_df : pd.DataFrame
        Raw test dataframe to process
    artifacts_path : str
        Path to the saved preprocessing artifacts
        
    Returns:
    --------
    pd.DataFrame
        Processed test data with same features and scaling as training data.
        If 'Bankrupt?' column exists in input, it will be preserved at the end.
    """
    # Load preprocessing artifacts
    artifacts = joblib.load(artifacts_path)
    scaler = artifacts['scaler']
    final_features = artifacts['final_features']
    survived_non_normalized = artifacts['survived_non_normalized']
    
    # Create a copy to avoid modifying original
    test_processed = test_df.copy()
    
    # Step 0: Preserve target column if it exists
    target_column = None
    if 'Bankrupt?' in test_processed.columns:
        target_column = test_processed[['Bankrupt?']].copy()
        test_processed = test_processed.drop(columns=['Bankrupt?'])
        print("Preserved 'Bankrupt?' column")
    
    # Step 1: Drop columns that aren't in final features
    cols_to_drop = [col for col in test_processed.columns if col not in final_features]
    if cols_to_drop:
        test_processed = test_processed.drop(columns=cols_to_drop)
        print(f"Dropped {len(cols_to_drop)} columns not in training features")
    
    # Step 2: Add any missing features as NaN (in case test data is missing some columns)
    missing_features = [col for col in final_features if col not in test_processed.columns]
    if missing_features:
        print(f"Warning: {len(missing_features)} features missing from test data: {missing_features}")
        for col in missing_features:
            test_processed[col] = None
    
    # Step 3: Scale the non-normalized features that survived variance filtering
    if survived_non_normalized:
        # Only scale features that exist in the test data
        features_to_scale = [col for col in survived_non_normalized if col in test_processed.columns]
        
        if features_to_scale:
            test_processed[features_to_scale] = scaler.transform(test_processed[features_to_scale])
            print(f"Scaled {len(features_to_scale)} non-normalized features")
    
    # Step 4: Ensure column order matches training data
    test_processed = test_processed[final_features]
    
    # Step 5: Add back the target column if it existed
    if target_column is not None:
        test_processed['Bankrupt?'] = target_column.values
    
    return test_processed

print("\nFunction 'process_data()' created successfully!")
print("Usage: test_processed = process_data(raw_test_df)")

# %% [markdown]
# # Test artifact to process data

# %%
# Load and process test data
test_df_raw = pd.read_csv('../data/test_data.csv')

# Remove leading/trailing spaces from column names
test_df_raw.columns = test_df_raw.columns.str.strip()

print("Raw test data loaded:")
print(f"Shape: {test_df_raw.shape}")
print(f"Columns: {len(test_df_raw.columns)}")

# Process test data using our function
test_df_processed = process_data(test_df_raw)

print(f"\n{'='*70}")
print("Test data after processing:")
print(f"Shape: {test_df_processed.shape}")
print(f"Columns: {len(test_df_processed.columns)}")

print(f"\n{'='*70}")
print("Comparison with training data:")
print(f"Training shape: {attributes_df_var.shape}")
print(f"Test shape: {test_df_processed.shape}")
print(f"Same number of features: {attributes_df_var.shape[1] == test_df_processed.shape[1]}")
print(f"Test samples: {test_df_processed.shape[0]}")

# %%
test_df_processed.describe()

# %%
# Test: Verify function preserves 'Bankrupt?' column when present
print("Testing process_test_data with 'Bankrupt?' column in input:")
print("="*70)

# Create test data WITH target column by combining features and target
test_with_target = test_df_processed.copy()
test_with_target['Bankrupt?'] = 0  # Add synthetic target

print(f"Input shape (with Bankrupt?): {test_with_target.shape}")
print(f"Input columns include 'Bankrupt?': {'Bankrupt?' in test_with_target.columns}")

# Process with target column - reloads from raw
test_raw_with_target = test_df_raw.copy()
test_raw_with_target['Bankrupt?'] = 0  # Add synthetic target
test_raw_with_target.columns = test_raw_with_target.columns.str.strip()

print(f"\nProcessing raw test data WITH target column...")
result = process_data(test_raw_with_target)

print(f"\nOutput shape: {result.shape}")
print(f"Output columns (last 5): {list(result.columns[-5:])}")
print(f"\n✓ 'Bankrupt?' preserved: {'Bankrupt?' in result.columns}")
print(f"✓ Position: Column {list(result.columns).index('Bankrupt?') + 1} of {len(result.columns)}")
print(f"✓ Total columns: {result.shape[1]} (42 features + 1 target)")

print(f"\nFirst 3 rows with target column:")
print(result[['Operating Expense Rate', 'Bankrupt?']].head(3))


