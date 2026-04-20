import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler


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
