import joblib
import os


def save_cluster_classifier(model, filepath):
    """
    Save a fitted cluster classification model using joblib.
    
    Parameters:
    -----------
    model : sklearn classifier model
        Fitted classification model (LogisticRegression, etc.) trained to predict cluster labels
    filepath : str
        Path where the model will be saved (.pkl file)
        
    Returns:
    --------
    str
        The filepath where the model was saved
    """
    # Create directory if it doesn't exist
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    
    # Save the model
    joblib.dump(model, filepath)
    print(f"✓ Cluster classifier saved to: {filepath}")
    
    return filepath


def load_cluster_classifier(filepath):
    """
    Load a previously saved cluster classification model using joblib.
    
    Parameters:
    -----------
    filepath : str
        Path to the saved model (.pkl file)
        
    Returns:
    --------
    sklearn classifier model
        The loaded classification model
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    model = joblib.load(filepath)
    print(f"✓ Cluster classifier loaded from: {filepath}")
    
    return model
