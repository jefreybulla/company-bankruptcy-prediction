"""
Helper functions for using Aania's cluster predictors.

These are required to use the saved joblib bundles:
- aania_cluster5_model.joblib
- aania_cluster1_model.joblib

My models are saved as plain dicts. The dict contains:
- 'selected_features': list of feature column names used by the model
- 'stacking_model':    the fitted sklearn StackingClassifier
- 'threshold':         the decision threshold for classifying as bankrupt
- 'n_features':        number of features used (for Table 3 reporting)

Usage:
    import joblib
    from aania_cluster_classes import predict_with_bundle

    bundle = joblib.load('aania_cluster5_model.joblib')
    predictions = predict_with_bundle(bundle, new_data_df)
"""


def predict_with_bundle(bundle, df_input):
    X_sel = df_input[bundle['selected_features']]
    proba = bundle['stacking_model'].predict_proba(X_sel)[:, 1]
    return (proba >= bundle['threshold']).astype(int)


def predict_proba_with_bundle(bundle, df_input):
    X_sel = df_input[bundle['selected_features']]
    return bundle['stacking_model'].predict_proba(X_sel)
