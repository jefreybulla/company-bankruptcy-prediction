#!/usr/bin/env python
# coding: utf-8

# # Aania — Cluster 5 stacking model
# 
# C5 has 916 companies, 72 of which are bankrupt (about 8% rate). This is the cluster I'm being graded on, so my goal here is high recall on bankrupts (TT/(TT+TF) per spec) without using too many features.

# ## 1. Setup

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# The spec uses TT/(TT+TF) as the accuracy metric — this is just recall on the bankrupt class. Sklearn's default accuracy score is misleading on imbalanced data so I'm rolling my own helper here.

# In[2]:


def eq1_accuracy(y_true, y_pred):
    TT = ((y_true == 1) & (y_pred == 1)).sum()
    TF = ((y_true == 1) & (y_pred == 0)).sum()
    return TT / (TT + TF) if (TT + TF) > 0 else 0.0

def show_confusion(y_true, y_pred, name):
    FF = ((y_true == 0) & (y_pred == 0)).sum()
    FT = ((y_true == 0) & (y_pred == 1)).sum()
    TT = ((y_true == 1) & (y_pred == 1)).sum()
    TF = ((y_true == 1) & (y_pred == 0)).sum()
    print(f'{name}:')
    print(f'  FF: {FF:>4}    FT: {FT:>4}')
    print(f'  TT: {TT:>4}    TF: {TF:>4}')
    print(f'  Eq.1 acc: {eq1_accuracy(y_true, y_pred):.4f}')
    return TT, TF


# ## 2. Load the cluster data

# In[3]:


df = pd.read_csv('../cluster_5.csv')
print(f'Shape: {df.shape}')
print(f'Bankrupts: {df["Bankrupt?"].sum()} / {len(df)} ({df["Bankrupt?"].mean()*100:.2f}%)')


# Drop columns that can't be features:
# - `Index` is a row identifier
# - `Cluster` is the cluster label (would leak the answer)
# - Any column with only one unique value gives us no information

# In[4]:


y = df['Bankrupt?'].values
X = df.drop(columns=['Bankrupt?', 'Index', 'Cluster'])

const_cols = [c for c in X.columns if X[c].nunique() <= 1]
print(f'Constant columns dropped: {const_cols}')
X = X.drop(columns=const_cols)
print(f'Features: {X.shape}')


# ## 3. Feature selection with L1 logistic regression
# 
# L1 regularization shrinks coefficients of weak features to zero, which gives us automatic feature selection. Lower C = more aggressive shrinkage = fewer features kept.
# 
# I'm sweeping C to see how many features survive at each level. The goal is fewest features without losing too much CV F1.

# In[5]:


X_scaled = StandardScaler().fit_transform(X)
cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)

print(f'{"C":>8} {"# features":>11} {"CV F1":>8}')
print('-' * 30)
for C in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
    lr = LogisticRegression(penalty='l1', solver='saga', C=C,
                            class_weight='balanced', max_iter=5000,
                            random_state=RANDOM_STATE)
    lr.fit(X_scaled, y)
    n_feat = (lr.coef_[0] != 0).sum()
    f1 = cross_val_score(lr, X_scaled, y, cv=cv, scoring='f1').mean()
    print(f'{C:>8.3f} {n_feat:>11d} {f1:>8.4f}')


# F1 doesn't improve much past 5 features (around 0.36 at C=0.01 vs 0.40 at higher C with way more features). Going with C=0.01 — picking the smallest viable set since N_features is also part of the grade.

# In[6]:


C_chosen = 0.01
selector = LogisticRegression(penalty='l1', solver='saga', C=C_chosen,
                              class_weight='balanced', max_iter=5000,
                              random_state=RANDOM_STATE)
selector.fit(X_scaled, y)

selected_features = X.columns[selector.coef_[0] != 0].tolist()
N_FEATURES = len(selected_features)
X_selected = X[selected_features]

print(f'Selected {N_FEATURES} features:\n')
for i, f in enumerate(selected_features, 1):
    coef = selector.coef_[0][list(X.columns).index(f)]
    print(f'  {i}. {f.strip():<55} coef={coef:+.3f}')


# These are interpretable financial signals — debt ratio and cash position go up for distressed firms, ROA and net worth go down. Makes sense.

# ## 4. Three base models
# 
# The spec needs 3+ base models for the stacking layer, all using the same features. I'm picking three with different inductive biases so the meta-learner has something useful to combine:
# 
# - **Logistic Regression** (with a scaler) — It’s good at picking up straightforward, additive relationships between financial ratios. I’m using class_weight='balanced' to deal with the class imbalance.
# - **Random Forest** — this one captures non-linear patterns and interactions. For example, high debt might only be risky when liquidity is low. It also uses class_weight='balanced'.
# - **Gradient Boosting** — builds trees sequentially and usually performs really well on tabular data by correcting previous mistakes.
# 
# Not using SMOTE here because we have 72 positive examples — enough to learn from without synthetic data, and `class_weight='balanced'` handles the imbalance with less risk of overfitting.

# In[7]:


base_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(class_weight='balanced', max_iter=2000,
                               random_state=RANDOM_STATE))
])

base_rf = RandomForestClassifier(n_estimators=200, max_depth=8,
                                  class_weight='balanced',
                                  random_state=RANDOM_STATE)

base_gb = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                     learning_rate=0.05,
                                     random_state=RANDOM_STATE)

base_models = [('lr', base_lr), ('rf', base_rf), ('gb', base_gb)]


# Quick CV sanity check on each base model individually before stacking them.

# In[8]:


for name, model in base_models:
    f1 = cross_val_score(model, X_selected, y, cv=cv, scoring='f1').mean()
    print(f'{name}: CV F1 = {f1:.4f}')


# ## 5. Stacking with cross-validation

# In[9]:


stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(class_weight='balanced', max_iter=2000,
                                       random_state=RANDOM_STATE),
    cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
    stack_method='predict_proba',
)

stacking.fit(X_selected, y)
print('stacking model fitted')


# ## 6. Threshold tuning
# 
# Default threshold is 0.5 (predict bankrupt if probability ≥ 0.5). With imbalanced data, that threshold is often too conservative — we miss bankrupts the model was somewhat sure about.
# 
# I'm sweeping thresholds from 0.20 to 0.65 and picking the one with the highest Eq.1 accuracy. Constraint: predicted-bankrupt rate has to stay under 25%, otherwise C5's contribution to the team submission could push us past the 20% sparsity budget at test time.
# 
# Using out-of-fold probabilities so the threshold isn't tuned on data the model was trained on.

# In[10]:


oof_proba = cross_val_predict(stacking, X_selected, y, cv=cv,
                              method='predict_proba')[:, 1]

print(f'{"thresh":>8} {"pred%":>8} {"TT":>4} {"TF":>4} {"Eq.1":>8}')
print('-' * 40)

best_thresh = 0.5
best_acc = 0.0
for t in np.arange(0.20, 0.65, 0.05):
    preds = (oof_proba >= t).astype(int)
    pred_rate = preds.mean()
    TT = ((y == 1) & (preds == 1)).sum()
    TF = ((y == 1) & (preds == 0)).sum()
    acc = eq1_accuracy(y, preds)
    flag = ' *' if (acc > best_acc and pred_rate < 0.25) else ''
    print(f'{t:>8.2f} {pred_rate*100:>7.2f}% {TT:>4} {TF:>4} {acc:>8.4f}{flag}')
    if acc > best_acc and pred_rate < 0.25:
        best_acc = acc
        best_thresh = t

print(f'\nPicked threshold: {best_thresh:.2f}  (CV Eq.1 acc {best_acc:.4f})')


# ## 7. Save model bundle
# 
# The spec asks for the preprocessing + fitted stacking model combined into one object saved with joblib. Using a dict — keeps things simple, no custom classes needed.

# In[11]:


model_bundle = {
    'selected_features': selected_features,
    'stacking_model':    stacking,
    'threshold':         best_thresh,
    'n_features':        N_FEATURES,
}

def predict_with_bundle(bundle, df_input):
    X_sel = df_input[bundle['selected_features']]
    proba = bundle['stacking_model'].predict_proba(X_sel)[:, 1]
    return (proba >= bundle['threshold']).astype(int)

joblib.dump(model_bundle, 'aania_cluster5_model.joblib')
print('saved: aania_cluster5_model.joblib')

loaded = joblib.load('aania_cluster5_model.joblib')
X_full = df.drop(columns=['Bankrupt?', 'Index', 'Cluster'] + const_cols)
status = 'PASSED' if (predict_with_bundle(loaded, X_full) == predict_with_bundle(model_bundle, X_full)).all() else 'FAILED'
print(f'reload check: {status}')


# ## 8. Table 3 numbers
# 
# Per spec page 7, TT and TF must be reported on the **original** training data for the cluster, not on a held-out split.

# In[12]:


y_pred_train = predict_with_bundle(model_bundle, X_full)

print('=' * 60)
print('TABLE 3 NUMBERS — Cluster 5')
print('=' * 60)
TT, TF = show_confusion(y, y_pred_train, 'Stacking model')
print()
print(f'  Companies:    {len(y)}')
print(f'  Bankrupt:     {y.sum()}')
print(f'  N_features:   {N_FEATURES}')
print(f'  Train Eq.1:   {eq1_accuracy(y, y_pred_train):.4f}')
print('=' * 60)


# ## 9. Confusion matrices
# 
# The spec asks for confusion matrices for each base model and the meta-model.

# In[13]:


fitted_bases = stacking.named_estimators_

for name in ['lr', 'rf', 'gb']:
    show_confusion(y, fitted_bases[name].predict(X_selected), f'Base [{name.upper()}]')
    print()

show_confusion(y, stacking.predict(X_selected), 'Meta (default threshold 0.5)')
print()
show_confusion(y, y_pred_train, f'Meta (tuned threshold {best_thresh:.2f})  <-- SUBMITTING')


# In[14]:


fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
plots = [
    ('LR', fitted_bases['lr'].predict(X_selected)),
    ('RF', fitted_bases['rf'].predict(X_selected)),
    ('GB', fitted_bases['gb'].predict(X_selected)),
    (f'Meta (t={best_thresh:.2f})', y_pred_train),
]
for ax, (name, pred) in zip(axes, plots):
    cm = confusion_matrix(y, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=['Healthy', 'Bankrupt'], yticklabels=['Healthy', 'Bankrupt'])
    ax.set_title(f'{name}\nEq.1 acc = {eq1_accuracy(y, pred):.3f}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.tight_layout()
plt.show()


# ## For the team report (Table 3 row)
# 
# | Subgroup ID | Name | Companies | Bankrupt | TT | TF | N_features |
# |---|---|---|---|---|---|---|
# | 5 | Aania | 916 | 72 | 66 | 6 | 5 |
