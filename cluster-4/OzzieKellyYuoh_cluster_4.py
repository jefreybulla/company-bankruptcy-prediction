#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

df = pd.read_csv('cluster_4.csv')
y = df['Bankrupt?']
X = df.drop(columns=['Bankrupt?'])


# In[ ]:


print(X.columns)


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

for col in df.columns[:-1]:
    plt.figure()
    sns.boxplot(x=df['Bankrupt?'], y=df[col])
    plt.title(f"{col} vs target")
    plt.show()


# The graphs give us a pretty good idea of how each feature is distributed across each of the differnt classes. We see that for some of the features the data is almost completely seperable with the negative class only having a single outlier preventing the data from being seperable. Since the cluster has such a small number of samples and there is really only 1 outlier in some of these graphs for the negative class a relatively accurate model in terms of recall can be made with just only a single feature such as Interest Coverage Ratio (Interest expense to EBIT). The amount of positive samples is also so low that increasing the number of features could lead to overfitting and since having a small numebr of features is important, might as well make it is small as possible.

# In[ ]:


X2 = df[["Interest Expense Ratio","Interest Coverage Ratio (Interest expense to EBIT)"]]
X2 = df[["Interest Coverage Ratio (Interest expense to EBIT)"]]


# ## Model Choice Summary
# 
# ### 1. Stacking Classifier
# - **Base learners:**
#   - SVM (polynomial kernel, degree=2)
#   - K-Nearest Neighbors (k=3)
#   - Random Forest (100 trees)
# - **Meta-model:** Logistic Regression (balanced class weights)
# 
# ### 2. SVM
# 
# - Captures nonlinear relationships using polynomial kernel
# - Class weight handles class imbalance
# 
# 
# ### 3. K-Nearest Neighbors
# 
# - Captures local structure in the feature space
# - Useful for identifying small clusters and minority patterns
# 
# ### 4. Random Forest
# 
# - Captures nonlinear interactions between features
# - Handles imbalance via class weighting
# 
# 
# ### 5. Logistic Regression
# 
# - Acts as a simple combiner of base model predictions
# 
# ### 6. SMOTE
# - Addresses severe class imbalance in training data
# - Generates synthetic minority samples
# 
# ### 7. StandardScaler
# 
# - Required for distance-based models (SVM, KNN)
# 
# ### 8. ColumnTransformer & CleanColumns
# - Removes whitespace and formatting inconsistencies in column names

# In[ ]:


import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

smote = SMOTE(sampling_strategy=0.8,k_neighbors=1, random_state=42)

base_models = [
    ("svm", SVC(kernel="poly", degree=2, class_weight="balanced", probability=True)),
    ("knn", KNeighborsClassifier(n_neighbors=3, weights="uniform")),
    ("rf", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42))
]

stack_model = StackingClassifier(estimators=base_models,final_estimator=LogisticRegression(class_weight="balanced"),cv=2)


class CleanColumns(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, z):
        z = z.copy()
        z.columns = z.columns.str.strip()
        return z

select = ColumnTransformer([("keep", "passthrough", ["Interest Coverage Ratio (Interest expense to EBIT)"])],remainder="drop")

pipe = Pipeline([
    ("clean_cols", CleanColumns()),
    ("select", select),
    ("smote", smote),
    ("scaler", StandardScaler()),
    ("model", stack_model)
])

cv = KFold(n_splits=2, shuffle=True, random_state=42)

scores = cross_val_score(pipe,X2,y,cv=cv,scoring="recall")

print("Recall scores:", scores)


pipe.fit(X2, y)

y_pred = pipe.predict(X)

print("\nPredictions:", y_pred)


# Recall scores for the cross validation get messed up due to the small number of positive samples in this cluster (only 2) and the splits resulting in either no positive examples at all or a very few number as such the recall scores of nan and 0 aren't really indicative of the model failing. 

# In[ ]:


import numpy as np



TP = np.sum((y_pred == 1) & (y == 1))

FN = np.sum((y_pred == 0) & (y == 1))

FP = np.sum((y_pred == 1) & (y == 0))
score = TP / (TP + FN) if (TP + FN) != 0 else 0

print("TP:", TP)
print("FN:", FN)
print("FP:", FP)
print("Score:", score)


# False Negatives and True Positives are the most important as they are directly used in calculating the score. False Negatives are also somewhat important due to their being a limit on the amount allowed, the number here is small enough to not really require worrying about. The model does correctly predict bankruptcy for the two positive cases in this cluster so using the score equation it would receive a 1.0 
# 

# In[ ]:


import joblib

joblib.dump(pipe, 'cluster_4.joblib')


# In[ ]:


pipe = joblib.load('cluster_4.joblib')

