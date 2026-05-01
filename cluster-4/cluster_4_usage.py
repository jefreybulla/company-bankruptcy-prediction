#!/usr/bin/env python
# coding: utf-8

# ## Usage for joblib
# 
# `joblib` is used to save and load trained machine learning models efficiently, especially those built with scikit-learn pipelines. It allows you to persist a trained model to disk and reuse it later without retraining.
# 
# ### Saving a model
# ```python
# import joblib
# 
# joblib.dump(pipe, 'model.joblib')

# In[ ]:


import joblib
import pandas as pd
from preprocessing import CleanColumns
import numpy as np

pipe = joblib.load('cluster_4.joblib')
df = pd.read_csv('cluster_4.csv')

y_pred = pipe.predict(df)

T = np.sum(y_pred == 1)
F = np.sum(y_pred == 0)

print("T:", T)
print("F:", F)

