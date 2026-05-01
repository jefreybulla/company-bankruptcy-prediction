#!/usr/bin/env python
# coding: utf-8

# ## How to load the pipeline
# 
# - Import the necessary custom class and libraries
# - Load the joblib file and the data
# - Run predict() on the data and pipeline

# ## Example

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

