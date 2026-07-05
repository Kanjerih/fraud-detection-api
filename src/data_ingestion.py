# !kaggle datasets download -d kanjerih/credit-card-fraud-dataset

import numpy as np
import pandas as pd

# +
import zipfile

with zipfile.ZipFile("credit-card-fraud-dataset.zip", 'r') as zip_ref:
    zip_ref.extractall()
# -

import os
os.listdir()

data = pd.read_csv('creditcard.csv')
data.head()



