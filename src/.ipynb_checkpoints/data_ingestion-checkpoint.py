# !kaggle datasets download -d kanjerih/credit-card-fraud-dataset

# +
import os
import zipfile
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_raw_data(csv_path="creditcard.csv"):
    # If the CSV path is just the filename, let's also check if it lives inside the 'src/' folder
    if not os.path.exists(csv_path) and os.path.exists(os.path.join("src", csv_path)):
        csv_path = os.path.join("src", csv_path)
        
    print(f"--> Reading dataset from: {csv_path}...")
    data = pd.read_csv(csv_path)
    
    # Clean duplicates immediately
    data = data.drop_duplicates()
    
    X = data.drop(['Class'], axis=1)
    y = data["Class"]
    
    # Stratified split to keep the 0.17% fraud ratio consistent
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    return X_train_raw, X_test_raw, y_train, y_test
