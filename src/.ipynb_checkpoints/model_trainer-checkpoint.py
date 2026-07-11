# +
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def train_fraud_model(X_train_scaled, y_train):
    print("--> Training RandomForestClassifier...")
    
    # Instantiate and fit the model
    rfc = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rfc.fit(X_train_scaled, y_train)
    
    # Calculate balance ratio in case you choose an imbalanced workflow later
    imbalance_ratio = (len(y_train) - sum(y_train)) / sum(y_train)
    
    return rfc, imbalance_ratio

