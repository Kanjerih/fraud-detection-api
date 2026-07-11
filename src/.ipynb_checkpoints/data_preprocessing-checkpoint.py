# +
import pandas as pd
from sklearn.preprocessing import RobustScaler

def clean_and_scale_data(X_train_raw, X_test_raw):
    # Initialize scalers
    scaler_amt = RobustScaler()
    scaler_time = RobustScaler()
    
    # We must copy dataframes to avoid mutating original views
    X_train_scaled = X_train_raw.copy()
    X_test_scaled = X_test_raw.copy()
    
    # Fit ONLY on train, transform both to eliminate data leakage
    X_train_scaled['Amount'] = scaler_amt.fit_transform(X_train_scaled[['Amount']])
    X_test_scaled['Amount'] = scaler_amt.transform(X_test_scaled[['Amount']])
    
    X_train_scaled['Time'] = scaler_time.fit_transform(X_train_scaled[['Time']])
    X_test_scaled['Time'] = scaler_time.transform(X_test_scaled[['Time']])
    
    # Feature engineering: add interaction shortcuts
    X_train_scaled['V11_V4_interaction'] = X_train_scaled['V11'] * X_train_scaled['V4']
    X_test_scaled['V11_V4_interaction'] = X_test_scaled['V11'] * X_test_scaled['V4']
    
    return X_train_scaled, X_test_scaled, scaler_amt, scaler_time