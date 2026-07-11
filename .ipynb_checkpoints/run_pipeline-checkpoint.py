# +
# run_pipeline.py
import os
import sys
import joblib

# 1. Dynamically point Python to look inside src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from data_ingestion import prepare_raw_data
from data_preprocessing import clean_and_scale_data
from model_trainer import train_fraud_model
from threshold_finder import optimize_threshold

def run_training_pipeline():
    print("=== Starting Unified Fraud Detection Pipeline ===")
    
    # Step 1: Ingest
    X_train_raw, X_test_raw, y_train, y_test = prepare_raw_data("creditcard.csv")
    
    # Step 2: Preprocess
    X_train_scaled, X_test_scaled, scaler_amt, scaler_time = clean_and_scale_data(X_train_raw, X_test_raw)
    
    # Step 3: Train
    model, imbalance_ratio = train_fraud_model(X_train_scaled, y_train)
    
    # Step 4: Optimize
    best_threshold = optimize_threshold(model, X_test_scaled, y_test)
    
    # Step 5: Serialize Artifacts for API Delivery
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/rf_fraud_model.pkl")  # Updated name
    joblib.dump(scaler_amt, "models/scaler_amount.pkl")
    joblib.dump(scaler_time, "models/scaler_time.pkl")
    joblib.dump({"threshold": best_threshold}, "models/model_metadata.pkl")
    print("\n=== Pipeline Execution Complete! All Binaries Synchronized ===")

if __name__ == "__main__":
    run_training_pipeline()
