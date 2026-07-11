# +
import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Real-Time Fraud Detection Engine")

# Locate the models directory relative to this file's location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Load the preprocessing rules and the trained brain
MODEL = joblib.load(os.path.join(MODELS_DIR, "rf_fraud_model.pkl"))
SCALER_AMOUNT = joblib.load(os.path.join(MODELS_DIR, "scaler_amount.pkl"))
SCALER_TIME = joblib.load(os.path.join(MODELS_DIR, "scaler_time.pkl"))
METADATA = joblib.load(os.path.join(MODELS_DIR, "model_metadata.pkl"))
THRESHOLD = METADATA["threshold"]

# DEBUG: Print out metadata keys to your terminal window on startup
print("\n--- METADATA DIAGNOSTICS ---")
print("Available keys in metadata:", list(METADATA.keys()))
if hasattr(MODEL, "feature_names_in_"):
    print("Model native features:", list(MODEL.feature_names_in_))
print("----------------------------\n")

# Hardcoded fallback list matching the exact structure used during your training
# (Assuming the interaction term was appended at the end after the V features)
FALLBACK_FEATURES = (
    ['Time', 'Amount'] + 
    [f'V{i}' for i in range(1, 29)] + 
    ['V11_V4_interaction']
)

# Extract features dynamically or drop back to fallback layout
if "feature_names" in METADATA:
    EXPECTED_FEATURES = list(METADATA["feature_names"])
elif "features" in METADATA:
    EXPECTED_FEATURES = list(METADATA["features"])
elif hasattr(MODEL, "feature_names_in_"):
    EXPECTED_FEATURES = list(MODEL.feature_names_in_)
else:
    EXPECTED_FEATURES = FALLBACK_FEATURES

# Define the expected JSON payload format matching the Kaggle layout
class Transaction(BaseModel):
    Time: float
    Amount: float
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float

@app.post("/predict")
def predict_live_transaction(tx: Transaction):
    try:
        # Pydantic safety translation
        tx_dict = tx.model_dump() if hasattr(tx, "model_dump") else tx.dict()
        
        # Convert payload to DataFrame row
        input_df = pd.DataFrame([tx_dict])
        
        # Scale numerical elements
        input_df['Amount'] = SCALER_AMOUNT.transform(input_df[['Amount']])
        input_df['Time'] = SCALER_TIME.transform(input_df[['Time']])
        input_df['V11_V4_interaction'] = input_df['V11'] * input_df['V4']
        
        # Force strict column reordering
        input_df = input_df[EXPECTED_FEATURES]
        
        # Compute probabilities
        probability = MODEL.predict_proba(input_df)[0][1]
        is_fraud = int(probability >= THRESHOLD)
        
        return {
            "is_fraud": is_fraud,
            "fraud_probability": float(probability),
            "applied_threshold": THRESHOLD
        }
        
    except Exception as e:
        
        return {
            "status": "failed",
            "error_message": str(e),
            "sent_columns": list(input_df.columns) if 'input_df' in locals() else [],
            "expected_columns": EXPECTED_FEATURES
        }
