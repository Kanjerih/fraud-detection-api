# Real-Time Fraud Detection Engine

A production-ready machine learning inference API built to predict credit card fraud transaction probabilities in real time. This system bridges the gap between a standalone data science notebook and a live, containerized cloud service.

## Live Demo
The API is fully deployed and interactive:
* **Swagger UI Documentation:** [fraud-detection-api-5gq5.onrender.com/docs](https://fraud-detection-api-5gq5.onrender.com/docs)
*(Note: As this is hosted on a free cloud tier, please allow up to 1 minute for the server to wake up on the first load).*

## Model Performance
Trained on the highly imbalanced Kaggle Credit Card Fraud Detection dataset, the underlying model is optimized to balance fraud detection against false alarms:
* **Precision:** `96.2%` (Minimizes false positives, protecting user experience)
* **Recall:** `77.55%` (Catches the vast majority of fraudulent activities)
* **Decision Threshold:** Calibrated at `0.45` for optimal risk management.

## System Architecture & Features
* **FastAPI Backend:** Built high-performance async endpoints for streaming JSON transaction payloads.
* **Pydantic Data Validation:** Enforces data integrity on incoming traffic before it ever hits the model.
* **On-the-Fly Preprocessing:** Features a production pipeline that dynamically handles data scaling (`StandardScaler` for Time and Amount) and applies feature interaction maps matching the training environment layout exactly.
* **Dockerized Container:** Fully containerized to guarantee environment consistency between local development and production.

## Tech Stack
* **Language:** Python 3.10+
* **Framework:** FastAPI, Uvicorn, Pydantic
* **Machine Learning:** Scikit-Learn (v1.7.2), Pandas, NumPy, Joblib
* **DevOps:** Docker, Git, Render Cloud Platform

## Local Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Kanjerih/fraud-detection-api.git](https://github.com/Kanjerih/fraud-detection-api.git)
   cd fraud-detection-api
