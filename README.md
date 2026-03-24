# MLflow Lab 1 - Experiment Tracking

## Overview
This lab demonstrates MLflow experiment tracking using:
- **Dataset**: Diabetes dataset (sklearn built-in)
- **Model**: Ridge Regression
- **Modification from original**: Changed from Wine Quality dataset + ElasticNet to Diabetes dataset + Ridge Regression

## Project Structure
```
mlops-mlflow-lab1-kishlaya/
├── linear_regression.py   # Train model + log to MLflow
├── serving.py             # Load saved model + make predictions
├── requirements.txt       # Dependencies
├── .gitignore
└── README.md
```

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate      # Mac/Linux
# venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

## Running the Lab

### Step 1: Train and Track
```bash
python linear_regression.py
```

### Step 2: View MLflow UI
```bash
mlflow ui --port=5001
```
Open browser → http://localhost:5001

### Step 3: Serve the Model
Replace `REPLACE_WITH_RUN_ID` in `serving.py` with your actual Run ID from the UI, then:
```bash
python serving.py
```

## What MLflow Tracks Automatically
- Model parameters (alpha)
- Metrics (RMSE, MAE, R2)
- Model artifact
- Input/output signature
