# =============================================================================
# MLflow Lab 1 - Model Serving
# =============================================================================
# This script loads the trained Ridge Regression model from MLflow tracking
# and uses it to make predictions on a sample input from the Diabetes dataset.
#

import mlflow
import mlflow.pyfunc
import pandas as pd
from sklearn.datasets import load_diabetes

# Use the same SQLite backend as training so we can find the logged run
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# ------------------------------------------------------------------
# Prepare sample input from the Diabetes dataset
# Using the first row as a sample prediction input
# ------------------------------------------------------------------
diabetes = load_diabetes()
feature_names = diabetes.feature_names

sample_input = {name: [float(diabetes.data[0][i])]
                for i, name in enumerate(feature_names)}
sample_df = pd.DataFrame(sample_input)

# ------------------------------------------------------------------
# Load the model from MLflow using the Run ID logged during training
# The Run ID uniquely identifies which experiment run to load from
# ------------------------------------------------------------------
RUN_ID = "9e91b24b8bad4f308a50527423db9763"
model_uri = f"runs:/{RUN_ID}/model"

print(f"Loading model from MLflow run: {RUN_ID}")
loaded_model = mlflow.pyfunc.load_model(model_uri)

# ------------------------------------------------------------------
# Make prediction and display results
# ------------------------------------------------------------------
prediction = loaded_model.predict(sample_df)
print(f"\nSample Input:\n{sample_df.to_string(index=False)}")
print(f"\nPredicted diabetes progression: {prediction[0]:.2f}")
