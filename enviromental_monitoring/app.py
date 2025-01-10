from fastapi import FastAPI, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import pandas as pd
import numpy as np
import mlflow
import mlflow.pyfunc
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Initialize FastAPI app
app = FastAPI()

# Prometheus metrics
REQUEST_COUNT = Counter("app_requests_total", "Total number of requests")
PREDICTION_DURATION = Histogram("app_prediction_duration_seconds", "Time taken for prediction")
MSE_METRIC = Gauge("app_mse", "Mean Squared Error of the predictions")
RMSE_METRIC = Gauge("app_rmse", "Root Mean Squared Error of the predictions")
MAE_METRIC = Gauge("app_mae", "Mean Absolute Error of the predictions")
R2_METRIC = Gauge("app_r2", "R2 Score of the predictions")

# MLflow configuration
MLFLOW_TRACKING_URL = "http://localhost:5000"  # MLflow local server
MLFLOW_MODEL_PATH = "arima_model4"  # Registered model name
mlflow.set_tracking_uri(MLFLOW_TRACKING_URL)

def load_mlflow_model(model_name, version=1):
    """Fetch the model from MLflow Model Registry by version."""
    model_uri = f"models:/{model_name}/{version}"
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        return model
    except Exception as e:
        print(f"Error loading model from MLflow: {e}")
        raise e  # Re-raise the error to be caught by FastAPI's exception handler

@app.get("/predict/")
async def predict():
    """Endpoint to make predictions using the dynamically fetched MLflow model."""
    REQUEST_COUNT.inc()

    # Load the dataset
    DATA_PATH = 'data/combined_preprocessed_data.csv'
    if not os.path.exists(DATA_PATH):
        return Response(status_code=500, content="Dataset not found.")

    data = pd.read_csv(DATA_PATH, parse_dates=["date"])

    # Ensure consistency with train_model.py preprocessing
    target_column = "components.pm2_5"
    X = data.drop(columns=[target_column, "date"])
    y = data[target_column]

    # Match train-test split logic
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Load the MLflow model
    try:
        model = load_mlflow_model(MLFLOW_MODEL_PATH)
    except Exception as e:
        return Response(status_code=500, content=f"Failed to load model: {str(e)}")

    # Prepare the test data (ARIMA expects only the target series)
    try:
        # ARIMA expects a single series (time series data) for prediction
        # Pass y_train (or the target series) for prediction
        predictions = model.predict(y_test)  # Passing only the target variable (y_test)

        # Flatten the predictions to avoid array ambiguity
        predictions = np.array(predictions).flatten()
    except Exception as e:
        return Response(status_code=500, content=f"Error making predictions: {str(e)}")

    # Calculate evaluation metrics (MSE, RMSE, MAE, R2) using y_test and predictions
    try:
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        # Safely update Prometheus metrics
        MSE_METRIC.set(mse)
        RMSE_METRIC.set(rmse)
        MAE_METRIC.set(mae)
        R2_METRIC.set(r2)
    except Exception as e:
        return Response(status_code=500, content=f"Error calculating metrics: {str(e)}")

    # Return predictions and metrics
    return {
        "predictions": predictions.tolist(),
        "actual": y_test.tolist(),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }

@app.get("/metrics")
def metrics():
    """Endpoint for Prometheus to scrape metrics."""
    return Response(content=generate_latest(), media_type="text/plain")

@app.get("/")
def root():
    """Root endpoint to verify the app is running."""
    return {"message": "Welcome to the ARIMA prediction API!"}
