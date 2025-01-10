import mlflow
import mlflow.pyfunc
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit

# Load the dataset
data = pd.read_csv("/mnt/c/Users/Ahmad/Desktop/MLOPS/Project2/course-project-OsamaFahim654/enviromental_monitoring/data/combined_preprocessed_data.csv", parse_dates=['date'])

# We will use 'components.pm2_5' as the target variable
target_column = 'components.pm2_5'

# Select features (exclude the target column and the timestamp column)
features = data.drop(columns=[target_column, 'date'])

# Split data into training and testing datasets
X = features
y = data[target_column]

# Train-test split (for validation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Function to perform GridSearch on ARIMA (using TimeSeriesSplit for time series cross-validation)
def grid_search_arima(y_train):
    best_rmse = float('inf')
    best_model = None
    best_params = None
    
    # Define the grid of hyperparameters (p, d, q)
    p_values = [0, 1, 2]
    d_values = [0, 1]
    q_values = [0, 1, 2]
    
    # Time series cross-validation (important to keep time order)
    tscv = TimeSeriesSplit(n_splits=5)
    
    for p in p_values:
        for d in d_values:
            for q in q_values:
                print(f"Training ARIMA({p},{d},{q})...")
                
                # Perform cross-validation
                rmse_values = []
                for train_index, val_index in tscv.split(y_train):
                    train, val = y_train.iloc[train_index], y_train.iloc[val_index]
                    
                    # Train the ARIMA model
                    model = ARIMA(train, order=(p, d, q))
                    model_fit = model.fit()
                    
                    # Predict on validation set
                    predictions = model_fit.predict(start=val.index[0], end=val.index[-1])
                    rmse = np.sqrt(mean_squared_error(val, predictions))
                    rmse_values.append(rmse)
                
                # Calculate average RMSE for the current model
                avg_rmse = np.mean(rmse_values)
                if avg_rmse < best_rmse:
                    best_rmse = avg_rmse
                    best_model = ARIMA(y_train, order=(p, d, q)).fit()
                    best_params = (p, d, q)
                    
    print(f"Best ARIMA Model: ARIMA{best_params} with RMSE: {best_rmse}")
    return best_model, best_params, best_rmse

# Perform grid search for hyperparameter tuning
best_arima_model, best_params, best_rmse = grid_search_arima(y_train)

# Create the ARIMAModel class for mlflow logging
class ARIMAModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
    
    def predict(self, context, model_input):
        predictions = self.model.predict(start=len(self.model.data.endog), end=len(self.model.data.endog) + len(model_input) - 1)
        return predictions

# Log the best model to MLflow
with mlflow.start_run():
    # Log the ARIMA model as part of the custom Python model
    mlflow.pyfunc.log_model(
        artifact_path="arima_model",
        python_model=ARIMAModel(model=best_arima_model)
    )

    # Save and log the ARIMA model as a .joblib file
    model_path = "/mnt/c/Users/Ahmad/Desktop/MLOPS/Project2/course-project-OsamaFahim654/enviromental_monitoring/arima_model.joblib"
    joblib.dump(best_arima_model, model_path)
    mlflow.log_artifact(model_path)

    # Log the hyperparameters (p, d, q)
    mlflow.log_param('best_p', best_params[0])
    mlflow.log_param('best_d', best_params[1])
    mlflow.log_param('best_q', best_params[2])

    # Logging model evaluation metrics
    start_index = len(y_train)  # Start where the test data begins
    end_index = len(y_train) + len(y_test) - 1  # End where the test data end
    predictions = best_arima_model.predict(start=start_index, end=end_index)
    
    #Evaluation metrics of the best model
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R2: {r2}")

    mlflow.log_metric('rmse', rmse)
    mlflow.log_metric('mae', mae)
    mlflow.log_metric('r2', r2)

    # Visualization of predictions vs actual values
    plot_path = "/mnt/c/Users/Ahmad/Desktop/MLOPS/Project2/course-project-OsamaFahim654/enviromental_monitoring/arima_predictions_vs_actual.png"
    plt.figure(figsize=(10, 6))
    plt.plot(y, label="Actual Values")
    plt.plot(range(start_index, end_index + 1), predictions, color='red', label="Predicted Values")
    plt.title("ARIMA Model Predictions vs Actual Values")
    plt.legend()
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path)
