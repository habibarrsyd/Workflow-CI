"""
Time Series Forecasting for Steel Industry Energy Consumption
============================================================

This module trains a single Random Forest model to forecast Usage_kWh 
for a CI/CD pipeline.

Author: Steel Industry Energy Forecasting Pipeline
Date: October 26, 2025
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_preprocessed_data():
    """Load preprocessed steel industry time series data"""
    try:
        # Path ke file preprocessed di dalam MLProject
        data_path = "dataset/steel_preprocessed.csv"
        
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        print(f"Data loaded successfully. Shape: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def prepare_time_series_features_target(df):
    """Prepare time series features and target variable for forecasting"""
    target_col = 'Usage_kWh'
    feature_cols = [col for col in df.columns 
                    if col != target_col and not col.startswith('Unnamed')]
    
    X = df[feature_cols]
    y = df[target_col]
    
    print(f"Time series features: {X.shape[1]} columns")
    print(f"Target samples: {y.shape[0]} time points")
    
    return X, y

def time_series_split(X, y, train_ratio=0.8):
    """Split time series data chronologically (no shuffling)"""
    split_index = int(len(X) * train_ratio)
    
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]
    
    print(f"Training period: {X_train.index.min()} to {X_train.index.max()}")
    print(f"Testing period: {X_test.index.min()} to {X_test.index.max()}")
    
    return X_train, X_test, y_train, y_test

def calculate_forecasting_metrics(y_true, y_pred):
    """Calculate time series forecasting specific metrics"""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100
    
    return {
        'mse': mse,
        'mae': mae, 
        'r2': r2,
        'rmse': rmse,
        'mape': mape,
        'smape': smape,
    }

def create_forecast_visualization(y_true, y_pred, model_name, test_index):
    """Create time series forecast visualization"""
    
    plt.figure(figsize=(15, 6)) # Disederhanakan jadi 1 plot
    
    plt.plot(test_index, y_true, label='Actual Usage_kWh', color='blue', alpha=0.7)
    plt.plot(test_index, y_pred, label='Predicted Usage_kWh', color='red', alpha=0.7)
    plt.title(f'{model_name} - Energy Consumption Forecast')
    plt.xlabel('Time')
    plt.ylabel('Usage (kWh)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plot_filename = f"{model_name.lower().replace(' ', '_')}_forecast.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_filename

def train_single_model_for_ci():
    """Train ONE Random Forest model and log it with MLflow for CI/CD."""
    
    df = load_preprocessed_data()
    if df is None:
        return
    
    X, y = prepare_time_series_features_target(df)
    X_train, X_test, y_train, y_test = time_series_split(X, y, train_ratio=0.8)
    
    mlflow.set_experiment("Steel_Industry_Energy_Forecasting")
    
    print("\nTraining Random Forest for CI/CD pipeline...")
    with mlflow.start_run(run_name="Random_Forest_For_Docker"):
        
        # --- Model Definition ---
        params = {
            "n_estimators": 100,
            "max_depth": 15,
            "min_samples_split": 5,
            "random_state": 42,
            "n_jobs": -1
        }
        rf_model = RandomForestRegressor(**params)
        rf_model.fit(X_train, y_train)
        
        # --- Predictions ---
        y_pred_test = rf_model.predict(X_test)
        
        # --- Metrics ---
        test_metrics = calculate_forecasting_metrics(y_test, y_pred_test)
        
        # --- Logging to MLflow ---
        print("Logging parameters, metrics, and model to MLflow...")
        
        # 1. Log Parameters
        mlflow.log_params(params)
        
        # 2. Log Metrics
        # Ubah nama metrik agar valid (ganti / dengan _)
        test_metrics_renamed = {f"test_{k}": v for k, v in test_metrics.items()}
        mlflow.log_metrics(test_metrics_renamed)
        
        # 3. Log Artifact (Plot)
        plot_file = create_forecast_visualization(y_test, y_pred_test, "Random Forest", X_test.index)
        mlflow.log_artifact(plot_file)
        
        # 4. Log Model (Paling Penting)
        # Simpan dengan nama "model" agar mlflow build-docker menemukannya
        mlflow.sklearn.log_model(
            sk_model=rf_model,
            artifact_path="model",  # Ini adalah nama folder artefak
            input_example=X_train.head() # Contoh input untuk schema
        )
        
        print(f"Random Forest - Test R2: {test_metrics['r2']:.4f}, Test MAPE: {test_metrics['mape']:.2f}%")
        print("✅ Model, metrics, and plots logged successfully for CI/CD.")

if __name__ == "__main__":
    train_single_model_for_ci()
    print("CI/CD Model training script completed!")