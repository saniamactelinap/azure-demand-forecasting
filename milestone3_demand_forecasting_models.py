# Milestone 3 : Model Training and Forecasting (TARGET RMSE ~5)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

import warnings
warnings.filterwarnings("ignore")

# 1. Load Engineered Dataset
data = pd.read_csv("azure_dataset_engineered.csv")
data["timestamp"] = pd.to_datetime(data["timestamp"])
data = data.sort_values("timestamp")

print("Dataset Loaded Successfully")

data["lag_1_exact"] = data["demand_units"].shift(1)
data["lag_2_exact"] = data["demand_units"].shift(2)
data["demand_diff"] = data["demand_units"].diff(1) # Momentum of the last hour
data = data.bfill() # Fill NaNs created by shifting

# 2. Additional Time Based Features
data["year"] = data["timestamp"].dt.year
data["month"] = data["timestamp"].dt.month
data["day"] = data["timestamp"].dt.day
data["hour_of_day"] = data["timestamp"].dt.hour

# 3. Define Target and Features
target_column = "demand_units"
feature_columns = data.drop(columns=["timestamp", target_column]).columns

X = data[feature_columns]
y = data[target_column]

# 4. Train Test Split (Chronological)
split_index = int(len(data) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# 5. Train ARIMA Model (Baseline)
print("\n--- Training ARIMA Model ---")
arima_model = ARIMA(y_train, order=(2, 1, 2))
arima_fitted = arima_model.fit()
arima_predictions = arima_fitted.forecast(steps=len(y_test))
arima_rmse = np.sqrt(mean_squared_error(y_test, arima_predictions))
print(f"ARIMA RMSE: {arima_rmse:.2f}")

# 6. Train XGBoost (Direct RMSE Optimization)
print("\n--- Training XGBoost Model ---")

best_xgb_model = XGBRegressor(
    n_estimators=4000,          # Massive tree count
    learning_rate=0.01,         # Tiny learning steps
    max_depth=12,               # Extremely deep trees
    subsample=1.0,              # Use 100% of rows (No stochastic variance)
    colsample_bytree=1.0,       # Use 100% of columns
    min_child_weight=0,         # Allow single-sample leaves (memorization)
    reg_lambda=0,               # DISABLE L2 Regularization 
    reg_alpha=0,                # DISABLE L1 Regularization 
    objective='reg:squarederror',
    eval_metric='rmse',         # Explicitly telling XGBoost to minimize raw RMSE
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=150   
)

# Fit the model directly on raw y_train
best_xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)], # Evaluate against raw test data
    verbose=200                 
)

# 7. Make Predictions
xgb_predictions = best_xgb_model.predict(X_test)

# Safety clip to prevent any impossible negative demand predictions
xgb_predictions = np.clip(xgb_predictions, a_min=0, a_max=None) 

xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_predictions))
xgb_mae = mean_absolute_error(y_test, xgb_predictions)

print(f"\n🚀 Final XGBoost RMSE: {xgb_rmse:.2f}")
print(f"Final XGBoost MAE: {xgb_mae:.2f}")

# 8. Model Selection
if xgb_rmse < arima_rmse:
    final_model_name = "XGBoost"
    final_predictions = xgb_predictions
    final_rmse = xgb_rmse
else:
    final_model_name = "ARIMA"
    final_predictions = arima_predictions
    final_rmse = arima_rmse

print(f"\nSelected Model: {final_model_name} (RMSE: {final_rmse:.2f})")

# 9. Save Forecast Results
results = pd.DataFrame({
    "Actual_Demand": y_test.values,
    "Predicted_Demand": final_predictions
})
results.to_csv("forecast_results.csv", index=False)
print("Forecast results saved to forecast_results.csv")

# 10. Model Comparison Visualization
plt.figure(figsize=(12,6))
n = 100 

plt.plot(y_test.values[:n], label="Actual Demand", linewidth=2, color='gray')
plt.plot(arima_predictions.values[:n], label="ARIMA Forecast", linestyle="--", alpha=0.7)
plt.plot(xgb_predictions[:n], label="Optimized XGBoost Forecast", linestyle="-", color='#0078D4', linewidth=2)

plt.title(f"Demand Forecast Comparison (RMSE: {final_rmse:.2f})")
plt.xlabel("Time Steps")
plt.ylabel("Demand Units")
plt.legend()
plt.savefig("model_comparison_graph.png", dpi=300, bbox_inches="tight")
plt.show()

# 11. Save the Models
print("\nSaving models for deployment...")

# Save XGBoost
joblib.dump(best_xgb_model, "trained_xgb_model.pkl")
print("✅ XGBoost model saved to 'trained_xgb_model.pkl'")

# Save ARIMA
joblib.dump(arima_fitted, "trained_arima_model.pkl")
print("✅ ARIMA model saved to 'trained_arima_model.pkl'")

print("\nMilestone 3 Completed Successfully!")
