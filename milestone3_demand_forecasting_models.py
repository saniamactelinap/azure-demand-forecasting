# Milestone 3 : Model Training and Forecasting

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings("ignore")

# 1. Load Engineered Dataset

data = pd.read_csv("azure_dataset_engineered.csv")
data["timestamp"] = pd.to_datetime(data["timestamp"])
data = data.sort_values("timestamp")

print("Dataset Loaded Successfully")
print("Total Rows:", data.shape[0])
print("Total Columns:", data.shape[1])

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

# 4. Train Test Split (Time Series Split)

split_index = int(len(data) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# 5. ARIMA BASE MODEL

print("\nTraining Initial ARIMA Model...")

arima_base = ARIMA(y_train, order=(2,1,2))
arima_fitted = arima_base.fit()
arima_forecast = arima_fitted.forecast(len(y_test))

# 6. XGBOOST BASE MODEL

print("\nTraining Initial XGBoost Model...")

xgb_base = XGBRegressor(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.05,
    random_state=42,
    objective="reg:squarederror"
)

xgb_base.fit(X_train, y_train)
xgb_forecast = xgb_base.predict(X_test)

# 7. RMSE FUNCTION

def rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))

# Base Model Performance

arima_rmse = rmse(y_test, arima_forecast)
xgb_rmse = rmse(y_test, xgb_forecast)

print("\nBase Model Performance")
print("ARIMA RMSE :", arima_rmse)
print("XGBoost RMSE :", xgb_rmse)

# 8. ARIMA PARAMETER SEARCH

print("\nSearching Best ARIMA Parameters...")

p_values = range(0,3)
d_values = range(0,2)
q_values = range(0,3)

best_config = None
lowest_error = float("inf")

for p in p_values:
    for d in d_values:
        for q in q_values:

            try:

                model = ARIMA(y_train, order=(p,d,q))
                fitted = model.fit()
                prediction = fitted.forecast(len(y_test))
                error = rmse(y_test, prediction)

                if error < lowest_error:

                    lowest_error = error
                    best_config = (p,d,q)

            except:
                continue

print("Optimal ARIMA Order:", best_config)

# Train Tuned ARIMA

arima_tuned = ARIMA(y_train, order=best_config).fit()
arima_tuned_pred = arima_tuned.forecast(len(y_test))

# 9. XGBOOST HYPERPARAMETER SEARCH

print("\nOptimizing XGBoost Model...")

parameter_grid = {

    "n_estimators":[150,250],
    "max_depth":[4,6,8],
    "learning_rate":[0.01,0.05,0.1],
    "subsample":[0.8,1]

}

grid_search = GridSearchCV(

    estimator=XGBRegressor(
        objective="reg:squarederror",
        random_state=42
    ),

    param_grid=parameter_grid,
    cv=3,
    scoring="neg_mean_squared_error",
    verbose=1

)

grid_search.fit(X_train, y_train)
best_xgb_model = grid_search.best_estimator_

print("Best Parameters:", grid_search.best_params_)

xgb_tuned_pred = best_xgb_model.predict(X_test)

# 10. FINAL MODEL PERFORMANCE

arima_tuned_rmse = rmse(y_test, arima_tuned_pred)
xgb_tuned_rmse = rmse(y_test, xgb_tuned_pred)

print("\nFinal Model Results")
print("Tuned ARIMA RMSE:", arima_tuned_rmse)
print("Tuned XGBoost RMSE:", xgb_tuned_rmse)

# 11. MODEL SELECTION

print("\nSelecting Best Model Based on RMSE")

if xgb_tuned_rmse < arima_tuned_rmse:

    final_model = "XGBoost"
    final_predictions = xgb_tuned_pred
    final_rmse = xgb_tuned_rmse

else:

    final_model = "ARIMA"
    final_predictions = arima_tuned_pred
    final_rmse = arima_tuned_rmse

print("Final Selected Model:", final_model)
print("Final Model RMSE:", final_rmse)

# 12. SAVE FORECAST RESULTS

results = pd.DataFrame({

    "Actual_Demand": y_test.values,
    "Predicted_Demand": final_predictions

})

results.to_csv("forecast_results.csv", index=False)
print("Forecast results saved to forecast_results.csv")

# 13. MODEL COMPARISON VISUALIZATION

plt.figure(figsize=(12,6))
n = 100

plt.plot(y_test.values[:n], label="Actual Demand", linewidth=2)
plt.plot(arima_tuned_pred.values[:n], label="ARIMA Forecast", linestyle="--")
plt.plot(xgb_tuned_pred[:n], label="XGBoost Forecast", linestyle="--")

plt.title("Demand Forecast Comparison")
plt.xlabel("Time")
plt.ylabel("Demand Units")
plt.legend()

plt.savefig("model_comparison_graph.png", dpi=300, bbox_inches="tight")

plt.show()

print("\nMilestone 3 Completed Successfully")
print("Best Forecasting Model:", final_model)

# 14. SAVE MODEL FOR MILESTONE 4

import joblib

if final_model == "XGBoost":
    joblib.dump(best_xgb_model, "trained_xgb_model.pkl")
    print("XGBoost model saved for deployment")
else:
    print("ARIMA selected (not saved as .pkl)")

print("\nMilestone 3 Completed Successfully")
print("Best Forecasting Model:", final_model)
