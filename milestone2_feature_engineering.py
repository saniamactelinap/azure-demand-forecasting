# MILESTONE 2

import pandas as pd
import numpy as np

# Load cleaned dataset
df = pd.read_csv("azure_dataset.csv")

# Ensure datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Sort chronologically
df = df.sort_values("timestamp").reset_index(drop=True)

# Time Based Features

df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.day
df["weekday"] = df["timestamp"].dt.weekday
df["month"] = df["timestamp"].dt.month
df["quarter"] = df["timestamp"].dt.quarter

df["is_weekend"] = (df["weekday"] >= 5).astype(int)
df["is_month_start"] = df["timestamp"].dt.is_month_start.astype(int)
df["is_month_end"] = df["timestamp"].dt.is_month_end.astype(int)
df["is_business_hour"] = df["hour"].between(8, 20).astype(int)

# Lag Features

df["lag_1"] = df["demand_units"].shift(1)
df["lag_24"] = df["demand_units"].shift(24)
df["lag_168"] = df["demand_units"].shift(168)

# Rolling Statistics

df["rolling_mean_6"] = df["demand_units"].rolling(6).mean()
df["rolling_mean_24"] = df["demand_units"].rolling(24).mean()
df["rolling_std_24"] = df["demand_units"].rolling(24).std()
df["rolling_max_24"] = df["demand_units"].rolling(24).max()
df["rolling_min_24"] = df["demand_units"].rolling(24).min()

df["demand_growth_rate"] = df["demand_units"].pct_change()

# Spike Detection

mean = df["demand_units"].mean()
std = df["demand_units"].std()

df["usage_spike"] = (df["demand_units"] > mean + std).astype(int)
df["extreme_spike"] = (df["demand_units"] > mean + 2*std).astype(int)

# Demand Driving Features

urgency_map = {"Low": 1, "Medium": 2, "High": 3}
df["urgency_encoded"] = df["customer_demand_urgency"].map(urgency_map)

sla_map = {"Standard": 1, "Premium": 2, "Mission-Critical": 3}
df["sla_encoded"] = df["sla_priority_level"].map(sla_map)

df["low_availability_flag"] = (df["service_availability_pct"] < 99).astype(int)
df["cost_per_unit"] = df["cost_usd"] / df["demand_units"]
df["is_auto_allocated"] = (df["capacity_allocator"] == "Auto").astype(int)

df["priority_score"] = df["urgency_encoded"] * df["sla_encoded"]

# One Hot Encoding
df = pd.get_dummies(df, columns=["azure_region", "service_type"], drop_first=True)

# Drop original categorical columns
df = df.drop(columns=["customer_demand_urgency",
                      "sla_priority_level",
                      "capacity_allocator"])

# Remove Nulls
df = df.dropna().reset_index(drop=True)

# Save Engineered Dataset
df.to_csv("azure_dataset_engineered.csv", index=False)

