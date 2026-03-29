# Azure Demand Forecasting & Capacity Optimization System

##[Deployed website:](https://azure-demand-forecasting.streamlit.app/)

##[Video link](https://drive.google.com/file/d/1htWkPRJ4Le2JVve0hLu461IDOEAMPYA0/view?usp=drive_link)

## Overview
This project focuses on building a data driven system to forecast **Azure Compute and Storage demand**
using historical usage data. The objective is to assist capacity planning teams in making informed
provisioning decisions, thereby reducing both **over provisioning** and **under provisioning**
of cloud infrastructure.

Since real Azure production data is confidential, representative **synthetic datasets** are used
to simulate realistic demand patterns across regions and services.

---

## Project Objectives
- Forecast Azure Compute and Storage demand accurately
- Support efficient capacity planning and provisioning decisions
- Improve utilization of cloud infrastructure
- Reduce cost impact caused by inaccurate demand estimation

---

## Scope of the Project
- Historical time series demand data (12–24 months)
- Multiple Azure regions
- Compute and Storage services only
- Focus on data preparation and modeling methodology

---

## Tech Stack
- Python
- Pandas
- NumPy
- Matplotlib
- Statsmodels(ARIMA)
- XGBoost
- Scikit-learn
- Feature Engineering Techniques  
- Machine Learning 
- Azure Machine Learning concepts
- Streamlit

---

## Repository Structure

```
azure-demand-forecasting/
│
├── azure_dataset.csv
├── azure_dataset_engineered.csv
│
├── azure.py
├── milestone2_feature_engineering.py
├── milestone3_demand_forecasting_models.py
├── milestone4_dashboard.py
│
├──trained_arima_model.pkl 
├──trained_xgb_model.pkl 
│
├── forecast_results.csv
├── model_comparison_graph.png
│
├── visualization.py
│
├── requirements.txt
├── README.md
└── MIT License
│
└── documentations/
│
└── plots/
      ├── demand_trend.png
      ├── demand_by_service.png
      ├── cost_by_region.png
      ├── demand_by_urgency.png
      └── demand_by_sla.png

```

---

## Dataset Description

**File:** `azure_dataset.csv` (Milestone 1 Output)

The dataset represents historical Azure service demand and includes the following fields:

- Date of usage (date only)
- Azure region
- Service type (Compute / Storage)
- Demand units
- Capacity allocator
- Cost incurred (USD)
- Service availability
- Demand Urgency
- SLA Priority

This dataset is synthetically generated to closely resemble real world Azure demand behavior.

**File:** `azure_dataset_engineered.csv` (Milestone 2 Output)

Feature enriched and model ready dataset including:

### Time Based Features
- Hour  
- Day  
- Weekday  
- Month  
- Quarter  
- Weekend flag  
- Business hour flag  
- Month start/end indicators  

### Trend & Lag Features
- Lag_1 (previous period demand)  
- Lag_24 (daily lag)  
- Lag_168 (weekly lag)  
- Rolling mean (7 period)  
- Rolling max  
- Rolling min  
- Rolling standard deviation  
- Growth rate  

### Demand Behavior Features
- Usage spike flag  
- Extreme spike flag  
- Demand volatility flag  

### Service & User Behavior Features
- SLA encoding  
- Urgency encoding  
- Priority score  
- Low availability flag  
- Cost per unit  
- Auto allocation flag  

This dataset is fully numeric and optimized for machine learning models.

---

## Milestones

### Milestone 1: Data Collection & Preparation
- Collected historical demand data representing Azure Compute and Storage usage
- Included regional demand variations
- Cleaned and validated datasets for consistency and accuracy
- Prepared data for feature engineering and future modeling

---

## Milestone 2: Feature Engineering & Data Wrangling 

- Identified demand driving features (trends, uptime, user behavior)  
- Engineered seasonality indicators  
- Created lag and rolling statistics features  
- Detected usage spikes and volatility patterns  
- Encoded categorical features into numeric form  
- Reshaped dataset into model ready structure  
- Generated a fully numeric engineered dataset for ML modeling  

---

### Milestone 3: Machine Learning Model Development

The objective of this milestone was to **train and evaluate forecasting models** to predict Azure demand accurately.

#### Models Implemented

**1. ARIMA (AutoRegressive Integrated Moving Average)**
- Traditional time series forecasting model
- Suitable for sequential demand patterns
- Tuned using parameter search for optimal (p,d,q) configuration
- [trained_arima_model.pkl](https://drive.google.com/file/d/1cE23UV8QnUo0Fy2W4KwMYttqXk8fVaMU/view?usp=drive_link)

**2. XGBoost Regressor**
- Gradient boosting machine learning model
- Handles nonlinear patterns and feature interactions effectively
- Hyperparameter tuning performed using GridSearchCV
- [trained_xgb_model.pkl](https://drive.google.com/file/d/1SwwuTu5Hgy7L-zM35gszHLMwEgYlB6_0/view?usp=drive_link)

---

#### Model Evaluation Metrics
Models were evaluated using:

- RMSE (Root Mean Squared Error)
- Forecast comparison visualization
- Prediction vs Actual demand comparison

---

#### Model Comparison

The forecasting results showed:

- ARIMA produced smooth average forecasts
- XGBoost captured demand spikes and volatility more accurately

Therefore, **XGBoost was selected as the final forecasting model** due to its superior performance on the test dataset.

---

# Milestone 4: Dashboard & Capacity Optimization

The objective of this milestone was to **develop an interactive dashboard** that visualizes Azure demand forecasts and provides actionable insights for capacity planning and optimization.

---

## Dashboard Features

**1. Interactive Forecast Visualization**
- Compare **actual demand** versus **predicted demand** from the XGBoost model  
- Filter by **service type** (Compute / Storage) and **Azure region**  
- Inspect trends and spikes with dynamic, interactive charts  

**2. Capacity Planning Insights**
- Identify periods of **over provisioning** and **under provisioning**  
- Detect **demand spikes** and **high volatility periods**  
- Recommend **resource allocation adjustments** to optimize utilization  

**3. Cost Analysis**
- Compare **predicted vs actual cost per unit**  
- Highlight potential **cost savings** from optimized capacity  

**4. KPI Overview**
- Key metrics including **total demand**, **forecast accuracy**, and **forecast vs actual deviations**  
- Visual summaries for **regions, services, and SLA priorities**  

**5. User-Friendly Interface**
- Built using **Streamlit** for rapid prototyping and deployment  
- Modular layout with KPI overview, trend analysis, and optimization recommendations  
- Real-time filtering and interactive visualizations for decision support  

---

## Dashboard Impact

- Provides **data-driven decision support** for cloud capacity planning  
- Reduces the risk of **over-provisioning**, lowering costs  
- Improves **resource utilization** and service reliability  
- Enables **strategic planning** based on forecast insights  

---

### Visualization Output

The project includes visualization comparing:

- Actual demand
- ARIMA predictions
- XGBoost predictions

This comparison graph is stored as:
`model_comparison_graph.png`

## Current Status
🟢 **Milestone 1: Completed**
🟢 **Milestone 2: Completed**  
🟢 **Milestone 3: Completed** 
🟢 **Milestone 4: Completed** 

---

## License
This project is licensed under the **MIT License**.  
See the `MIT License` file for more details.

