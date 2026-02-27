# Azure Demand Forecasting & Capacity Optimization System

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
- Machine Learning (future milestones)
- Azure Machine Learning concepts

---

## Repository Structure

```
azure-demand-forecasting/
├── azure.py
├── azure_dataset.csv
├── milestone2_feature_engineering.py
├── azure_dataset_engineered.py
├── visualization.py
├── README.md
└── MIT License
└── plots/
      ├── demand_trend.png
      ├── demand_by_service.png
      ├── cost_by_region.png
      ├── demand_by_urgency.png
      └── demand_by_sla.png
```

---

## Dataset Description

**File:** `azure_dataset.csv` (Milestone 1 output)

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

## Current Status
🟢 **Milestone 1: Completed**
🟢 **Milestone 2: Completed**  
🟡 **Milestone 3: Model Development (Upcoming)**

---

## Future Work
- Build forecasting models (ARIMA / XGBoost / LSTM)  
- Evaluate model accuracy  
- Optimize hyperparameters  
- Generate capacity planning insights  
- Build visualization dashboards

---

## License
This project is licensed under the **MIT License**.  
See the `MIT License` file for more details.

