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

**File:** `azure_dataset.csv`

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

---

## Milestones

### Milestone 1: Data Collection & Preparation
- Collected historical demand data representing Azure Compute and Storage usage
- Included regional demand variations
- Cleaned and validated datasets for consistency and accuracy
- Prepared data for feature engineering and future modeling

---

## Current Status
🟢 **Milestone 1: Completed**

---

## Future Work
- Feature engineering and data enrichment
- Demand forecasting using machine learning models
- Model evaluation and accuracy improvement
- Integration with capacity planning insights

---

## License
This project is licensed under the **MIT License**.  
See the `MIT License` file for more details.

