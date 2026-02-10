# Azure Demand Forecasting & Capacity Optimization System

## Overview
This project focuses on building a data-driven system to forecast Azure Compute and Storage
demand using historical usage data. The objective is to assist capacity planning teams in
making informed provisioning decisions, thereby reducing over-provisioning and
under-provisioning of cloud infrastructure.

Since real Azure production data is confidential, representative synthetic datasets are used
to simulate realistic demand patterns across regions and services.

---

## Project Objectives
- Forecast Azure Compute and Storage demand
- Support efficient capacity planning decisions
- Improve utilization of cloud infrastructure
- Reduce cost impact due to inaccurate demand estimation

---

## Scope of the Project
- Historical time-series demand data (12–24 months)
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
azure-demand-forecasting/
│
├── data/
│ ├── raw/ # Raw historical demand data
│ └── processed/ # Cleaned and validated datasets
│
├── notebooks/ # Data analysis and experimentation notebooks
├── src/ # Python scripts
│
├── README.md # Project documentation
├── LICENSE # MIT License

---

## Milestones

### Milestone 1: Data Collection & Preparation
- Collected historical demand data representing Azure Compute and Storage usage
- Included regional demand variations
- Validated and cleaned datasets for consistency and accuracy
- Prepared datasets for feature engineering and modeling

---

## Status
🟡 **Milestone 1: In Progress / Completed**

---

## Author
Infosys Intern – Azure Demand Forecasting Project
