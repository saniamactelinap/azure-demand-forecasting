import pandas as pd
import matplotlib.pyplot as plt
import os

# Create folder to save plots
if not os.path.exists("plots"):
    os.makedirs("plots")

# Load dataset
df = pd.read_csv("azure_dataset.csv")

# Convert timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(by='timestamp')

plt.style.use("seaborn-v0_8")

# Demand Trend Over Time

plt.figure(figsize=(12,6))
df.groupby('timestamp')['demand_units'].sum().plot(color='blue')
plt.title("Total Demand Over Time", fontsize=14)
plt.xlabel("Time")
plt.ylabel("Demand Units")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/demand_trend.png")
plt.close()

# Average Demand by Service Type

plt.figure(figsize=(8,5))
df.groupby('service_type')['demand_units'].mean().plot(
    kind='bar',
    color=['orange','green']
)
plt.title("Average Demand by Service Type", fontsize=14)
plt.xlabel("Service Type")
plt.ylabel("Average Demand Units")
plt.tight_layout()
plt.savefig("plots/demand_by_service.png")
plt.close()

# Average Cost by Region

plt.figure(figsize=(10,5))
df.groupby('azure_region')['cost_usd'].mean().plot(
    kind='bar',
    color=['purple','red','teal','gold','brown']
)
plt.title("Average Cost by Region", fontsize=14)
plt.xlabel("Azure Region")
plt.ylabel("Average Cost (USD)")
plt.tight_layout()
plt.savefig("plots/cost_by_region.png")
plt.close()

# Demand by Urgency Level

plt.figure(figsize=(8,5))
df.groupby('customer_demand_urgency')['demand_units'].mean().plot(
    kind='bar',
    color=['green','orange','red']
)
plt.title("Average Demand by Urgency Level", fontsize=14)
plt.xlabel("Demand Urgency")
plt.ylabel("Average Demand Units")
plt.tight_layout()
plt.savefig("plots/demand_by_urgency.png")
plt.close()

# Demand by SLA Priority

plt.figure(figsize=(8,5))
df.groupby('sla_priority_level')['demand_units'].mean().plot(
    kind='bar',
    color=['blue','purple','crimson']
)
plt.title("Average Demand by SLA Priority Level", fontsize=14)
plt.xlabel("SLA Priority")
plt.ylabel("Average Demand Units")
plt.tight_layout()
plt.savefig("plots/demand_by_sla.png")
plt.close()

print("All visualizations saved inside 'plots/' folder")
