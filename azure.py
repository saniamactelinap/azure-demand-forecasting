import pandas as pd

# Load dataset
df = pd.read_csv("azure_dataset.csv")

# Convert date column
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by='date')

# Standardize region names
df['azure_region'] = (
    df['azure_region']
    .str.lower()
    .str.replace(" ", "-")
)

df['azure_region'] = df['azure_region'].replace({
    'us-east': 'US-East',
    'us-west': 'US-West',
    'india-west': 'India-West',
    'india-south': 'India-South',
    'europe-north': 'Europe-North'
})

# Remove duplicates
df = df.drop_duplicates()

# Handle missing values
df['demand_units'] = df['demand_units'].interpolate()
df['cost_usd'] = df['cost_usd'].fillna(df['demand_units'] * 0.5)
df['service_availability'] = df['service_availability'].ffill()
df['holiday'] = df['holiday'].fillna(0)

# Basic validation
print("Dataset shape:", df.shape)
print("Missing values:\n", df.isnull().sum())
print("\nAverage demand by region:")
print(df.groupby('azure_region')['demand_units'].mean())