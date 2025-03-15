import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import re

# Load Data
df = pd.read_csv("data/Bengaluru_House_Data.csv")

def convert_sqft(value):
    try:
        if '-' in value:
            low, high = map(float, value.split('-'))
            return (low + high) / 2
        elif 'Acres' in value:
            return float(re.findall(r'\d+\.?\d*', value)[0]) * 43560
        elif value.replace('.', '', 1).isdigit():
            return float(value)
        else:
            return np.nan
    except:
        return np.nan

# Apply conversion
df['total_sqft'] = df['total_sqft'].astype(str).apply(convert_sqft)
df.dropna(subset=['total_sqft'], inplace=True)

# Select features and target
features = ['location', 'total_sqft', 'bath', 'bhk']
target = 'price'
X = df[features]
y = df[target]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train Model
pipeline.fit(X_train[['total_sqft', 'bath', 'bhk']], y_train)

# Save Model
with open("../models/house_price_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

# Save Locations
locations = sorted(df['location'].unique())
with open("../models/locations.pkl", "wb") as f:
    pickle.dump(locations, f)

print("Preprocessing complete. Model and locations saved.")
