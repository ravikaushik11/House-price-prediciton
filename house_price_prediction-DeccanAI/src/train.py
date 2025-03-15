import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load dataset
df = pd.read_csv("data/Bengaluru_House_Data.csv")
logger.info("Dataset loaded successfully.")

# Drop unnecessary columns if they exist
columns_to_drop = ["society", "availability"]
df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

# Handle missing values
df.dropna(inplace=True)

# Identify categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns
label_encoders = {}

# Encode categorical columns
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
logger.info("Categorical features encoded successfully.")

# Ensure the models directory exists
os.makedirs("models", exist_ok=True)

# Save encoders for future use
with open("models/label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)
logger.info("Label encoders saved.")

# Define features and target
X = df.drop(columns=["price"])
y = df["price"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logger.info("Dataset split into training and testing sets.")

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)
logger.info("Model training completed.")

# Save trained model
with open("models/house_price_model.pkl", "wb") as f:
    pickle.dump(model, f)
logger.info("Trained model saved.")

# Model Evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
logger.info(f"Evaluation - MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R²: {r2}")
