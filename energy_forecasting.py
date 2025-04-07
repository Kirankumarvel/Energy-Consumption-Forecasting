# energy_forecasting.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Removed unused seaborn import

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import skew

# Load dataset (example: energy dataset from UCI or synthetic for now)
try:
	try:
		data = pd.read_csv(r"C:\Users\kiran\OneDrive\Desktop\python\machine-learning-python\Random Forest Performance\data\data\energy_data.csv")  # Ensure this file exists in the 'data' folder
		if data.empty:
			raise pd.errors.EmptyDataError("Dataset is empty.")
	except (FileNotFoundError, pd.errors.EmptyDataError) as e:
		print(f"Error: {e}. Generating synthetic data.")
		rng = np.random.default_rng(seed=42)  # Use numpy's random generator with a seed for reproducibility
		data = pd.DataFrame({
			"Feature1": rng.random(100),
			"Feature2": rng.random(100),
			"Appliances": rng.integers(50, 500, size=100)
		})
except FileNotFoundError:
	print("Error: The file 'data/energy_data.csv' was not found. Please check the file path.")
	exit()

# Basic preprocessing
data = data.dropna()
X = data.drop("Appliances", axis=1)
y = data["Appliances"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42, max_features='sqrt', min_samples_leaf=1)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")

# Plot Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color="blue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel("Actual Energy Consumption")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted Energy Consumption")
plt.show()

# Residuals
residuals = y_test - y_pred
plt.hist(residuals, bins=30, color='lightblue', edgecolor='black')
plt.title(f"Residuals Distribution\nSkewness: {skew(residuals):.2f}")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# Feature Importances
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()
