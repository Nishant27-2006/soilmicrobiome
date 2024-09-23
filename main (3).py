
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset (replace 'sample.csv' with your file path)
data = pd.read_csv('sample.csv')

# Features for prediction
features = ['TEMP', 'DEWP', 'PRCP', 'MXSPD']
target = 'TEMP'  # Placeholder for soil-related target (e.g., soil moisture)

# Drop missing values
data_clean = data[features].dropna()

# Split the data into train and test sets
X = data_clean.drop(columns=[target])
y = data_clean[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Make predictions and calculate mean squared error
y_pred = rf_model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plot True vs Predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title("True vs Predicted Temperature (Random Forest Model)")
plt.xlabel("True Temperature")
plt.ylabel("Predicted Temperature")
plt.grid(True)
plt.savefig("true_vs_pred_rf.png")

# Feature Importance visualization
feature_importances = rf_model.feature_importances_
plt.figure(figsize=(8, 6))
plt.barh(features[:-1], feature_importances, color='green')
plt.title("Feature Importance in Random Forest Model")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.grid(True)
plt.savefig("feature_importance_rf.png")
