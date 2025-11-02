import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.datasets import fetch_california_housing
import pickle  

# --- Load dataset ---
data = fetch_california_housing(as_frame=True)
df = data.frame

# --- Features and Target ---
X = df.drop(columns=['MedHouseVal'])
y = df['MedHouseVal'] * 100000  # Convert to USD

# --- Split the data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Load saved model ---
model = joblib.load('model/california_housing.pkl')

# --- Make predictions ---
y_pred = model.predict(X_test)

# --- Evaluate performance ---
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"Model R² Score: {r2:.3f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
