import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = {
    "Month": pd.date_range(start="2015-01-01", periods=60, freq="M"),
    "Consumption": [120, 125, 130, 128, 122, 121, 135, 140, 145, 142, 130, 125] * 5,
}

df = pd.DataFrame(data)

df["Time_Index"] = np.arange(len(df))  
df["Month_Sin"] = np.sin(2 * np.pi * df["Time_Index"] / 12) 
df["Month_Cos"] = np.cos(2 * np.pi * df["Time_Index"] / 12)  

X = df[["Time_Index", "Month_Sin", "Month_Cos"]]
y = df["Consumption"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Coefficients:", dict(zip(X.columns, model.coef_)))
print("Intercept:", model.intercept_)

future_time_index = np.arange(len(df), len(df) + 12)
future_features = pd.DataFrame({
    "Time_Index": future_time_index,
    "Month_Sin": np.sin(2 * np.pi * future_time_index / 12),
    "Month_Cos": np.cos(2 * np.pi * future_time_index / 12),
})

future_predictions = model.predict(future_features)

future_results = pd.DataFrame({
    "Month": pd.date_range(start=df["Month"].iloc[-1] + pd.offsets.MonthEnd(), periods=12, freq="M"),
    "Predicted_Consumption": future_predictions,
})
print("\nFuture Predictions:")
print(future_results)
