import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

data = {
    "Location_Score": [8, 6, 9, 7, 6, 5, 8],
    "Size_SqFt": [1200, 850, 1500, 1000, 800, 650, 1300],
    "Bedrooms": [3, 2, 4, 3, 2, 1, 3],
    "Price": [300000, 200000, 400000, 250000, 180000, 150000, 320000],
}
df = pd.DataFrame(data)

X = df[["Location_Score", "Size_SqFt", "Bedrooms"]]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r_squared = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

{
    "R-squared": round(r_squared, 2),
    "Mean Squared Error": round(mse, 2),
    "Intercept": model.intercept_,
    "Coefficients": dict(zip(X.columns, model.coef_)),
}
