import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

data = {
    "Credit_Score": [750, 680, 720, 640, 800, 600, 710],
    "Income_Level": [50000, 40000, 55000, 30000, 80000, 20000, 45000],
    "Employment_Status": [1, 0, 1, 0, 1, 0, 1],
    "Default": [0, 1, 0, 1, 0, 1, 0],
}
df = pd.DataFrame(data)

X = df[["Credit_Score", "Income_Level", "Employment_Status"]]
y = df["Default"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

coefficients = dict(zip(X.columns, model.coef_[0]))
intercept = model.intercept_[0]

y_pred = model.predict(X_test)
classification_metrics = classification_report(y_test, y_pred, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred).tolist()

sample_applicant = np.array([[700, 45000, 1]])
probability_default = model.predict_proba(sample_applicant)[0, 1]

{
    "Intercept": intercept,
    "Coefficients": coefficients,
    "Classification_Report": classification_metrics,
    "Confusion_Matrix": conf_matrix,
    "Sample_Applicant_Probability_Default": round(probability_default, 2),
}
