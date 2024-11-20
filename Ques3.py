import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

data = {
    "Ad_Budget": [1000, 1500, 800, 2000, 500],
    "Target_Age": [25, 35, 20, 40, 30],
    "Medium": ["Online", "TV", "Online", "Print", "Online"],
    "Success": [1, 1, 0, 1, 0],
}

df = pd.DataFrame(data)

X = df[["Ad_Budget", "Target_Age", "Medium"]]
y = df["Success"]

preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(), ["Medium"])], remainder="passthrough"
)
X = preprocessor.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, output_dict=True)

cv_score = cross_val_score(tree, X, y, cv=5).mean()

results = {
    "Confusion_Matrix": conf_matrix.tolist(),
    "Classification_Report": class_report,
    "Cross_Validation_Score": cv_score,
}

results
