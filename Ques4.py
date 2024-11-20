import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

data = {
    "Market_Trend": [1, 0, 1, 1, 0, 0, 1],
    "Economic_Indicator": [2.5, 1.2, 3.1, 2.8, 1.0, 0.8, 3.0],
    "Investment_Return": [1, 0, 1, 1, 0, 0, 1],
}
df = pd.DataFrame(data)

X = df[["Market_Trend", "Economic_Indicator"]]
y = df["Investment_Return"]

tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X, y)

plt.figure(figsize=(10, 6))
plot_tree(tree, feature_names=X.columns, class_names=["Failure", "Success"], filled=True)
plt.title("Decision Tree Visualization")
plt.show()

feature_importance = pd.Series(tree.feature_importances_, index=X.columns)
feature_importance.plot(kind="bar", color="skyblue", title="Feature Importance")
plt.ylabel("Importance")
plt.show()
