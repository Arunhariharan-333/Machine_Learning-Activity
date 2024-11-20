import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = {
    "Symptom1": [1, 0, 1, 1, 0, 0, 1],
    "Symptom2": [2, 1, 3, 2, 1, 0, 3],
    "Age": [25, 34, 23, 45, 50, 36, 29],
    "History": [1, 0, 1, 0, 1, 0, 1],
    "Disease": [1, 0, 1, 1, 0, 0, 1],
}
df = pd.DataFrame(data)

X = df[["Symptom1", "Symptom2", "Age", "History"]]
y = df["Disease"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

k_values = range(1, 21)
accuracies = [
    cross_val_score(KNeighborsClassifier(n_neighbors=k), X_train, y_train, cv=5).mean()
    for k in k_values
]

plt.plot(k_values, accuracies, marker='o')
plt.title("Accuracy vs. K Value")
plt.xlabel("K")
plt.ylabel("Cross-Validated Accuracy")
plt.xticks(k_values)
plt.grid()
plt.show()

optimal_k = k_values[accuracies.index(max(accuracies))]

final_knn = KNeighborsClassifier(n_neighbors=optimal_k)
final_knn.fit(X_train, y_train)
accuracy = final_knn.score(X_test, y_test)

{"Optimal_K": optimal_k, "Test_Set_Accuracy": round(accuracy, 2)}
