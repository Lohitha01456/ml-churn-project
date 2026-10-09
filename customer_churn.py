import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


data = pd.read_csv("customer_churn.csv")

print("Missing values before cleaning:")
print(data.isnull().sum())

data["support_tickets"] = data["support_tickets"].fillna(
    data["support_tickets"].median()
)

data["weekly_usage_hours"] = data["weekly_usage_hours"].fillna(
    data["weekly_usage_hours"].median()
)

print("\nMissing values after cleaning:")
print(data.isnull().sum())



X = data[
    [
        "monthly_charge",
        "months_as_customer",
        "support_tickets",
        "weekly_usage_hours",
    ]
]

y = data["churned"]

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

print("\nTraining examples:", len(X_train))
print("Testing examples:", len(X_test))

model = DecisionTreeClassifier(
    max_depth=4,
    random_state=42,
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nPredictions:")
print(predictions)

results = X_test.copy()
results["actual"] = y_test.values
results["predicted"] = predictions

print("\nActual vs Predicted:")
print(results[["actual", "predicted"]])

accuracy = accuracy_score(y_test, predictions)

print("\nTest accuracy:", round(accuracy, 3))