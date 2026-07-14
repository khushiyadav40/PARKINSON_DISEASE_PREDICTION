import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

print("Loading dataset...")
df = pd.read_csv("data/parkinsons.csv")
print(f"Dataset shape: {df.shape}")


X = df.drop(["status", "name"], axis=1)
y = df["status"]

print("Class balance:")
print(y.value_counts())


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=42),
    "SVM": SVC(class_weight="balanced", random_state=42),
}

results = {}

def evaluate(y_test, y_pred, model_name):
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n--- {model_name} ---")
    print("Accuracy:", round(acc, 4))
    print("Precision:", round(prec, 4))
    print("Recall:", round(rec, 4))
    print("F1 Score:", round(f1, 4))
    print("Confusion Matrix:\n", cm)

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

print("\nTraining and evaluating models...")
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    results[name] = evaluate(y_test, y_pred, name)



best_model_name = max(results, key=lambda name: results[name]["f1"])
print(f"\nBest model based on F1 score: {best_model_name}")


if best_model_name == "Random Forest":
    param_grid = {"n_estimators": [50, 100, 150], "max_depth": [None, 5, 10]}
    grid_model = RandomForestClassifier(class_weight="balanced", random_state=42)
elif best_model_name == "Decision Tree":
    param_grid = {"max_depth": [None, 3, 5, 7, 10], "min_samples_split": [2, 5, 10]}
    grid_model = DecisionTreeClassifier(class_weight="balanced", random_state=42)
elif best_model_name == "SVM":
    param_grid = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
    grid_model = SVC(class_weight="balanced", random_state=42)
else:  
    param_grid = {"C": [0.01, 0.1, 1, 10]}
    grid_model = LogisticRegression(class_weight="balanced", max_iter=1000)

print(f"\nTuning {best_model_name} with GridSearchCV...")
grid_search = GridSearchCV(grid_model, param_grid, cv=5, scoring="f1")
grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_
print("Best parameters found:", grid_search.best_params_)


y_pred_final = best_model.predict(X_test_scaled)
evaluate(y_test, y_pred_final, f"Tuned {best_model_name}")


joblib.dump(best_model, "models/parkinsons_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
