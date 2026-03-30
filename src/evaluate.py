import json
import joblib
import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

test_df = pd.read_csv("data/test.csv")
model = joblib.load("models/model.pkl")
best_params = joblib.load("models/best_params.pkl")

target_col = "species"
X_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col]

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

metrics = {
    "accuracy": float(accuracy),
    "f1_score": float(f1)
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

mlflow.set_experiment("penguins-optuna")
with mlflow.start_run(run_name="best_model_evaluation"):
    mlflow.log_params(best_params)
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_f1_weighted", f1)
    mlflow.sklearn.log_model(model, artifact_path="model")

print(metrics)
print("Saved metrics.json")