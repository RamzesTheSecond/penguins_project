import os
import yaml
import joblib
import optuna
import mlflow
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

os.makedirs("models", exist_ok=True)

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

n_trials = params["train"]["n_trials"]
cv_folds = params["train"]["cv_folds"]
random_state = params["train"]["random_state"]

train_df = pd.read_csv("data/train.csv")

target_col = "species"
X = train_df.drop(columns=[target_col])
y = train_df[target_col]

categorical_cols = ["island", "sex"]
numerical_cols = ["culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g"]

mlflow.set_experiment("penguins-optuna")

def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 2, 20)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numerical_cols),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=cv_folds,
        scoring="f1_weighted",
        n_jobs=-1
    )

    mean_score = scores.mean()

    with mlflow.start_run(nested=True):
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
        })
        mlflow.log_metric("cv_f1_weighted", mean_score)

    return mean_score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=n_trials)

best_params = study.best_params

final_preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numerical_cols),
    ]
)

final_model = RandomForestClassifier(
    **best_params,
    random_state=random_state,
    n_jobs=-1,
)

final_pipeline = Pipeline([
    ("preprocessor", final_preprocessor),
    ("model", final_model)
])

final_pipeline.fit(X, y)

joblib.dump(final_pipeline, "models/model.pkl")
joblib.dump(best_params, "models/best_params.pkl")

print("Saved models/model.pkl and models/best_params.pkl")
print(best_params)