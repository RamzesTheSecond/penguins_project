import os
import yaml
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

test_size = params["prepare"]["test_size"]
random_state = params["prepare"]["random_state"]

df = pd.read_csv("data/penguins.csv")
df = df.dropna()

target_col = "species"
categorical_cols = ["island", "sex"]

train_df, test_df = train_test_split(
    df,
    test_size=test_size,
    random_state=random_state,
    stratify=df[target_col]
)

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoder.fit(train_df[categorical_cols])

train_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test.csv", index=False)
joblib.dump(encoder, "models/encoder.pkl")

print("Saved data/train.csv, data/test.csv, models/encoder.pkl")