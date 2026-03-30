import os
import openml

os.makedirs("data", exist_ok=True)

dataset = openml.datasets.get_dataset(42585)
X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

df = X.copy()
df["species"] = y

df.to_csv("data/penguins.csv", index=False)

print("Saved data/penguins.csv")