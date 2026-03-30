import joblib
import bentoml

model = joblib.load("models/model.pkl")
encoder = joblib.load("models/encoder.pkl")

bentoml.sklearn.save_model(
    "penguins_classifier",
    model,
    metadata={"task": "penguins_classification"},
)

bentoml.picklable_model.save_model(
    "penguins_encoder",
    encoder,
    metadata={"type": "onehot_encoder"},
)

print("Saved models to BentoML store")