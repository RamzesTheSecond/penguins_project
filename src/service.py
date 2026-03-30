import pandas as pd
import bentoml
from pydantic import BaseModel

class PenguinFeatures(BaseModel):
    culmen_length_mm: float
    culmen_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    sex: str
    island: str

@bentoml.service
class PenguinsService:
    def __init__(self):
        self.model = bentoml.sklearn.load_model("penguins_classifier:latest")

    @bentoml.api
    def predict(self, input_data: PenguinFeatures) -> dict:
        df = pd.DataFrame([{
            "culmen_length_mm": input_data.culmen_length_mm,
            "culmen_depth_mm": input_data.culmen_depth_mm,
            "flipper_length_mm": input_data.flipper_length_mm,
            "body_mass_g": input_data.body_mass_g,
            "sex": input_data.sex,
            "island": input_data.island,
        }])

        prediction = self.model.predict(df)
        return {"prediction": str(prediction[0])}