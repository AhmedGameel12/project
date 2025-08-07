from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load the trained model
model = joblib.load("indoor_nav_model.joblib")

class RSSIInput(BaseModel):
    rssi: list[float]

@app.post("/predict")
def predict_location(data: RSSIInput):
    try:
        input_array = np.array([data.rssi], dtype=float)  # shape: (1, n_features)
        prediction = model.predict(input_array)  # shape: (1, 2)
        return {"x": float(prediction[0][0]), "y": float(prediction[0][1])}
    except Exception as e:
        return {"error": str(e)}