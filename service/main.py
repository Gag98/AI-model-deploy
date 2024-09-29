import os
import joblib
from fastapi import FastAPI


app = FastAPI()

model_path = os.path.join("weights", "rf_weights.joblib")
model = joblib.load(model_path)

@app.post("/coords-classification/")
def create_upload_file(inpt: dict):
    result = model.predict(inpt["coords"]).item()
    response = {"label": result}
    return response
