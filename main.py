from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import joblib
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model dan metadata
model = joblib.load("car_price_model_cpu_optimized.pkl")
with open("metadata.json", "r") as f:
    metadata = json.load(f)

class CarData(BaseModel):
    Brand: str
    Year: float
    Kms_Driven: float
    Horsepower: float
    # Pastikan semua kolom input sesuai dengan yang ada di X_train saat training
    Model: Optional[str] = None
    Mileage_kmpl: Optional[float] = None
    Engine_CC: Optional[float] = None
    Fuel_Type: Optional[str] = None
    Transmission: Optional[str] = None
    Owner_Type: Optional[str] = None
    Color: Optional[str] = None
    City: Optional[str] = None

@app.post("/api/predict")
def predict_price(data: CarData):
    input_dict = data.dict()
    
    # Imputasi sederhana jika ada data kosong
    for col in input_dict:
        if input_dict[col] is None:
            input_dict[col] = metadata["means"].get(col, 0)
    
    input_df = pd.DataFrame([input_dict])
    
    # Model memprediksi dalam USD (langsung dari model)
    prediction = model.predict(input_df)[0]
    
    # Kirim angka mentah ke frontend
    return {"predicted_price": float(prediction)}
