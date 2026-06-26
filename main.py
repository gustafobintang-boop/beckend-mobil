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

print("[*] Memuat model dan metadata ringan...")
# 1. Muat Model
model = joblib.load("car_price_model_cpu.pkl")

# 2. Muat Metadata (Pengganti CSV)
with open("metadata.json", "r") as f:
    metadata = json.load(f)

unique_values = metadata["unique_values"]
means = metadata["means"]
modes = metadata["modes"]

class CarData(BaseModel):
    Brand: str
    Year: float
    Kms_Driven: float
    Horsepower: float
    Model: Optional[str] = None
    Mileage_kmpl: Optional[float] = None
    Engine_CC: Optional[float] = None
    Fuel_Type: Optional[str] = None
    Transmission: Optional[str] = None
    Owner_Type: Optional[str] = None
    Color: Optional[str] = None
    City: Optional[str] = None
    Insurance_Valid: Optional[float] = None
    Service_History: Optional[float] = None
    Accidents: Optional[float] = None
    Tax_Paid: Optional[float] = None
    Number_of_Doors: Optional[float] = None
    Seats: Optional[float] = None

@app.get("/api/form-options")
def get_options():
    return unique_values

@app.post("/api/predict")
def predict_price(data: CarData):
    input_dict = data.dict()
    
    # Imputasi otomatis jika user mengosongkan nilai opsional
    for col, val in input_dict.items():
        if val is None or val == "":
            if col in means:
                input_dict[col] = means[col]
            elif col in modes:
                input_dict[col] = modes[col]
                
    # Model XGBoost butuh format DataFrame 1 baris
    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)[0]
    
    return {"predicted_price": float(prediction)}