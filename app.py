from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
# Import our preprocessing function from src
from src.data.preprocess import preprocess_data

app = FastAPI(title="Waste Management Recycling Rate Predictor", version="1.0")

# Load the trained model and preprocessor at startup
model = joblib.load("models/trained_model.pkl")
# Assume a preprocessor (e.g., a ColumnTransformer) was saved during training
preprocessor = joblib.load("models/preprocessor.pkl")

# Define the expected input schema using Pydantic
class PredictionInput(BaseModel):
    City_District: str
    Waste_Type: str
    Waste_Generated_Tons_Day: float
    Population_Density_People_km2: float
    Municipal_Efficiency_Score: int
    Disposal_Method: str
    Cost_of_Waste_Management_Rs_Ton: float
    Awareness_Campaigns_Count: int
    Landfill_Name: str
    Landfill_Location_Lat_Long: str
    Landfill_Capacity_Tons: float
    Year: int

    class Config:
        schema_extra = {
            "example": {
                "City_District": "Bangalore",
                "Waste_Type": "Plastic",
                "Waste_Generated_Tons_Day": 2500.5,
                "Population_Density_People_km2": 4500.0,
                "Municipal_Efficiency_Score": 7,
                "Disposal_Method": "Recycling",
                "Cost_of_Waste_Management_Rs_Ton": 2800.0,
                "Awareness_Campaigns_Count": 10,
                "Landfill_Name": "Bangalore Landfill",
                "Landfill_Location_Lat_Long": "12.9716, 77.5946",
                "Landfill_Capacity_Tons": 50000.0,
                "Year": 2023
            }
        }

@app.get("/")
async def root():
    return {"message": "Welcome to the Waste Management Prediction API"}

@app.post("/predict", response_class=JSONResponse)
async def predict(input: PredictionInput):
    """
    Predict recycling rate for a single data input.
    """
    try:
        # Convert Pydantic model to DataFrame
        input_dict = input.dict()
        # Fix key for the geospatial column to match training data
        input_dict['Landfill Location (Lat, Long)'] = input_dict.pop('Landfill_Location_Lat_Long')
        input_df = pd.DataFrame([input_dict])

        # Preprocess the input (using the same function from training)
        processed_input = preprocess_data(input_df, is_training=False, preprocessor=preprocessor)

        # Make prediction
        prediction = model.predict(processed_input)
        result = prediction[0]

        return {"predicted_recycling_rate_percent": round(result, 2)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_batch", response_class=JSONResponse)
async def predict_batch(file: UploadFile = File(...)):
    """
    Predict recycling rate from a uploaded CSV file.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a CSV file.")

    try:
        df = pd.read_csv(file.file)
        # Preprocess the batch
        processed_batch = preprocess_data(df, is_training=False, preprocessor=preprocessor)
        predictions = model.predict(processed_batch)
        df['Predicted_Recycling_Rate_Percent'] = predictions.round(2)
        # Return predictions as JSON
        return JSONResponse(content=df.to_dict(orient='records'))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

# You can run this with: `uvicorn src.app:app --reload`