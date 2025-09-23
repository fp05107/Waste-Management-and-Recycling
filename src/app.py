# src/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
from typing import List, Optional
import os
from datetime import datetime
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse
from fastapi import Request

# Import preprocessing function
from src.data.preprocess import preprocess_data

app = FastAPI(
    title="Waste Management Recycling Rate Predictor API",
    description="A machine learning API to predict recycling rates for Indian cities based on waste management parameters",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add this after creating the FastAPI app, before your routes
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load model and preprocessor
try:
    model = joblib.load('models/final_model.pkl')
    # model = joblib.load('./final_model.pkl')
    preprocessor = joblib.load('models/preprocessor.pkl')

    print("✅ Model and preprocessor loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    preprocessor = None

# Pydantic models for request/response validation
class WasteData(BaseModel):
    City_District: str = Field(..., description="Name of the city or district")
    Waste_Type: str = Field(..., description="Type of waste (Plastic, Organic, E-Waste, Construction, Hazardous)")
    Waste_Generated_Tons_Day: float = Field(..., ge=0, description="Waste generated in tons per day")
    Population_Density_People_km2: float = Field(..., ge=0, description="Population density in people per km²")
    Municipal_Efficiency_Score: int = Field(..., ge=1, le=10, description="Municipal efficiency score (1-10)")
    Cost_of_Waste_Management_Rs_Ton: float = Field(..., ge=0, description="Cost of waste management in ₹ per ton")
    Awareness_Campaigns_Count: int = Field(..., ge=0, description="Number of awareness campaigns")
    Landfill_Name: str = Field(..., description="Name of the landfill site")
    Landfill_Location_Lat_Long: str = Field(..., description="Landfill coordinates as 'lat, long'")
    Landfill_Capacity_Tons: float = Field(..., ge=0, description="Landfill capacity in tons")
    Year: int = Field(..., ge=2019, le=2023, description="Year of data (2019-2023)")

    class Config:
        schema_extra = {
            "example": {
                "City_District": "Mumbai",
                "Waste_Type": "Plastic",
                "Waste_Generated_Tons_Day": 2500.5,
                "Population_Density_People_km2": 11191.0,
                "Municipal_Efficiency_Score": 7,
                "Cost_of_Waste_Management_Rs_Ton": 3056.0,
                "Awareness_Campaigns_Count": 14,
                "Landfill_Name": "Mumbai Landfill",
                "Landfill_Location_Lat_Long": "19.0760, 72.8777",
                "Landfill_Capacity_Tons": 45575.0,
                "Year": 2023
            }
        }

class PredictionResponse(BaseModel):
    prediction_id: str
    timestamp: str
    predicted_recycling_rate: float
    confidence_interval: dict
    model_version: str

class BatchPredictionRequest(BaseModel):
    records: List[WasteData]

class BatchPredictionResponse(BaseModel):
    predictions: List[dict]
    total_records: int
    average_recycling_rate: float

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Waste Management Recycling Rate Prediction API",
        "version": "1.0.0",
        "description": "Predict recycling rates for Indian cities",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/predict/batch"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = "healthy" if model and preprocessor else "unhealthy"
    return {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(data: WasteData):
    """
    Predict recycling rate for a single data point
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        input_dict = data.dict()
        
        # Convert to match training data column names
        formatted_dict = {
            'City/District': input_dict['City_District'],
            'Waste Type': input_dict['Waste_Type'],
            'Waste Generated (Tons/Day)': input_dict['Waste_Generated_Tons_Day'],
            'Population Density (People/km²)': input_dict['Population_Density_People_km2'],
            'Municipal Efficiency Score (1-10)': input_dict['Municipal_Efficiency_Score'],
            'Cost of Waste Management (₹/Ton)': input_dict['Cost_of_Waste_Management_Rs_Ton'],
            'Awareness Campaigns Count': input_dict['Awareness_Campaigns_Count'],
            'Landfill Name': input_dict['Landfill_Name'],
            'Landfill Location (Lat, Long)': input_dict['Landfill_Location_Lat_Long'],
            'Landfill Capacity (Tons)': input_dict['Landfill_Capacity_Tons'],
            'Year': input_dict['Year']
        }
        
        input_df = pd.DataFrame([formatted_dict])
        
        # Preprocess the input
        X_processed, _, _ = preprocess_data(input_df, is_training=False, preprocessor=preprocessor)
        
        # Make prediction
        prediction = model.predict(X_processed)[0]
        
        # Calculate confidence interval (simplified based on RMSE)
        confidence_margin = 2.71  # Based on your RMSE of 2.71
        lower_bound = max(0, prediction - confidence_margin)
        upper_bound = min(100, prediction + confidence_margin)
        
        return PredictionResponse(
            prediction_id=f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            predicted_recycling_rate=round(prediction, 2),
            confidence_interval={
                "lower_bound": round(lower_bound, 2),
                "upper_bound": round(upper_bound, 2)
            },
            model_version="1.0.0"
        )
        
    except Exception as e:
        print(f"------------{e}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict recycling rates for multiple data points
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        records = []
        for item in request.records:
            input_dict = item.dict()
            formatted_dict = {
                'City/District': input_dict['City_District'],
                'Waste Type': input_dict['Waste_Type'],
                'Waste Generated (Tons/Day)': input_dict['Waste_Generated_Tons_Day'],
                'Population Density (People/km²)': input_dict['Population_Density_People_km2'],
                'Municipal Efficiency Score (1-10)': input_dict['Municipal_Efficiency_Score'],
                'Cost of Waste Management (₹/Ton)': input_dict['Cost_of_Waste_Management_Rs_Ton'],
                'Awareness Campaigns Count': input_dict['Awareness_Campaigns_Count'],
                'Landfill Name': input_dict['Landfill_Name'],
                'Landfill Location (Lat, Long)': input_dict['Landfill_Location_Lat_Long'],
                'Landfill Capacity (Tons)': input_dict['Landfill_Capacity_Tons'],
                'Year': input_dict['Year']
            }
            records.append(formatted_dict)
        
        input_df = pd.DataFrame(records)
        
        # Preprocess the input
        X_processed, _, _ = preprocess_data(input_df, is_training=False, preprocessor=preprocessor)
        
        # Make predictions
        predictions = model.predict(X_processed)
        
        # Prepare response
        results = []
        for i, (record, pred) in enumerate(zip(request.records, predictions)):
            results.append({
                "record_id": i + 1,
                "city": record.City_District,
                "waste_type": record.Waste_Type,
                "predicted_recycling_rate": round(pred, 2),
                "year": record.Year
            })
        
        avg_rate = np.mean(predictions)
        
        return BatchPredictionResponse(
            predictions=results,
            total_records=len(results),
            average_recycling_rate=round(avg_rate, 2)
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Get information about the trained model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    model_type = type(model).__name__
    model_params = model.get_params() if hasattr(model, 'get_params') else {}
    
    return {
        "model_type": model_type,
        "model_parameters": model_params,
        "training_metrics": {
            "test_rmse": 2.71,
            "test_r2": 0.973,
            "cv_mean_rmse": 2.94
        },
        "feature_count": model.n_features_in_ if hasattr(model, 'n_features_in_') else "Unknown"
    }

# Add these routes to your existing app
@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    """Serve the main UI"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/ui", response_class=HTMLResponse)
async def serve_ui_alt(request: Request):
    """Alternative UI endpoint"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health-ui")
async def health_ui():
    """Simple health check page"""
    return {"status": "healthy", "message": "Waste Management API is running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)