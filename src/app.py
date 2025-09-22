import os
import sys
from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional

# Add src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import from our modules
from models.predict import WasteManagementPredictor
from utils.helpers import validate_input_data, format_prediction_result, get_project_root

# Initialize FastAPI app
app = FastAPI(title="Waste Management Recycling Rate Predictor", version="1.0.0")

# Get project root directory
project_root = get_project_root()

# Setup templates and static files
templates_dir = os.path.join(project_root, "templates")
templates = Jinja2Templates(directory=templates_dir)

# Setup static files if they exist
static_dir = os.path.join(project_root, "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Initialize model
predictor = WasteManagementPredictor()
model_ready = False

# Load trained model on startup
@app.on_event("startup")
async def load_model():
    global model_ready
    model_path = os.path.join(project_root, "models", "trained_model.pkl")
    
    if os.path.exists(model_path):
        try:
            predictor.load_model(model_path)
            model_ready = True
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Failed to load model: {e}")
            model_ready = False
    else:
        print(f"Model file not found at {model_path}. Please train the model first.")
        model_ready = False

# Pydantic model for API requests
class PredictionRequest(BaseModel):
    city_district: str
    waste_type: str
    waste_generated: float
    population_density: float
    municipal_efficiency: int
    disposal_method: str
    cost_management: float
    awareness_campaigns: int
    landfill_name: str
    landfill_lat: float
    landfill_long: float
    landfill_capacity: float
    year: int

class PredictionResponse(BaseModel):
    recycling_rate: float
    city_district: str
    waste_type: str

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main HTML interface"""
    return templates.TemplateResponse("index.html", {"request": request})

def prepare_input_data(city_district: str, waste_type: str, waste_generated: float,
                      population_density: float, municipal_efficiency: int, 
                      disposal_method: str, cost_management: float,
                      awareness_campaigns: int, landfill_name: str,
                      landfill_lat: float, landfill_long: float,
                      landfill_capacity: float, year: int) -> dict:
    """Centralized function to prepare input data for prediction"""
    input_data = {
        'City/District': city_district,
        'Waste Type': waste_type,
        'Waste Generated (Tons/Day)': waste_generated,
        'Population Density (People/km²)': population_density,
        'Municipal Efficiency Score (1-10)': municipal_efficiency,
        'Disposal Method': disposal_method,
        'Cost of Waste Management (₹/Ton)': cost_management,
        'Awareness Campaigns Count': awareness_campaigns,
        'Landfill Name': landfill_name,
        'Landfill Location (Lat)': landfill_lat,
        'Landfill Location (Long)': landfill_long,
        'Landfill Capacity (Tons)': landfill_capacity,
        'Year': year
    }
    
    # Validate input data
    try:
        validated_data = validate_input_data(input_data)
        return validated_data
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict", response_model=PredictionResponse)
async def predict_recycling_rate(request: PredictionRequest):
    """API endpoint for recycling rate prediction"""
    
    if not model_ready:
        raise HTTPException(status_code=503, detail="Model is not ready. Please ensure the model is trained and loaded.")
    
    # Prepare input data
    input_data = prepare_input_data(
        request.city_district, request.waste_type, request.waste_generated,
        request.population_density, request.municipal_efficiency,
        request.disposal_method, request.cost_management,
        request.awareness_campaigns, request.landfill_name,
        request.landfill_lat, request.landfill_long,
        request.landfill_capacity, request.year
    )
    
    # Make prediction
    try:
        prediction = predictor.predict(input_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    return PredictionResponse(
        recycling_rate=round(float(prediction), 2),
        city_district=request.city_district,
        waste_type=request.waste_type
    )

@app.post("/predict-form")
async def predict_form(
    request: Request,
    city_district: str = Form(...),
    waste_type: str = Form(...),
    waste_generated: float = Form(...),
    population_density: float = Form(...),
    municipal_efficiency: int = Form(...),
    disposal_method: str = Form(...),
    cost_management: float = Form(...),
    awareness_campaigns: int = Form(...),
    landfill_name: str = Form(...),
    landfill_lat: float = Form(...),
    landfill_long: float = Form(...),
    landfill_capacity: float = Form(...),
    year: int = Form(...)
):
    """Form-based prediction endpoint"""
    
    if not model_ready:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error_message": "Model is not ready. Please ensure the model is trained and loaded."
        })
    
    # Prepare input data
    try:
        input_data = prepare_input_data(
            city_district, waste_type, waste_generated,
            population_density, municipal_efficiency,
            disposal_method, cost_management,
            awareness_campaigns, landfill_name,
            landfill_lat, landfill_long,
            landfill_capacity, year
        )
    except HTTPException as e:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error_message": e.detail
        })
    
    # Make prediction
    try:
        prediction = predictor.predict(input_data)
        prediction_result = round(float(prediction), 2)
    except Exception as e:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error_message": f"Prediction failed: {str(e)}"
        })
    
    return templates.TemplateResponse("result.html", {
        "request": request,
        "prediction": prediction_result,
        "city_district": city_district,
        "waste_type": waste_type,
        "input_data": input_data
    })

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    metrics = predictor.get_model_metrics() if model_ready else {}
    return {
        "status": "healthy", 
        "model_loaded": model_ready,
        "model_metrics": metrics
    }

@app.get("/model-info")
async def model_info():
    """Get model information and performance metrics"""
    if not model_ready:
        raise HTTPException(status_code=503, detail="Model is not ready")
    
    metrics = predictor.get_model_metrics()
    return {
        "model_loaded": True,
        "model_type": "Random Forest Regressor",
        "target_variable": "Recycling Rate (%)",
        "metrics": metrics
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)