# src/utils/helpers.py
import pandas as pd
import joblib
import os
from pathlib import Path
import numpy as np
from typing import Dict, Any, List

def load_artifacts():
    """
    Load the trained model and preprocessor from disk
    
    Returns:
        tuple: (model, preprocessor) if successful
        
    Raises:
        Exception: If artifacts cannot be loaded
    """
    try:
        # Get the project root directory (one level up from src)
        project_root = Path(__file__).parent.parent
        
        # Define paths to model files
        model_path = project_root / 'models' / 'final_model.pkl'
        preprocessor_path = project_root / 'models' / 'preprocessor.pkl'
        
        print(f"Looking for model at: {model_path}")
        print(f"Looking for preprocessor at: {preprocessor_path}")
        
        # Check if files exist
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor file not found at: {preprocessor_path}")
        
        # Load the artifacts
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        
        print("✅ Model and preprocessor loaded successfully!")
        return model, preprocessor
        
    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")
        raise

def prepare_api_input(api_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert API input data to the format expected by the model
    
    Args:
        api_data: Dictionary from API request
        
    Returns:
        pd.DataFrame: Data formatted for model prediction
    """
    # Map API field names to dataset column names
    column_mapping = {
        'City_District': 'City/District',
        'Waste_Type': 'Waste Type',
        'Waste_Generated_Tons_Day': 'Waste Generated (Tons/Day)',
        'Population_Density_People_km2': 'Population Density (People/km²)',
        'Municipal_Efficiency_Score': 'Municipal Efficiency Score (1-10)',
        'Cost_of_Waste_Management_Rs_Ton': 'Cost of Waste Management (₹/Ton)',
        'Awareness_Campaigns_Count': 'Awareness Campaigns Count',
        'Landfill_Name': 'Landfill Name',
        'Landfill_Location_Lat_Long': 'Landfill Location (Lat, Long)',
        'Landfill_Capacity_Tons': 'Landfill Capacity (Tons)',
        'Year': 'Year'
    }
    
    # Create DataFrame with correct column names
    formatted_data = {}
    for api_key, dataset_key in column_mapping.items():
        if api_key in api_data:
            formatted_data[dataset_key] = api_data[api_key]
    
    return pd.DataFrame([formatted_data])

def predict_single(model, preprocessor, input_data: Dict[str, Any]) -> float:
    """
    Make a prediction for a single input
    
    Args:
        model: Trained model
        preprocessor: Fitted preprocessor
        input_data: Input data dictionary
        
    Returns:
        float: Predicted recycling rate
        
    Raises:
        Exception: If prediction fails
    """
    try:
        # Prepare the input data
        inference_df = prepare_api_input(input_data)
        
        # Use the preprocess_data function from the imported module
        from src.data.preprocess import preprocess_data
        
        # Preprocess the data using the fitted preprocessor
        X_processed, _, _ = preprocess_data(
            inference_df, 
            is_training=False, 
            preprocessor=preprocessor
        )
        
        # Make prediction
        prediction = model.predict(X_processed)[0]
        
        # Ensure prediction is within reasonable bounds (0-100%)
        prediction = max(0, min(100, prediction))
        
        return round(prediction, 2)
        
    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")

def predict_batch(model, preprocessor, batch_data: List[Dict[str, Any]]) -> List[float]:
    """
    Make predictions for multiple inputs
    
    Args:
        model: Trained model
        preprocessor: Fitted preprocessor
        batch_data: List of input data dictionaries
        
    Returns:
        List[float]: List of predicted recycling rates
    """
    try:
        predictions = []
        for input_data in batch_data:
            prediction = predict_single(model, preprocessor, input_data)
            predictions.append(prediction)
        
        return predictions
        
    except Exception as e:
        raise Exception(f"Batch prediction failed: {str(e)}")

def calculate_confidence_interval(prediction: float, rmse: float = 2.71) -> Dict[str, float]:
    """
    Calculate confidence interval for a prediction
    
    Args:
        prediction: The predicted value
        rmse: Root Mean Square Error from model training
        
    Returns:
        Dict: Lower and upper bounds of confidence interval
    """
    margin = rmse * 1.96  # 95% confidence interval
    
    lower_bound = max(0, prediction - margin)
    upper_bound = min(100, prediction + margin)
    
    return {
        "lower_bound": round(lower_bound, 2),
        "upper_bound": round(upper_bound, 2)
    }

def get_model_info(model) -> Dict[str, Any]:
    """
    Extract information about the trained model
    
    Args:
        model: Trained model
        
    Returns:
        Dict: Model information
    """
    model_type = type(model).__name__
    
    info = {
        "model_type": model_type,
        "model_version": "1.0.0"
    }
    
    # Add model-specific information
    if hasattr(model, 'feature_importances_'):
        info["has_feature_importances"] = True
        info["num_features"] = len(model.feature_importances_)
    
    if hasattr(model, 'get_params'):
        info["parameters"] = model.get_params()
    
    return info

def validate_input_data(input_data: Dict[str, Any]) -> bool:
    """
    Validate the input data from API request
    
    Args:
        input_data: Data to validate
        
    Returns:
        bool: True if data is valid
    """
    required_fields = [
        'City_District', 'Waste_Type', 'Waste_Generated_Tons_Day',
        'Population_Density_People_km2', 'Municipal_Efficiency_Score',
        'Cost_of_Waste_Management_Rs_Ton', 'Awareness_Campaigns_Count',
        'Landfill_Name', 'Landfill_Location_Lat_Long', 'Landfill_Capacity_Tons', 'Year'
    ]
    
    # Check if all required fields are present
    for field in required_fields:
        if field not in input_data:
            return False
    
    # Validate numerical fields
    try:
        float(input_data['Waste_Generated_Tons_Day'])
        float(input_data['Population_Density_People_km2'])
        int(input_data['Municipal_Efficiency_Score'])
        float(input_data['Cost_of_Waste_Management_Rs_Ton'])
        int(input_data['Awareness_Campaigns_Count'])
        float(input_data['Landfill_Capacity_Tons'])
        int(input_data['Year'])
    except (ValueError, TypeError):
        return False
    
    return True

# Test function
def test_helpers():
    """Test the helper functions"""
    try:
        # Test loading artifacts
        model, preprocessor = load_artifacts()
        print("✅ Artifacts loaded successfully")
        
        # Test input preparation
        test_data = {
            'City_District': 'Mumbai',
            'Waste_Type': 'Plastic',
            'Waste_Generated_Tons_Day': 2500.5,
            'Population_Density_People_km2': 11191.0,
            'Municipal_Efficiency_Score': 7,
            'Cost_of_Waste_Management_Rs_Ton': 3056.0,
            'Awareness_Campaigns_Count': 14,
            'Landfill_Name': 'Mumbai Landfill',
            'Landfill_Location_Lat_Long': '19.0760, 72.8777',
            'Landfill_Capacity_Tons': 45575.0,
            'Year': 2023
        }
        
        df = prepare_api_input(test_data)
        print(f"✅ Input preparation successful. DataFrame shape: {df.shape}")
        
        # Test validation
        is_valid = validate_input_data(test_data)
        print(f"✅ Input validation: {is_valid}")
        
        return True
        
    except Exception as e:
        print(f"❌ Helper test failed: {e}")
        return False

if __name__ == "__main__":
    test_helpers()