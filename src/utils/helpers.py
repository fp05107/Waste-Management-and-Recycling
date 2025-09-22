import os
import pandas as pd
import logging
from typing import Dict, Any, List, Optional

def get_project_root():
    """Get the project root directory"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(current_dir))

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_data(filename: str, data_type: str = "raw") -> pd.DataFrame:
    """Load data from the data directory"""
    project_root = get_project_root()
    data_path = os.path.join(project_root, 'data', data_type, filename)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
    
    return pd.read_csv(data_path)

def validate_input_data(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and clean input data for predictions"""
    required_fields = [
        'City/District', 'Waste Type', 'Waste Generated (Tons/Day)',
        'Population Density (People/km²)', 'Municipal Efficiency Score (1-10)',
        'Disposal Method', 'Cost of Waste Management (₹/Ton)', 
        'Awareness Campaigns Count', 'Landfill Name',
        'Landfill Location (Lat)', 'Landfill Location (Long)',
        'Landfill Capacity (Tons)', 'Year'
    ]
    
    # Check for missing fields
    missing_fields = [field for field in required_fields if field not in input_data]
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")
    
    # Validate data types and ranges
    validated_data = input_data.copy()
    
    # Validate municipal efficiency score
    efficiency = validated_data['Municipal Efficiency Score (1-10)']
    if not (1 <= efficiency <= 10):
        raise ValueError("Municipal Efficiency Score must be between 1 and 10")
    
    # Validate positive values
    positive_fields = [
        'Waste Generated (Tons/Day)', 'Population Density (People/km²)',
        'Cost of Waste Management (₹/Ton)', 'Landfill Capacity (Tons)'
    ]
    
    for field in positive_fields:
        if validated_data[field] < 0:
            raise ValueError(f"{field} must be positive")
    
    # Validate latitude and longitude ranges for India
    lat = validated_data['Landfill Location (Lat)']
    lon = validated_data['Landfill Location (Long)']
    
    if not (6 <= lat <= 38):
        raise ValueError("Latitude must be between 6 and 38 for Indian locations")
    
    if not (68 <= lon <= 98):
        raise ValueError("Longitude must be between 68 and 98 for Indian locations")
    
    return validated_data

def get_allowed_values() -> Dict[str, List[str]]:
    """Get allowed values for categorical fields"""
    return {
        'City/District': [
            'Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 
            'Pune', 'Ahmedabad', 'Surat', 'Jaipur', 'Lucknow', 'Kanpur', 
            'Nagpur', 'Indore', 'Thane', 'Bhopal', 'Visakhapatnam', 'Patna'
        ],
        'Waste Type': ['Plastic', 'Organic', 'E-Waste', 'Construction', 'Hazardous'],
        'Disposal Method': ['Landfill', 'Recycling', 'Incineration', 'Composting'],
        'Landfill Name': [
            'Central Landfill', 'East Zone Dump', 'West Side Facility', 
            'North Point Landfill', 'South City Dump', 'Metro Waste Site'
        ]
    }

def format_prediction_result(prediction: float, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Format prediction result with additional context"""
    result = {
        'recycling_rate': round(float(prediction), 2),
        'city_district': input_data['City/District'],
        'waste_type': input_data['Waste Type'],
        'prediction_confidence': 'High' if 20 <= prediction <= 80 else 'Medium',
        'recommendation': get_recommendation(prediction)
    }
    
    return result

def get_recommendation(recycling_rate: float) -> str:
    """Get recommendation based on recycling rate"""
    if recycling_rate >= 70:
        return "Excellent recycling performance. Maintain current practices and share best practices."
    elif recycling_rate >= 50:
        return "Moderate performance. Consider increasing awareness campaigns and improving municipal efficiency."
    else:
        return "Low performance. Urgent action needed to improve waste management systems and practices."

def create_directories():
    """Create necessary project directories"""
    project_root = get_project_root()
    directories = [
        'data/raw', 'data/processed', 'models', 'static', 'Notebooks'
    ]
    
    for directory in directories:
        dir_path = os.path.join(project_root, directory)
        os.makedirs(dir_path, exist_ok=True)
        
    print("Project directories created successfully")

if __name__ == "__main__":
    # Test helper functions
    logger = setup_logging()
    logger.info("Testing helper functions...")
    
    # Test validation
    test_data = {
        'City/District': 'Mumbai',
        'Waste Type': 'Plastic',
        'Waste Generated (Tons/Day)': 25.5,
        'Population Density (People/km²)': 15000.0,
        'Municipal Efficiency Score (1-10)': 7,
        'Disposal Method': 'Recycling',
        'Cost of Waste Management (₹/Ton)': 1200.0,
        'Awareness Campaigns Count': 5,
        'Landfill Name': 'Central Landfill',
        'Landfill Location (Lat)': 19.0760,
        'Landfill Location (Long)': 72.8777,
        'Landfill Capacity (Tons)': 100000.0,
        'Year': 2023
    }
    
    try:
        validated = validate_input_data(test_data)
        logger.info("Input validation passed")
        
        # Test prediction formatting
        result = format_prediction_result(65.5, validated)
        logger.info(f"Formatted result: {result}")
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")