import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

def get_project_root():
    """Get the project root directory"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(current_dir))

class WasteManagementPredictor:
    def __init__(self):
        self.model = None
        self.encoders = {}
        self.scaler = None
        self.feature_names = None
        self.metrics = None
        
    def load_model(self, model_path=None):
        """Load a trained model and encoders"""
        if model_path is None:
            project_root = get_project_root()
            model_path = os.path.join(project_root, 'models', 'trained_model.pkl')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.encoders = model_data['encoders']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.metrics = model_data.get('metrics', {})
        print(f"Model loaded from {model_path}")
        
    def prepare_features(self, df):
        """Prepare features for prediction"""
        df_processed = df.copy()
        
        # Encode categorical variables
        categorical_cols = ['City/District', 'Waste Type', 'Disposal Method', 'Landfill Name']
        
        for col in categorical_cols:
            if col in df_processed.columns:
                if col in self.encoders:
                    # Handle unseen categories during prediction
                    df_processed[col] = df_processed[col].map(
                        lambda x: self.encoders[col].transform([x])[0] 
                        if x in self.encoders[col].classes_ else -1
                    )
                else:
                    # If encoder doesn't exist, create a dummy encoded value
                    df_processed[col] = 0
        
        # Select features for prediction
        feature_cols = [
            'City/District', 'Waste Type', 'Waste Generated (Tons/Day)',
            'Population Density (People/km²)', 'Municipal Efficiency Score (1-10)',
            'Disposal Method', 'Cost of Waste Management (₹/Ton)', 
            'Awareness Campaigns Count', 'Landfill Name',
            'Landfill Location (Lat)', 'Landfill Location (Long)',
            'Landfill Capacity (Tons)', 'Year'
        ]
        
        # Only use columns that exist in the dataframe
        available_cols = [col for col in feature_cols if col in df_processed.columns]
        X = df_processed[available_cols]
        
        return X
        
    def predict(self, input_data):
        """Make prediction on new data"""
        if self.model is None:
            raise ValueError("Model not loaded. Please call load_model() first.")
            
        if isinstance(input_data, dict):
            # Convert single prediction to DataFrame
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        
        # Prepare features
        X = self.prepare_features(df)
        
        # Ensure columns match training data
        if self.feature_names:
            # Reorder and fill missing columns
            for col in self.feature_names:
                if col not in X.columns:
                    X[col] = 0  # Default value for missing features
            X = X[self.feature_names]
        
        # Scale features
        if self.scaler is None:
            raise ValueError("Scaler not loaded. Please call load_model() first.")
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = self.model.predict(X_scaled)
        
        return prediction[0] if len(prediction) == 1 else prediction
    
    def get_model_metrics(self):
        """Get model performance metrics"""
        return self.metrics
    
    def is_model_loaded(self):
        """Check if model is loaded"""
        return self.model is not None

# Global predictor instance
predictor = WasteManagementPredictor()

def load_predictor():
    """Load the global predictor instance"""
    try:
        predictor.load_model()
        return True
    except Exception as e:
        print(f"Failed to load model: {e}")
        return False

def make_prediction(input_data):
    """Make a prediction using the global predictor"""
    if not predictor.is_model_loaded():
        raise ValueError("Model not loaded")
    return predictor.predict(input_data)

if __name__ == "__main__":
    # Test prediction
    predictor = WasteManagementPredictor()
    predictor.load_model()
    
    # Test prediction with sample data
    sample_input = {
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
    
    prediction = predictor.predict(sample_input)
    print(f"\nSample Prediction:")
    print(f"Input: {sample_input['City/District']}, {sample_input['Waste Type']}")
    print(f"Predicted Recycling Rate: {prediction:.2f}%")