# src/utils/helpers.py
import pandas as pd
import joblib
import os
from pathlib import Path
from src.data.preprocess import preprocess_data, prepare_inference_data

def load_artifacts():
    """Load model and preprocessor artifacts with proper path handling"""
    try:
        # Try different possible paths for model
        model_paths = [
            'models/final_model.pkl',
            '../models/final_model.pkl',
            './models/final_model.pkl'
        ]
        
        # Try different possible paths for preprocessor
        preprocessor_paths = [
            'models/preprocessor.pkl',
            '../models/preprocessor.pkl', 
            './models/preprocessor.pkl'
        ]
        
        model = None
        preprocessor = None
        
        # Find and load model
        for path in model_paths:
            if os.path.exists(path):
                model = joblib.load(path)
                print(f"✅ Model loaded from: {path}")
                break
        
        # Find and load preprocessor
        for path in preprocessor_paths:
            if os.path.exists(path):
                preprocessor = joblib.load(path)
                print(f"✅ Preprocessor loaded from: {path}")
                break
        
        if model is None:
            raise FileNotFoundError("Model file not found")
        if preprocessor is None:
            raise FileNotFoundError("Preprocessor file not found")
            
        return model, preprocessor
        
    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")
        raise

def predict_single(model, preprocessor, input_data):
    """Make prediction for a single input"""
    try:
        # Prepare inference data
        inference_df = prepare_inference_data(input_data)
        
        # Preprocess - pass the loaded preprocessor
        X_processed, _, _ = preprocess_data(
            inference_df, 
            is_training=False, 
            preprocessor=preprocessor  # Pass the loaded preprocessor
        )
        
        # Predict
        prediction = model.predict(X_processed)[0]
        # Ensure prediction is within reasonable bounds
        return max(0, min(100, round(prediction, 2)))
        
    except Exception as e:
        raise Exception(f"Prediction error: {e}")