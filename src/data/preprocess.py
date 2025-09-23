# src/data/preprocess.py
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from geopy.distance import geodesic
import os
from pathlib import Path

def preprocess_data(df, is_training=True, target_col='Recycling Rate (%)', preprocessor=None):
    """
    Preprocess waste management data for modeling
    
    Parameters:
    - df: Input DataFrame
    - is_training: Whether this is training data
    - target_col: Name of the target column
    - preprocessor: Pre-fitted preprocessor (for inference)
    
    Returns:
    - X_processed: Processed features
    - y: Target variable (if available)
    - preprocessor: Fitted preprocessor (if training)
    """
    
    # Feature Engineering - Geospatial Features
    df_clean = df.copy()
    print(f"🚀 ~ df_clean:..... {df_clean}")
    
    # Parse Landfill Location
    df_clean[['Landfill_Lat', 'Landfill_Lon']] = df_clean['Landfill Location (Lat, Long)'] \
        .str.split(',', expand=True).astype(float)

    # Calculate average coordinates for each city
    city_coords = df_clean.groupby('City/District')[['Landfill_Lat', 'Landfill_Lon']].mean()

    # Calculate distance from city center to landfill
    def calculate_distance(row):
        city = row['City/District']
        landfill_lat = row['Landfill_Lat']
        landfill_lon = row['Landfill_Lon']
        
        if city in city_coords.index and pd.notna(landfill_lat) and pd.notna(landfill_lon):
            city_lat = city_coords.loc[city, 'Landfill_Lat']
            city_lon = city_coords.loc[city, 'Landfill_Lon']
            return geodesic((city_lat, city_lon), (landfill_lat, landfill_lon)).km
        return np.nan

    df_clean['Landfill_Distance_km'] = df_clean.apply(calculate_distance, axis=1)

    # Additional Feature Engineering
    # Waste generation per capita
    df_clean['Waste_Per_Capita'] = df_clean["Waste Generated (Tons/Day)"] / df_clean["Population Density (People/km²)"]
    
    df_clean['Cost_Efficiency'] = 0.022251	
    
    # Cost efficiency
    # if training_stats and 'cost_efficiency_mean' in training_stats:
    #     df['Cost_Efficiency'] = training_stats['cost_efficiency_mean']
    # else:
    #     # Fallback: calculate based on typical values
    #     typical_recycling_rate = 50
    #     df['Cost_Efficiency'] = typical_recycling_rate / df['Cost of Waste Management (₹/Ton)']
    

    # Year as Categorical
    df_clean['Year_Categorical'] = df_clean['Year'].astype('category')

    # Handle Data Leakage - Remove Problematic Features
    features_to_drop = ['Disposal Method', 'Landfill Location (Lat, Long)']
    df_final = df_clean.drop(columns=features_to_drop, errors="ignore")

    # Separate features and target
    if target_col in df_final.columns:
        y = df_final[target_col].copy()
        X = df_final.drop(columns=[target_col])
    else:
        y = None
        X = df_final

    # Remove Cost_Efficiency if it exists for inference
    # if not is_training and 'Cost_Efficiency' in X.columns:
    #     X = X.drop(columns=['Cost_Efficiency'])

    # Identify feature types
    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # THE KEY FIX: Only create new preprocessor if we're training or if none was provided
    if is_training or preprocessor is None:
        # Create preprocessing pipeline
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', TargetEncoder(random_state=42))
        ])

        preprocessor = ColumnTransformer([
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ])

    # Fit and transform the data
    if is_training:
        X_processed = preprocessor.fit_transform(X, y)
        # Save the preprocessor for future use
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        joblib.dump(preprocessor, models_dir / 'preprocessor.pkl')
        print("✅ Preprocessor fitted and saved successfully!")
        return X_processed, y, preprocessor
    else:
        # For inference, use the provided preprocessor (which should be already fitted)
        if preprocessor is None:
            # Try to load the preprocessor
            possible_paths = [
                'models/preprocessor.pkl',
                '../models/preprocessor.pkl',
                './models/preprocessor.pkl'
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    preprocessor = joblib.load(path)
                    print(f"✅ Preprocessor loaded from: {path}")
                    break
            
            if preprocessor is None:
                raise ValueError("No preprocessor provided and could not find saved preprocessor")
        
        # Ensure the preprocessor is fitted
        if not hasattr(preprocessor, 'transformers_'):
            raise ValueError("Provided preprocessor is not fitted. Call 'fit' first.")
            
        X_processed = preprocessor.transform(X)
        return X_processed, y, preprocessor


def prepare_inference_data(input_dict):
    """
    Prepare a single input dictionary for inference
    Converts API input format to model input format
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
        if api_key in input_dict:
            formatted_data[dataset_key] = input_dict[api_key]
    
    return pd.DataFrame([formatted_data])