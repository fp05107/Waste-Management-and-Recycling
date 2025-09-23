# src/data/preprocess.py
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from geopy.distance import geodesic

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

    # Cost efficiency
    # df_clean["Cost_Efficiency"] = df_clean['Recycling Rate (%)'] / df_clean['Cost of Waste Management (₹/Ton)']

    # Cost efficiency (only in training when target is available)
    if is_training and 'Recycling Rate (%)' in df_clean.columns:
        df_clean["Cost_Efficiency"] = (df_clean['Recycling Rate (%)'] / df_clean['Cost of Waste Management (₹/Ton)']
    )
    else:
        df_clean["Cost_Efficiency"] = np.nan  # placeholder for inference


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

    # Identify feature types
    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

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
        joblib.dump(preprocessor, '../models/preprocessor.pkl')
        return X_processed, y, preprocessor
    else:
        if preprocessor is None:
            # Load the pre-fitted preprocessor
            preprocessor = joblib.load('./models/preprocessor.pkl')
        X_processed = preprocessor.transform(X)
        return X_processed, y, preprocessor