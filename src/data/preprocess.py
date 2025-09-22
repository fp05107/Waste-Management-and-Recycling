# src/data/preprocess.py
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from category_encoders import TargetEncoder
import joblib

# Custom transformer for geospatial feature engineering
class GeospatialEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.city_coords = {}  # Will store avg lat/lon for each city

    def fit(self, X, y=None):
        # Calculate average landfill location for each city (proxy for city center)
        if 'Landfill Location (Lat, Long)' in X.columns and 'City/District' in X.columns:
            coords_df = X[['City/District', 'Landfill Location (Lat, Long)']].copy()
            coords_df[['Lat', 'Lon']] = coords_df['Landfill Location (Lat, Long)'].str.split(', ', expand=True).astype(float)
            self.city_coords = coords_df.groupby('City/District')[['Lat', 'Lon']].mean().to_dict('index')
        return self

    def transform(self, X):
        X_copy = X.copy()
        # 1. Split the Landfill Location into two columns
        if 'Landfill Location (Lat, Long)' in X_copy.columns:
            X_copy[['Landfill_Lat', 'Landfill_Lon']] = X_copy['Landfill Location (Lat, Long)'].str.split(', ', expand=True).astype(float)

        # 2. Calculate distance from city center (proxy) to landfill
        if 'City/District' in X_copy.columns and self.city_coords:
            def calculate_distance(row):
                city = row['City/District']
                landfill_lat = row.get('Landfill_Lat')
                landfill_lon = row.get('Landfill_Lon')
                if city in self.city_coords and pd.notna(landfill_lat) and pd.notna(landfill_lon):
                    city_lat = self.city_coords[city]['Lat']
                    city_lon = self.city_coords[city]['Lon']
                    # Using geopy for accurate distance calculation
                    return geodesic((city_lat, city_lon), (landfill_lat, landfill_lon)).km
                return np.nan
            X_copy['Landfill_Distance_km'] = X_copy.apply(calculate_distance, axis=1)
        return X_copy

def preprocess_data(df, is_training=True, target_col='Recycling Rate (%)', preprocessor=None):
    """
    Main preprocessing function.
    is_training: True for training data, False for test/inference data.
    target_col: Name of the target variable.
    preprocessor: A fitted ColumnTransformer pipeline (used when is_training=False).
    """
    # Handle the target column
    if target_col in df.columns:
        y = df[target_col].copy()
        df = df.drop(columns=[target_col])
    else:
        y = None

    # --- FEATURE ENGINEERING & CLEANING ---
    # 1. Handle known data leakage: Drop 'Disposal Method'
    if 'Disposal Method' in df.columns:
        df = df.drop(columns=['Disposal Method'])

    # 2. Apply custom geospatial engineering
    geo_engineer = GeospatialEngineer()
    if is_training:
        df_processed = geo_engineer.fit_transform(df)
    else:
        df_processed = geo_engineer.transform(df)

    # Drop the original messy column
    if 'Landfill Location (Lat, Long)' in df_processed.columns:
        df_processed = df_processed.drop(columns=['Landfill Location (Lat, Long)'])

    # --- DEFINE FEATURE GROUPS ---
    # Identify numerical and categorical features
    numerical_features = df_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df_processed.select_dtypes(include=['object']).columns.tolist()

    # Remove target if it was mistakenly included
    if target_col in numerical_features:
        numerical_features.remove(target_col)

    # --- CREATE PREPROCESSING PIPELINE ---
    if is_training:
        # Define transformers for each type
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Use Target Encoding for high cardinality categories
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('target_encoder', TargetEncoder())  # Will be fitted with target y
        ])

        # Combine transformers
        preprocessor = ColumnTransformer(transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

        # Fit and transform the training data
        X_processed = preprocessor.fit_transform(df_processed, y)
        # Save the preprocessor for later use
        joblib.dump(preprocessor, '../models/preprocessor.pkl')
        joblib.dump(geo_engineer, '../models/geo_engineer.pkl') # Save geo engineer too

    else:
        # Load the pre-fitted preprocessor and transform
        if preprocessor is None:
            preprocessor = joblib.load('../models/preprocessor.pkl')
        X_processed = preprocessor.transform(df_processed)

    return X_processed, y, preprocessor if is_training else preprocessor

# Example usage (for testing the script):
if __name__ == "__main__":
    # Load data
    sample_df = pd.read_csv('../data/raw/Waste_Management_and_Recycling_India.csv')
    # Preprocess
    X_train, y_train, fitted_preprocessor = preprocess_data(sample_df, is_training=True)
    print(f"Processed training data shape: {X_train.shape}")