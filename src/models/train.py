import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

def get_project_root():
    """Get the project root directory"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(current_dir))

def load_dataset():
    """Load the waste management dataset"""
    project_root = get_project_root()
    data_path = os.path.join(project_root, 'data', 'raw', 'waste_management_data.csv')
    return pd.read_csv(data_path)

def prepare_features(df, encoders=None, scaler=None):
    """Prepare features for training or prediction"""
    df_processed = df.copy()
    
    # Encode categorical variables
    categorical_cols = ['City/District', 'Waste Type', 'Disposal Method', 'Landfill Name']
    
    if encoders is None:
        encoders = {}
    
    for col in categorical_cols:
        if col in df_processed.columns:
            if col not in encoders:
                encoders[col] = LabelEncoder()
                df_processed[col] = encoders[col].fit_transform(df_processed[col])
            else:
                # Handle unseen categories during prediction
                df_processed[col] = df_processed[col].map(
                    lambda x: encoders[col].transform([x])[0] 
                    if x in encoders[col].classes_ else -1
                )
    
    # Select features for prediction (exclude target variable)
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
    
    # Scale features
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    return X_scaled, X.columns.tolist(), encoders, scaler

def train_model():
    """Train the waste management model"""
    print("Loading dataset...")
    df = load_dataset()
    
    # Prepare features
    print("Preparing features...")
    X_scaled, feature_names, encoders, scaler = prepare_features(df)
    y = df['Recycling Rate (%)']
    
    # Split data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Train model
    print("Training model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }
    
    print(f"Model Training Complete!")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"R² Score: {metrics['r2']:.3f}")
    
    # Save model
    print("Saving model...")
    project_root = get_project_root()
    model_path = os.path.join(project_root, 'models', 'trained_model.pkl')
    
    model_data = {
        'model': model,
        'encoders': encoders,
        'scaler': scaler,
        'feature_names': feature_names,
        'metrics': metrics
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model_data, model_path)
    print(f"Model saved to {model_path}")
    
    return model_data, metrics

def generate_predictions_csv():
    """Generate predictions for the test set and save as CSV"""
    print("Generating prediction CSV...")
    
    # Load dataset
    df = load_dataset()
    
    # Load trained model
    project_root = get_project_root()
    model_path = os.path.join(project_root, 'models', 'trained_model.pkl')
    
    if not os.path.exists(model_path):
        print("Model not found. Please train the model first.")
        return
    
    model_data = joblib.load(model_path)
    model = model_data['model']
    encoders = model_data['encoders']
    scaler = model_data['scaler']
    feature_names = model_data['feature_names']
    
    # Prepare features
    X_scaled, _, _, _ = prepare_features(df, encoders, scaler)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    
    # Create predictions dataframe
    predictions_df = df.copy()
    predictions_df['Predicted_Recycling_Rate'] = predictions
    predictions_df['Prediction_Error'] = predictions_df['Recycling Rate (%)'] - predictions
    
    # Save predictions
    predictions_path = os.path.join(project_root, 'predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")
    
    return predictions_path

if __name__ == "__main__":
    # Train the model
    model_data, metrics = train_model()
    
    # Generate predictions CSV
    generate_predictions_csv()
    
    print("\nTraining complete!")