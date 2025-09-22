import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

def generate_waste_management_data(n_samples=1000):
    """Generate synthetic waste management dataset based on hackathon specifications"""
    
    # Define categorical options
    cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 
              'Pune', 'Ahmedabad', 'Surat', 'Jaipur', 'Lucknow', 'Kanpur', 
              'Nagpur', 'Indore', 'Thane', 'Bhopal', 'Visakhapatnam', 'Patna']
    
    waste_types = ['Plastic', 'Organic', 'E-Waste', 'Construction', 'Hazardous']
    
    disposal_methods = ['Landfill', 'Recycling', 'Incineration', 'Composting']
    
    landfill_names = ['Central Landfill', 'East Zone Dump', 'West Side Facility', 
                      'North Point Landfill', 'South City Dump', 'Metro Waste Site']
    
    # Generate random data
    np.random.seed(42)  # For reproducibility
    random.seed(42)
    
    data = []
    
    for i in range(n_samples):
        city = random.choice(cities)
        waste_type = random.choice(waste_types)
        
        # Generate correlated features
        population_density = np.random.uniform(1000, 25000)  # People/km²
        municipal_efficiency = np.random.randint(1, 11)  # 1-10 scale
        
        # Waste generation correlated with population density
        base_waste = population_density / 1000 * np.random.uniform(0.5, 2.0)
        waste_generated = max(1.0, base_waste + np.random.normal(0, 5))
        
        # Recycling rate influenced by municipal efficiency and waste type
        base_recycling = municipal_efficiency * 5  # Base on efficiency
        
        # Adjust for waste type
        if waste_type == 'Organic':
            base_recycling += np.random.uniform(10, 25)
        elif waste_type == 'Plastic':
            base_recycling += np.random.uniform(5, 20)
        elif waste_type == 'E-Waste':
            base_recycling += np.random.uniform(15, 30)
        elif waste_type == 'Construction':
            base_recycling += np.random.uniform(20, 40)
        elif waste_type == 'Hazardous':
            base_recycling += np.random.uniform(0, 10)
        
        # Add noise and ensure valid range
        recycling_rate = max(5.0, min(95.0, base_recycling + np.random.normal(0, 8)))
        
        # Cost correlated with waste type and efficiency
        base_cost = 1000 + (10 - municipal_efficiency) * 200
        if waste_type == 'Hazardous':
            base_cost *= 2.5
        elif waste_type == 'E-Waste':
            base_cost *= 1.8
        elif waste_type == 'Construction':
            base_cost *= 1.3
        
        cost_management = max(500, base_cost + np.random.normal(0, 300))
        
        # Other features
        awareness_campaigns = np.random.poisson(municipal_efficiency)
        landfill_name = random.choice(landfill_names)
        
        # Generate realistic coordinates for Indian cities
        if city in ['Mumbai', 'Pune', 'Thane']:
            lat_base, lon_base = 19.0760, 72.8777
        elif city in ['Delhi']:
            lat_base, lon_base = 28.7041, 77.1025
        elif city in ['Bangalore']:
            lat_base, lon_base = 12.9716, 77.5946
        elif city in ['Chennai']:
            lat_base, lon_base = 13.0827, 80.2707
        elif city in ['Kolkata']:
            lat_base, lon_base = 22.5726, 88.3639
        elif city in ['Hyderabad']:
            lat_base, lon_base = 17.3850, 78.4867
        else:
            lat_base, lon_base = 23.0225, 72.5714  # Default to Ahmedabad area
        
        # Add small random offset for landfill location
        landfill_lat = lat_base + np.random.uniform(-0.5, 0.5)
        landfill_lon = lon_base + np.random.uniform(-0.5, 0.5)
        
        landfill_capacity = np.random.uniform(10000, 500000)  # Tons
        
        disposal_method = random.choice(disposal_methods)
        year = random.choice([2019, 2020, 2021, 2022, 2023])
        
        data.append({
            'City/District': city,
            'Waste Type': waste_type,
            'Waste Generated (Tons/Day)': round(waste_generated, 2),
            'Recycling Rate (%)': round(recycling_rate, 2),
            'Population Density (People/km²)': round(population_density, 2),
            'Municipal Efficiency Score (1-10)': municipal_efficiency,
            'Disposal Method': disposal_method,
            'Cost of Waste Management (₹/Ton)': round(cost_management, 2),
            'Awareness Campaigns Count': awareness_campaigns,
            'Landfill Name': landfill_name,
            'Landfill Location (Lat)': round(landfill_lat, 6),
            'Landfill Location (Long)': round(landfill_lon, 6),
            'Landfill Capacity (Tons)': round(landfill_capacity, 2),
            'Year': year
        })
    
    return pd.DataFrame(data)

def save_dataset(df, filename='waste_management_data.csv'):
    """Save dataset to data/raw directory"""
    # Get the project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    raw_data_path = os.path.join(project_root, 'data', 'raw', filename)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
    
    # Save the dataset
    df.to_csv(raw_data_path, index=False)
    print(f"Dataset saved to {raw_data_path}")
    return raw_data_path

if __name__ == "__main__":
    # Generate dataset
    df = generate_waste_management_data(1000)
    
    # Save to CSV
    save_dataset(df)
    print(f"Generated dataset with {len(df)} samples")
    print(f"Dataset shape: {df.shape}")
    print("\nDataset preview:")
    print(df.head())
    print("\nDataset info:")
    print(df.info())