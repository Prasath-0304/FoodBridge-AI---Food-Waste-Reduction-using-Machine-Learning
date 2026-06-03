"""Data generation module for creating synthetic FoodBridge AI datasets."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class DataGenerator:
    """Generate synthetic datasets for FoodBridge AI project."""
    
    @staticmethod
    def generate_surplus_demand_data(n_samples: int = 500, 
                                    n_locations: int = 20) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate synthetic surplus supply and demand data for Chennai."""
        
        logger.info(f"Generating {n_samples} samples for {n_locations} Chennai locations")
        
        # Chennai area-based locations with coordinates (Latitude, Longitude)
        # Major areas in Chennai with realistic coordinates
        chennai_locations = {
            'T_Nagar': (13.0389, 80.2377),
            'Pallavaram': (12.9716, 80.2090),
            'Tambaram': (12.9166, 80.1751),
            'Velachery': (13.0044, 80.2230),
            'Saidapet': (13.0044, 80.2138),
            'Mylapore': (13.0349, 80.2680),
            'Adyar': (13.0029, 80.2447),
            'Guindy': (13.0035, 80.2198),
            'Kodambakkam': (13.0474, 80.1906),
            'Chetpet': (13.0626, 80.2294),
            'Nungambakkam': (13.0577, 80.2406),
            'Teynampet': (13.0522, 80.2549),
            'Alandur': (13.0184, 80.2063),
            'Madipakkam': (12.9947, 80.1933),
            'Tiruneermalai': (12.9856, 80.1747),
            'Ramakrishnapur': (13.0662, 80.2011),
            'Kelambakkam': (12.8718, 80.1913),
            'Mambakkam': (13.0898, 80.1604),
            'Chengalpattu': (12.6738, 80.1957),
            'Urapakkam': (12.8382, 80.1609)
        }
        
        location_names = list(chennai_locations.keys())
        location_coords = list(chennai_locations.values())
        
        # Generate surplus data
        surplus_data = []
        start_date = datetime(2023, 1, 1)
        
        for i in range(n_samples):
            loc_idx = np.random.randint(0, min(n_locations, len(location_names)))
            area_name = location_names[loc_idx]
            lat, lon = location_coords[loc_idx]
            
            surplus_data.append({
                'location_id': loc_idx,
                'area_name': area_name,
                'latitude': lat + np.random.normal(0, 0.005),  # Add slight variation
                'longitude': lon + np.random.normal(0, 0.005),
                'restaurant_name': f'{area_name}-Restaurant-{loc_idx}',
                'food_type': np.random.choice(['Cooked', 'Packaged', 'Raw']),
                'quantity_kg': np.random.gamma(5, 2) * 10,  # 10-100 kg range
                'preparation_time': (start_date + timedelta(days=i)).isoformat(),
                'storage_temperature': np.random.normal(5, 2),  # Cold storage
                'event_flag': np.random.choice([0, 1], p=[0.7, 0.3]),
                'weather_condition': np.random.choice(['Clear', 'Rainy', 'Humid', 'Hot']),
                'day_of_week': (i % 7),
                'humidity_pct': np.random.normal(60, 15),
                'shelf_life_hours': np.random.gamma(2, 20),  # Hours until spoilage
            })
        
        surplus_df = pd.DataFrame(surplus_data)
        
        # Generate demand data (Shelters) - Time series matching surplus data
        demand_data = []
        for i in range(n_samples):
            shelter_idx = np.random.randint(0, min(n_locations, len(location_names)))
            area_name = location_names[shelter_idx]
            lat, lon = location_coords[shelter_idx]
            demand_data.append({
                'center_id': shelter_idx,
                'area_name': area_name,
                'latitude': lat,
                'longitude': lon,
                'shelter_name': f'{area_name}-Shelter',
                'number_of_beneficiaries': np.random.randint(50, 500),
                'daily_food_requirement_kg': np.random.gamma(3, 30),
                'special_requirements': np.random.choice(['None', 'Children', 'Elderly', 'Special Diet']),
                'region_poverty_index': np.random.uniform(0.3, 0.9),
                'food_inflation_rate': np.random.uniform(5, 15) / 100,
                'demand_timestamp': (start_date + timedelta(days=i)).isoformat(),
            })
        
        demand_df = pd.DataFrame(demand_data)
        
        logger.info(f"Surplus data shape: {surplus_df.shape}")
        logger.info(f"Demand data shape: {demand_df.shape}")
        logger.info(f"Locations: {location_names[:min(n_locations, len(location_names))]}")
        
        return surplus_df, demand_df
    
    @staticmethod
    def generate_food_quality_images_labels(n_samples: int = 100) -> pd.DataFrame:
        """Generate synthetic food quality labels for image classification."""
        
        labels_data = []
        for i in range(n_samples):
            labels_data.append({
                'image_id': f'food_{i:04d}.jpg',
                'quality_label': np.random.choice(['Fresh', 'Moderate', 'Spoiled'], 
                                                 p=[0.5, 0.3, 0.2]),
                'confidence': np.random.uniform(0.7, 1.0),
                'color_intensity': np.random.uniform(50, 255),
                'texture_score': np.random.uniform(0, 10),
                'mold_detected': np.random.choice([0, 1], p=[0.85, 0.15]),
                'shelf_life_remaining_hours': np.random.gamma(2, 15),
            })
        
        return pd.DataFrame(labels_data)
    
    @staticmethod
    def generate_beneficiary_feedback(n_samples: int = 200) -> pd.DataFrame:
        """Generate synthetic beneficiary feedback for sentiment analysis."""
        
        positive_keywords = ['good', 'fresh', 'excellent', 'healthy', 'nutritious', 'delicious']
        negative_keywords = ['bad', 'spoiled', 'unhealthy', 'low quality', 'waste']
        neutral_keywords = ['okay', 'average', 'acceptable', 'normal']
        
        feedback_data = []
        for i in range(n_samples):
            sentiment_choice = np.random.choice(['positive', 'negative', 'neutral'], p=[0.5, 0.2, 0.3])
            
            if sentiment_choice == 'positive':
                words = np.random.choice(positive_keywords, size=np.random.randint(2, 5))
                text = f"The food was {' and '.join(words)}"
                label = 1
            elif sentiment_choice == 'negative':
                words = np.random.choice(negative_keywords, size=np.random.randint(2, 4))
                text = f"The food was {' and '.join(words)}"
                label = 0
            else:
                words = np.random.choice(neutral_keywords, size=np.random.randint(1, 3))
                text = f"The food was {' and '.join(words)}"
                label = 0
            
            feedback_data.append({
                'feedback_id': i,
                'feedback_text': text,
                'rating': np.random.randint(1, 6),
                'sentiment_label': label,
                'region': np.random.choice(['North', 'South', 'East', 'West', 'Central']),
                'delivery_delay_hours': np.random.exponential(2),
                'timestamp': (datetime(2023, 1, 1) + timedelta(days=i)).isoformat(),
            })
        
        return pd.DataFrame(feedback_data)
    
    @staticmethod
    def generate_weather_data(n_locations: int = 20, n_days: int = 365) -> pd.DataFrame:
        """Generate synthetic weather data."""
        
        weather_data = []
        start_date = datetime(2023, 1, 1)
        
        for loc_idx in range(n_locations):
            for day in range(n_days):
                weather_data.append({
                    'location_id': loc_idx,
                    'date': (start_date + timedelta(days=day)).isoformat(),
                    'temperature_celsius': np.random.normal(25, 8),
                    'humidity_percent': np.random.normal(65, 15),
                    'rainfall_mm': np.random.gamma(2, 2),
                    'wind_speed_kmh': np.random.exponential(5),
                })
        
        return pd.DataFrame(weather_data)
