"""Machine Learning models for FoodBridge AI."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

logger = logging.getLogger(__name__)

class SurplusPredictionModel:
    """Predict daily food surplus quantity using multiple algorithms."""
    
    def __init__(self, config: dict = None):
        """Initialize surplus prediction model."""
        self.config = config or {}
        self.models = {}
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train multiple models for surplus prediction."""
        logger.info(f"Training surplus prediction models with {X_train.shape[0]} samples")
        
        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Linear Regression (Baseline)
        self.models['linear_regression'] = LinearRegression()
        self.models['linear_regression'].fit(X_train_scaled, y_train)
        
        # Random Forest
        rf_params = self.config.get('random_forest', {})
        self.models['random_forest'] = RandomForestRegressor(
            n_estimators=rf_params.get('n_estimators', 100),
            max_depth=rf_params.get('max_depth', 15),
            min_samples_split=rf_params.get('min_samples_split', 5),
            random_state=42
        )
        self.models['random_forest'].fit(X_train_scaled, y_train)
        
        # Gradient Boosting
        gb_params = self.config.get('xgboost', {})
        self.models['gradient_boosting'] = GradientBoostingRegressor(
            n_estimators=gb_params.get('n_estimators', 100),
            max_depth=gb_params.get('max_depth', 6),
            learning_rate=gb_params.get('learning_rate', 0.1),
            random_state=42
        )
        self.models['gradient_boosting'].fit(X_train_scaled, y_train)
        
        self.is_fitted = True
        logger.info(f"✓ Trained {len(self.models)} models for surplus prediction")
        
    def predict(self, X: pd.DataFrame, model_name: str = 'random_forest') -> np.ndarray:
        """Make predictions using specified model."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self.models.keys())}")
        
        return self.models[model_name].predict(X_scaled)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series):
        """Evaluate all trained models."""
        X_test_scaled = self.scaler.transform(X_test)
        results = {}
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test_scaled)
            results[name] = {
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred)
            }
            logger.info(f"{name}: MAE={results[name]['mae']:.4f}, RMSE={results[name]['rmse']:.4f}, R²={results[name]['r2']:.4f}")
        
        return results


class DemandForecastingModel:
    """Forecast food demand for shelters and orphanages using time series."""
    
    def __init__(self, config: dict = None):
        """Initialize demand forecasting model."""
        self.config = config or {}
        self.model = None
        self.scaler = StandardScaler()
        self.sequence_length = config.get('lstm', {}).get('sequence_length', 7) if config else 7
        
    def prepare_sequences(self, X: np.ndarray, y: np.ndarray):
        """Prepare sequences for LSTM model."""
        sequences_X, sequences_y = [], []
        
        for i in range(len(X) - self.sequence_length):
            sequences_X.append(X[i:i + self.sequence_length])
            sequences_y.append(y[i + self.sequence_length])
        
        return np.array(sequences_X), np.array(sequences_y)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train demand forecasting model."""
        logger.info(f"Training demand forecasting LSTM model")
        
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.optimizers import Adam
            
            # Prepare sequences
            X_seq, y_seq = self.prepare_sequences(X_train, y_train)
            
            # Normalize data
            X_seq_scaled = X_seq.reshape(-1, X_seq.shape[-1])
            X_seq_scaled = self.scaler.fit_transform(X_seq_scaled)
            X_seq_scaled = X_seq_scaled.reshape(X_seq.shape)
            
            # Build LSTM model
            lstm_params = self.config.get('lstm', {})
            self.model = Sequential([
                LSTM(
                    lstm_params.get('units', 64),
                    activation='relu',
                    input_shape=(self.sequence_length, X_seq.shape[-1]),
                    return_sequences=True
                ),
                Dropout(lstm_params.get('dropout', 0.2)),
                LSTM(
                    lstm_params.get('units', 64),
                    activation='relu',
                    return_sequences=False
                ),
                Dropout(lstm_params.get('dropout', 0.2)),
                Dense(32, activation='relu'),
                Dense(1)
            ])
            
            self.model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse'
            )
            
            # Train
            epochs = lstm_params.get('epochs', 50)
            batch_size = lstm_params.get('batch_size', 32)
            
            self.model.fit(
                X_seq_scaled, y_seq,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0
            )
            
            logger.info("✓ Demand forecasting LSTM model trained")
        except ImportError:
            logger.warning("TensorFlow not available. Using simple linear regression instead.")
            self.model = LinearRegression()
            self.model.fit(X_train, y_train)


class HungerRiskSegmentation:
    """Segment regions by hunger risk using clustering algorithms."""
    
    def __init__(self, config: dict = None):
        """Initialize clustering model."""
        self.config = config or {}
        self.kmeans_model = None
        self.scaler = StandardScaler()
        
    def train(self, X: pd.DataFrame):
        """Train clustering model."""
        logger.info(f"Training hunger risk segmentation with K-Means")
        
        from sklearn.cluster import KMeans
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train KMeans
        kmeans_params = self.config.get('kmeans', {})
        self.kmeans_model = KMeans(
            n_clusters=kmeans_params.get('n_clusters', 5),
            init=kmeans_params.get('init', 'k-means++'),
            random_state=42
        )
        
        clusters = self.kmeans_model.fit_predict(X_scaled)
        logger.info(f"✓ Segmented regions into {kmeans_params.get('n_clusters', 5)} clusters")
        
        return clusters
    
    def get_cluster_distribution(self, clusters: np.ndarray) -> dict:
        """Get distribution of clusters."""
        unique, counts = np.unique(clusters, return_counts=True)
        return {int(cluster): int(count) for cluster, count in zip(unique, counts)}


class WastePatternMiner:
    """Mine patterns in food waste data using association rule mining."""
    
    def __init__(self, config: dict = None):
        """Initialize pattern miner."""
        self.config = config or {}
        self.patterns = {}
        
    def analyze_patterns(self, df: pd.DataFrame) -> dict:
        """Analyze waste patterns from data."""
        logger.info("Mining food waste patterns")
        
        patterns = {
            'high_waste_conditions': self._find_high_waste_conditions(df),
            'temporal_patterns': self._find_temporal_patterns(df),
            'location_patterns': self._find_location_patterns(df),
            'food_type_patterns': self._find_food_type_patterns(df)
        }
        
        self.patterns = patterns
        logger.info("✓ Pattern mining completed")
        return patterns
    
    def _find_high_waste_conditions(self, df: pd.DataFrame) -> dict:
        """Find conditions leading to high waste."""
        if 'quantity_kg' not in df.columns:
            return {}
        
        high_waste_threshold = df['quantity_kg'].quantile(0.75)
        high_waste_data = df[df['quantity_kg'] > high_waste_threshold]
        
        patterns = {}
        
        # Weather impact
        if 'weather_condition' in high_waste_data.columns:
            weather_impact = high_waste_data['weather_condition'].value_counts()
            patterns['weather_impact'] = weather_impact.to_dict()
        
        # Event impact
        if 'event_flag' in high_waste_data.columns:
            patterns['event_impact'] = {
                'high_waste_with_events': (high_waste_data['event_flag'].sum() / len(high_waste_data)) * 100
            }
        
        return patterns
    
    def _find_temporal_patterns(self, df: pd.DataFrame) -> dict:
        """Find temporal patterns."""
        patterns = {}
        
        if 'day_of_week' in df.columns:
            patterns['day_of_week_avg'] = df.groupby('day_of_week')['quantity_kg'].mean().to_dict()
        
        return patterns
    
    def _find_location_patterns(self, df: pd.DataFrame) -> dict:
        """Find location-based patterns."""
        patterns = {}
        
        if 'location_id' in df.columns:
            patterns['high_waste_locations'] = df.groupby('location_id')['quantity_kg'].sum().nlargest(5).to_dict()
        
        return patterns
    
    def _find_food_type_patterns(self, df: pd.DataFrame) -> dict:
        """Find food type patterns."""
        patterns = {}
        
        if 'food_type' in df.columns:
            patterns['food_type_distribution'] = df['food_type'].value_counts().to_dict()
            patterns['food_type_avg_quantity'] = df.groupby('food_type')['quantity_kg'].mean().to_dict()
        
        return patterns
