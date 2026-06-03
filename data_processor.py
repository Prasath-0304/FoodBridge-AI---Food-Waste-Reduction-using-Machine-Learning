"""Data processing module for FoodBridge AI."""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """Data processor for preparing and transforming data."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize data processor with config."""
        self.config = config
        self.scaler = StandardScaler()
        self.encoders = {}
        self.feature_names = None
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from CSV file."""
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Data shape: {df.shape}")
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, method: str = "mean") -> pd.DataFrame:
        """Handle missing values in dataframe."""
        logger.info(f"Handling missing values using {method}")
        
        if method == "mean":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif method == "median":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif method == "forward_fill":
            df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Fill categorical columns with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')
        
        logger.info(f"Missing values after handling: {df.isnull().sum().sum()}")
        return df
    
    def remove_outliers(self, df: pd.DataFrame, method: str = "iqr", 
                       threshold: float = 1.5) -> pd.DataFrame:
        """Remove outliers from numeric columns."""
        logger.info(f"Removing outliers using {method}")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if method == "iqr":
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        elif method == "zscore":
            for col in numeric_cols:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df = df[z_scores < threshold]
        
        logger.info(f"Shape after outlier removal: {df.shape}")
        return df
    
    def encode_categorical(self, df: pd.DataFrame, method: str = "onehot") -> pd.DataFrame:
        """Encode categorical variables."""
        logger.info(f"Encoding categorical variables using {method}")
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if method == "onehot":
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        elif method == "label":
            for col in categorical_cols:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    df[col] = self.encoders[col].fit_transform(df[col])
                else:
                    df[col] = self.encoders[col].transform(df[col])
        
        return df
    
    def normalize_features(self, df: pd.DataFrame, method: str = "standard",
                          fit: bool = True) -> pd.DataFrame:
        """Normalize numeric features."""
        logger.info(f"Normalizing features using {method}")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if method == "standard":
            if fit:
                df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
            else:
                df[numeric_cols] = self.scaler.transform(df[numeric_cols])
        elif method == "minmax":
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            if fit:
                df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            else:
                df[numeric_cols] = scaler.transform(df[numeric_cols])
        
        return df
    
    def create_time_features(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """Create time-based features from datetime column."""
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
            df['year'] = df[date_col].dt.year
            df['month'] = df[date_col].dt.month
            df['day'] = df[date_col].dt.day
            df['dayofweek'] = df[date_col].dt.dayofweek
            df['quarter'] = df[date_col].dt.quarter
            df['weekofyear'] = df[date_col].dt.isocalendar().week
            df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        return df
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                   test_size: float = 0.2, validation_size: float = 0.15,
                   random_state: int = 42) -> Tuple:
        """Split data into train, validation, and test sets."""
        logger.info(f"Splitting data: test_size={test_size}, validation_size={validation_size}")
        
        # First split: train+val / test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Second split: train / val
        val_ratio = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state
        )
        
        logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def preprocess(self, df: pd.DataFrame, config: Dict[str, Any],
                   target_col: str = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Complete preprocessing pipeline."""
        # Handle missing values
        df = self.handle_missing_values(df, method=config.get('handle_missing', 'mean'))
        
        # Remove outliers
        df = self.remove_outliers(df, method=config.get('outlier_method', 'iqr'))
        
        # Encode categorical
        df = self.encode_categorical(df, method=config.get('encode_categorical', 'onehot'))
        
        # Normalize
        df = self.normalize_features(df, method=config.get('normalize_method', 'standard'))
        
        # Split features and target
        if target_col and target_col in df.columns:
            y = df[target_col]
            X = df.drop(columns=[target_col])
        else:
            X = df
            y = None
        
        self.feature_names = X.columns.tolist()
        
        return X, y
