"""Model training utilities for FoodBridge AI."""

import numpy as np
import logging
from typing import Dict, Any, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Evaluate model performance."""
    
    @staticmethod
    def evaluate_regression(y_true, y_pred, model_name: str = "Model") -> Dict[str, float]:
        """Evaluate regression model."""
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
        }
        
        logger.info(f"\n{model_name} Regression Metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric.upper()}: {value:.4f}")
        
        return metrics
    
    @staticmethod
    def evaluate_classification(y_true, y_pred, model_name: str = "Model") -> Dict[str, float]:
        """Evaluate classification model."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        
        logger.info(f"\n{model_name} Classification Metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric.upper()}: {value:.4f}")
        
        return metrics
    
    @staticmethod
    def evaluate_baseline(y_true, baseline_pred, metric_type: str = "regression") -> Dict[str, float]:
        """Evaluate baseline model performance."""
        if metric_type == "regression":
            return ModelEvaluator.evaluate_regression(y_true, baseline_pred, "Baseline")
        else:
            return ModelEvaluator.evaluate_classification(y_true, baseline_pred, "Baseline")
