"""Deep Learning models for FoodBridge AI."""

import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class FoodQualityCNN:
    """CNN model for food quality classification (Fresh/Moderate/Spoiled)."""
    
    def __init__(self, config: dict = None):
        """Initialize CNN model for food classification."""
        self.config = config or {}
        self.model = None
        self.classes = ['fresh', 'moderate', 'spoiled']
        
    def build_model(self):
        """Build CNN model architecture."""
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, 
                                                 Dropout, BatchNormalization)
            from tensorflow.keras.optimizers import Adam
            
            logger.info("Building EfficientNet-based food quality CNN")
            
            self.model = Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                Dropout(0.25),
                
                Conv2D(64, (3, 3), activation='relu'),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                Dropout(0.25),
                
                Conv2D(128, (3, 3), activation='relu'),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                Dropout(0.25),
                
                Conv2D(256, (3, 3), activation='relu'),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                Dropout(0.25),
                
                Flatten(),
                Dense(512, activation='relu'),
                BatchNormalization(),
                Dropout(0.5),
                
                Dense(256, activation='relu'),
                BatchNormalization(),
                Dropout(0.5),
                
                Dense(3, activation='softmax')  # 3 classes
            ])
            
            self.model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info("✓ CNN model built successfully")
            
        except ImportError:
            logger.error("TensorFlow not available. Please install tensorflow to use CNN models.")
            raise
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray):
        """Train the CNN model."""
        if self.model is None:
            self.build_model()
        
        logger.info(f"Training food quality CNN with {X_train.shape[0]} samples")
        
        cnn_params = self.config.get('food_quality_cnn', {})
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=cnn_params.get('epochs', 50),
            batch_size=cnn_params.get('batch_size', 32),
            verbose=1
        )
        
        logger.info("✓ Food quality CNN training completed")
        return history
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on food images."""
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = self.model.predict(X)
        class_predictions = np.argmax(predictions, axis=1)
        confidence = np.max(predictions, axis=1)
        
        return class_predictions, confidence
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        """Evaluate model on test data."""
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        logger.info(f"Food Quality CNN - Test Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        return {'loss': loss, 'accuracy': accuracy}


class SpoilageForecastingLSTM:
    """LSTM model to predict shelf-life and spoilage time."""
    
    def __init__(self, config: dict = None):
        """Initialize spoilage forecasting LSTM model."""
        self.config = config or {}
        self.model = None
        
    def build_model(self, input_shape: Tuple):
        """Build LSTM model for spoilage forecasting."""
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed
            from tensorflow.keras.optimizers import Adam
            
            logger.info("Building spoilage forecasting LSTM model")
            
            lstm_params = self.config.get('spoilage_forecasting', {})
            
            self.model = Sequential([
                LSTM(
                    lstm_params.get('units', 64),
                    activation='relu',
                    input_shape=input_shape,
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
                Dense(1)  # Predict shelf-life hours
            ])
            
            self.model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            logger.info("✓ Spoilage forecasting LSTM model built")
            
        except ImportError:
            logger.error("TensorFlow not available.")
            raise
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None):
        """Train spoilage forecasting model."""
        if self.model is None:
            self.build_model(X_train.shape[1:])
        
        logger.info(f"Training spoilage forecasting LSTM with {X_train.shape[0]} samples")
        
        lstm_params = self.config.get('spoilage_forecasting', {})
        
        fit_kwargs = {
            'epochs': lstm_params.get('epochs', 50),
            'batch_size': lstm_params.get('batch_size', 32),
            'verbose': 1
        }
        
        if X_val is not None and y_val is not None:
            fit_kwargs['validation_data'] = (X_val, y_val)
        
        history = self.model.fit(X_train, y_train, **fit_kwargs)
        
        logger.info("✓ Spoilage forecasting LSTM training completed")
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict shelf-life for food items."""
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = self.model.predict(X, verbose=0)
        return np.maximum(predictions.flatten(), 0)  # Ensure non-negative predictions


class SentimentAnalysisBERT:
    """BERT-based sentiment analysis for beneficiary feedback."""
    
    def __init__(self, config: dict = None):
        """Initialize sentiment analysis model."""
        self.config = config or {}
        self.model = None
        self.tokenizer = None
        
    def setup_model(self):
        """Setup BERT model for sentiment analysis."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            
            logger.info("Setting up DistilBERT for sentiment analysis")
            
            model_name = "distilbert-base-uncased"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=2  # Binary sentiment: positive/negative
            )
            
            logger.info("✓ DistilBERT model loaded")
            
        except ImportError:
            logger.error("Transformers library not available.")
            raise
    
    def train(self, texts: list, labels: list, epochs: int = 3):
        """Fine-tune BERT model for sentiment analysis."""
        if self.model is None:
            self.setup_model()
        
        logger.info(f"Fine-tuning DistilBERT with {len(texts)} samples")
        
        try:
            from transformers import TextClassificationPipeline
            import torch
            from torch.optim import AdamW
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            
            optimizer = AdamW(self.model.parameters(), lr=2e-5)
            
            for epoch in range(epochs):
                logger.info(f"Epoch {epoch + 1}/{epochs}")
                for text, label in zip(texts, labels):
                    inputs = self.tokenizer(
                        text,
                        max_length=128,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    outputs = self.model(**inputs, labels=torch.tensor([label]).to(device))
                    loss = outputs.loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            logger.info("✓ BERT fine-tuning completed")
            
        except ImportError:
            logger.warning("PyTorch not available. Using inference only.")
    
    def predict_sentiment(self, texts: list) -> list:
        """Predict sentiment for given texts."""
        if self.model is None:
            self.setup_model()
        
        if self.tokenizer is None:
            self.setup_model()
        
        predictions = []
        
        try:
            import torch
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            self.model.eval()
            
            for text in texts:
                inputs = self.tokenizer(
                    text,
                    max_length=128,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    predicted_class = logits.argmax().item()
                    confidence = torch.softmax(logits, dim=1).max().item()
                
                predictions.append({
                    'text': text,
                    'sentiment': 'positive' if predicted_class == 1 else 'negative',
                    'confidence': confidence
                })
        
        except Exception as e:
            logger.error(f"Error in sentiment prediction: {e}")
        
        return predictions
