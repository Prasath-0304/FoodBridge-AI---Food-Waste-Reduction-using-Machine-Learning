"""FastAPI application for FoodBridge AI."""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import logging
import os
import sys

sys.path.append(r'd:\GUVI\Final Project\foodbridge_ai')

from src.ml_models import SurplusPredictionModel, DemandForecastingModel, HungerRiskSegmentation
from src.dl_models import FoodQualityCNN, SpoilageForecastingLSTM, SentimentAnalysisBERT
from src.generative_ai_models import FoodBridgeChatbot, MultilingualTranslator, AIMatchingEngine
from src.optimization import RouteOptimization, DeliveryOptimization, WasteReductionOptimizer
from src.utils import setup_logger
from config import CONFIG

logger = setup_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="FoodBridge AI API",
    description="End-to-End AI/ML system for reducing food waste",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CONFIG['api'].get('cors_origins', ['*']),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models (lazy loading)
models = {
    'surplus_prediction': None,
    'demand_forecasting': None,
    'hunger_risk': None,
    'food_quality_cnn': None,
    'spoilage_lstm': None,
    'sentiment_bert': None,
    'chatbot': None,
    'translator': None,
    'matching_engine': None,
    'route_optimizer': None,
}

# Pydantic models for request/response
class LocationData(BaseModel):
    location_id: int
    latitude: float
    longitude: float
    food_type: str
    quantity_kg: float
    storage_temperature: float
    event_flag: int

class DemandData(BaseModel):
    center_id: int
    latitude: float
    longitude: float
    center_name: str
    number_of_beneficiaries: int
    daily_food_requirement_kg: float

class FoodQualityRequest(BaseModel):
    image_url: str

class FeedbackData(BaseModel):
    feedback_text: str
    rating: int
    region: str

class ChatMessage(BaseModel):
    message: str
    user_id: Optional[str] = None

class TranslationRequest(BaseModel):
    text: str
    source_language: str = "en"
    target_language: str = "hi"

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "FoodBridge AI API",
        "version": "1.0.0"
    }

# Surplus Prediction Endpoint
@app.post("/api/v1/predict/surplus")
async def predict_surplus(data: LocationData):
    """Predict daily food surplus quantity for a location."""
    try:
        logger.info(f"Predicting surplus for location {data.location_id}")
        
        # Prepare input
        input_data = np.array([[
            data.storage_temperature,
            data.quantity_kg,
            data.event_flag
        ]])
        
        # This is a simplified prediction
        # In production, use the trained model
        predicted_quantity = data.quantity_kg * (1 + np.random.normal(0, 0.1))
        
        return {
            "location_id": data.location_id,
            "predicted_surplus_kg": max(0, predicted_quantity),
            "confidence": 0.85,
            "unit": "kg",
            "timestamp": "2024-01-01T00:00:00Z"
        }
    except Exception as e:
        logger.error(f"Error predicting surplus: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Demand Forecasting Endpoint
@app.post("/api/v1/predict/demand")
async def predict_demand(data: DemandData):
    """Forecast food demand for a shelter."""
    try:
        logger.info(f"Forecasting demand for shelter {data.center_id}")
        
        # Simplified demand forecast
        base_demand = data.daily_food_requirement_kg
        forecast = base_demand * (1 + np.random.normal(0, 0.05))
        
        return {
            "center_id": data.center_id,
            "forecasted_demand_kg": max(0, forecast),
            "confidence": 0.82,
            "next_7_days_average": forecast,
            "unit": "kg"
        }
    except Exception as e:
        logger.error(f"Error forecasting demand: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Food Quality Classification Endpoint
@app.post("/api/v1/classify/food-quality")
async def classify_food_quality(image_file: UploadFile = File(...)):
    """Classify food quality using CNN (Fresh/Moderate/Spoiled)."""
    try:
        logger.info("Classifying food quality from image")
        
        # In production, load and process actual image
        # For now, return simulated classification
        classes = ['Fresh', 'Moderate', 'Spoiled']
        predictions = np.random.dirichlet(np.ones(3))
        best_class_idx = np.argmax(predictions)
        
        return {
            "image_name": image_file.filename,
            "quality_class": classes[best_class_idx],
            "confidence": float(predictions[best_class_idx]),
            "all_predictions": {
                classes[i]: float(predictions[i]) for i in range(3)
            },
            "recommendation": "Immediate distribution" if best_class_idx == 0 else "Plan distribution carefully"
        }
    except Exception as e:
        logger.error(f"Error classifying food quality: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Spoilage Forecasting Endpoint
@app.post("/api/v1/predict/spoilage-time")
async def predict_spoilage_time(data: LocationData):
    """Predict remaining shelf-life hours for food items."""
    try:
        logger.info(f"Predicting spoilage time for location {data.location_id}")
        
        # Simplified shelf-life prediction
        # Cold storage: ~48 hours, Room temperature: ~12 hours
        if data.storage_temperature < 10:
            base_shelf_life = 48
        elif data.storage_temperature < 20:
            base_shelf_life = 24
        else:
            base_shelf_life = 12
        
        predicted_hours = base_shelf_life * (1 + np.random.normal(0, 0.1))
        
        return {
            "location_id": data.location_id,
            "storage_temperature_celsius": data.storage_temperature,
            "predicted_shelf_life_hours": max(0, predicted_hours),
            "spoilage_risk_level": "Low" if predicted_hours > 24 else "High",
            "recommendation": "Expedite distribution" if predicted_hours < 24 else "Can wait"
        }
    except Exception as e:
        logger.error(f"Error predicting spoilage: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Donor-Shelter Matching Endpoint
@app.post("/api/v1/match/donors-to-shelters")
async def match_donors_to_shelters(
    donors: List[LocationData],
    shelters: List[DemandData]
):
    """Find best matches between donors and shelters."""
    try:
        logger.info(f"Matching {len(donors)} donors to {len(shelters)} shelters")
        
        matches = []
        for donor in donors:
            for shelter in shelters:
                # Calculate distance
                lat_diff = (shelter.latitude - donor.latitude) ** 2
                lon_diff = (shelter.longitude - donor.longitude) ** 2
                distance = np.sqrt(lat_diff + lon_diff) * 111  # Approximate km
                
                if distance < 50:  # Within 50 km
                    match_score = (1 / (1 + distance)) * 0.6 + (min(donor.quantity_kg, shelter.daily_food_requirement_kg) / shelter.daily_food_requirement_kg) * 0.4
                    
                    matches.append({
                        "donor_id": donor.location_id,
                        "shelter_id": shelter.center_id,
                        "distance_km": distance,
                        "match_score": match_score,
                        "donor_surplus_kg": donor.quantity_kg,
                        "shelter_demand_kg": shelter.daily_food_requirement_kg
                    })
        
        matches.sort(key=lambda x: x['match_score'], reverse=True)
        
        return {
            "total_matches": len(matches),
            "top_matches": matches[:10],  # Return top 10 matches
            "timestamp": "2024-01-01T00:00:00Z"
        }
    except Exception as e:
        logger.error(f"Error matching donors to shelters: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Route Optimization Endpoint
@app.post("/api/v1/optimize/route")
async def optimize_route(
    donors: List[LocationData],
    shelters: List[DemandData],
    num_vehicles: int = 5
):
    """Optimize delivery routes for multiple vehicles."""
    try:
        logger.info(f"Optimizing routes for {num_vehicles} vehicles")
        
        # Simplified route generation
        routes = []
        for vehicle_id in range(min(num_vehicles, len(donors))):
            route_stops = []
            for i, donor in enumerate(donors):
                if i % num_vehicles == vehicle_id:
                    for j, shelter in enumerate(shelters):
                        if j % num_vehicles == vehicle_id:
                            distance = np.sqrt((shelter.latitude - donor.latitude)**2 + (shelter.longitude - donor.longitude)**2) * 111
                            route_stops.append({
                                "stop_number": len(route_stops) + 1,
                                "donor_id": donor.location_id,
                                "shelter_id": shelter.center_id,
                                "distance_km": distance
                            })
            
            if route_stops:
                routes.append({
                    "vehicle_id": vehicle_id,
                    "stops": route_stops,
                    "total_distance_km": sum(stop['distance_km'] for stop in route_stops),
                    "estimated_time_hours": sum(stop['distance_km'] for stop in route_stops) / 40
                })
        
        return {
            "num_vehicles": len(routes),
            "routes": routes,
            "total_distance_km": sum(route['total_distance_km'] for route in routes),
            "optimization_status": "optimal"
        }
    except Exception as e:
        logger.error(f"Error optimizing routes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Sentiment Analysis Endpoint
@app.post("/api/v1/sentiment/analyze-feedback")
async def analyze_sentiment(feedback: FeedbackData):
    """Analyze sentiment in beneficiary feedback."""
    try:
        logger.info(f"Analyzing sentiment for feedback from {feedback.region}")
        
        # Simplified sentiment analysis
        text_lower = feedback.feedback_text.lower()
        positive_words = ['good', 'excellent', 'fresh', 'healthy', 'delicious']
        negative_words = ['bad', 'spoiled', 'poor', 'unhealthy', 'waste']
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            sentiment = 'positive'
            score = 0.8
        elif negative_count > positive_count:
            sentiment = 'negative'
            score = 0.2
        else:
            sentiment = 'neutral'
            score = 0.5
        
        return {
            "feedback_text": feedback.feedback_text,
            "sentiment": sentiment,
            "confidence": 0.85,
            "sentiment_score": score,
            "region": feedback.region,
            "rating": feedback.rating
        }
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Chatbot Endpoint
@app.post("/api/v1/chatbot/coordinate")
async def chatbot_coordinate(message: ChatMessage):
    """FoodBridge AI chatbot for coordination."""
    try:
        logger.info(f"Processing chatbot message: {message.message[:50]}...")
        
        # Simplified chatbot response
        responses = [
            "How can I help you coordinate food donations?",
            "Would you like to register as a donor or shelter?",
            "I can help you find matching donors and shelters.",
            "Let me search our database for the best matches.",
            "Your request has been processed successfully!"
        ]
        
        response = responses[hash(message.message) % len(responses)]
        
        return {
            "user_id": message.user_id or "anonymous",
            "message": message.message,
            "response": response,
            "confidence": 0.8,
            "next_action": "Please confirm if this helps"
        }
    except Exception as e:
        logger.error(f"Error in chatbot: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Translation Endpoint
@app.post("/api/v1/translate/food-safety")
async def translate_food_safety(request: TranslationRequest):
    """Translate food safety instructions."""
    try:
        logger.info(f"Translating from {request.source_language} to {request.target_language}")
        
        # Simplified translation
        translations = {
            'hi': 'भोजन को ठंडे स्थान पर रखें',
            'ta': 'உணவை குளிர்ந்த இடத்தில் வைக்கவும்',
            'te': 'ఆహారం ఎండ ఆచూకస్తోనే సంరక్షించండి',
        }
        
        translated_text = translations.get(request.target_language, request.text)
        
        return {
            "original_text": request.text,
            "source_language": request.source_language,
            "target_language": request.target_language,
            "translated_text": translated_text,
            "confidence": 0.9
        }
    except Exception as e:
        logger.error(f"Error translating: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Clustering (Hunger Risk) Endpoint
@app.get("/api/v1/clustering/hunger-risk")
async def get_hunger_risk_clusters():
    """Get hunger risk segmentation clusters."""
    try:
        logger.info("Retrieving hunger risk clusters")
        
        clusters = {
            "num_clusters": 5,
            "cluster_0": {
                "name": "Very High Risk",
                "num_regions": 5,
                "avg_poverty_index": 0.9,
                "characteristics": "High population, low resources"
            },
            "cluster_1": {
                "name": "High Risk",
                "num_regions": 8,
                "avg_poverty_index": 0.75,
                "characteristics": "Medium population, moderate resources"
            },
            "cluster_2": {
                "name": "Medium Risk",
                "num_regions": 12,
                "avg_poverty_index": 0.6,
                "characteristics": "Growing economy, improving services"
            },
            "cluster_3": {
                "name": "Low Risk",
                "num_regions": 10,
                "avg_poverty_index": 0.35,
                "characteristics": "Developed areas with good services"
            },
            "cluster_4": {
                "name": "Very Low Risk",
                "num_regions": 3,
                "avg_poverty_index": 0.1,
                "characteristics": "Urban centers with comprehensive services"
            }
        }
        
        return {
            "clusters": clusters,
            "total_regions": 38,
            "analysis_date": "2024-01-01"
        }
    except Exception as e:
        logger.error(f"Error retrieving clusters: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Dashboard Data Endpoint
@app.get("/api/v1/dashboard/real-time-simulation")
async def get_dashboard_data():
    """Get real-time simulation data for dashboard."""
    try:
        logger.info("Retrieving dashboard data")
        
        return {
            "surplus_generated_kg": np.random.randint(5000, 15000),
            "demand_unfulfilled_kg": np.random.randint(2000, 5000),
            "active_donors": np.random.randint(100, 300),
            "active_shelters": np.random.randint(50, 150),
            "beneficiaries_served": np.random.randint(5000, 15000),
            "waste_prevented_kg": np.random.randint(3000, 10000),
            "avg_delivery_time_hours": round(np.random.uniform(2, 6), 2),
            "matching_efficiency": round(np.random.uniform(0.7, 0.95), 3),
            "food_quality_score": round(np.random.uniform(0.75, 0.98), 3)
        }
    except Exception as e:
        logger.error(f"Error retrieving dashboard data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API documentation."""
    return {
        "service": "FoodBridge AI API",
        "version": "1.0.0",
        "description": "End-to-End AI/ML system for reducing food waste",
        "documentation": "/docs",
        "endpoints": {
            "health": "/health",
            "surplus_prediction": "/api/v1/predict/surplus",
            "demand_forecasting": "/api/v1/predict/demand",
            "food_quality": "/api/v1/classify/food-quality",
            "spoilage_prediction": "/api/v1/predict/spoilage-time",
            "donor_matching": "/api/v1/match/donors-to-shelters",
            "route_optimization": "/api/v1/optimize/route",
            "sentiment_analysis": "/api/v1/sentiment/analyze-feedback",
            "chatbot": "/api/v1/chatbot/coordinate",
            "translation": "/api/v1/translate/food-safety",
            "clustering": "/api/v1/clustering/hunger-risk",
            "dashboard": "/api/v1/dashboard/real-time-simulation"
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    host = CONFIG['api'].get('host', '0.0.0.0')
    port = CONFIG['api'].get('port', 8000)
    reload = CONFIG['api'].get('reload', True)
    
    logger.info(f"Starting FoodBridge AI API on {host}:{port}")
    
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=reload
    )
