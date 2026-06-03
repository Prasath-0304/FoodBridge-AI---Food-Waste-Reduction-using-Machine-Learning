"""Generative AI models for FoodBridge AI."""

import logging
import numpy as np
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

class FoodBridgeChatbot:
    """RAG-based chatbot for coordinating donors, NGOs, and shelters."""
    
    def __init__(self, config: dict = None):
        """Initialize chatbot with RAG pipeline."""
        self.config = config or {}
        self.model = None
        self.tokenizer = None
        self.knowledge_base = None
        
    def setup_rag_pipeline(self, documents: List[str] = None):
        """Setup Retrieval-Augmented Generation pipeline."""
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity
            
            logger.info("Setting up RAG pipeline for FoodBridge chatbot")
            
            # Use sentence-transformers for embeddings
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize knowledge base with default documents
            if documents is None:
                documents = self._get_default_knowledge_base()
            
            self.knowledge_base = {
                'documents': documents,
                'embeddings': self.model.encode(documents)
            }
            
            logger.info(f"✓ RAG pipeline initialized with {len(documents)} documents")
            
        except ImportError:
            logger.error("sentence-transformers not available.")
            raise
    
    def _get_default_knowledge_base(self) -> List[str]:
        """Get default knowledge base documents."""
        return [
            "FoodBridge AI connects food surplus sources with food-insecure populations",
            "Donors can register their available food surplus with location and quantity",
            "Shelters and orphanages can register their food requirements",
            "The system matches donors to nearest recipients to minimize delivery time",
            "Fresh food is prioritized over packaged and raw food",
            "Special dietary requirements like children, elderly, and allergies are considered",
            "Delivery optimization uses routes that minimize time and spoilage",
            "Real-time tracking helps monitor food from donor to beneficiary",
            "Feedback from beneficiaries helps improve the system",
            "The chatbot can help with food safety, quality, and handling instructions"
        ]
    
    def retrieve_context(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve relevant documents for the query."""
        if self.knowledge_base is None:
            self.setup_rag_pipeline()
        
        try:
            query_embedding = self.model.encode(query)
            from sklearn.metrics.pairwise import cosine_similarity
            
            similarities = cosine_similarity(
                [query_embedding],
                self.knowledge_base['embeddings']
            )[0]
            
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            context = [self.knowledge_base['documents'][i] for i in top_indices]
            
            return context
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []
    
    def generate_response(self, query: str) -> Dict:
        """Generate response using RAG pipeline."""
        try:
            from transformers import pipeline
            
            # Retrieve relevant context
            context = self.retrieve_context(query)
            
            if not context:
                return {
                    'query': query,
                    'response': "I'm not sure about that. Could you rephrase your question?",
                    'confidence': 0.5
                }
            
            # Use a simple retrieval-based approach
            context_text = " ".join(context)
            
            # Generate simple response (in production, use GPT or similar)
            response = f"Based on the FoodBridge AI system: {context[0]}"
            
            return {
                'query': query,
                'response': response,
                'context': context,
                'confidence': 0.8
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                'query': query,
                'response': "I encountered an error processing your request.",
                'confidence': 0.0
            }


class MultilingualTranslator:
    """Translate food safety instructions across multiple languages."""
    
    def __init__(self, config: dict = None):
        """Initialize multilingual translator."""
        self.config = config or {}
        self.model = None
        self.tokenizer = None
        self.supported_languages = [
            'hi', 'ta', 'te', 'kn', 'ml', 'bn', 'mr', 'gu', 'pa', 'ur'  # Indian languages
        ]
        
    def setup_translator(self):
        """Setup MarianMT translator model."""
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            
            logger.info("Setting up multilingual translator")
            
            # Use marian models for translation
            model_name = "Helsinki-NLP/opus-mt-en-hi"  # English to Hindi example
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            logger.info("✓ Multilingual translator initialized")
            
        except ImportError:
            logger.error("Transformers not available for translation.")
            raise
    
    def translate(self, text: str, source_lang: str = 'en', 
                  target_lang: str = 'hi') -> str:
        """Translate text to target language."""
        if self.model is None:
            self.setup_translator()
        
        try:
            if source_lang == target_lang:
                return text
            
            # In a real scenario, load the appropriate model for source-target pair
            inputs = self.tokenizer(text, return_tensors="pt", padding=True)
            outputs = self.model.generate(**inputs)
            translated = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            return translated[0] if translated else text
            
        except Exception as e:
            logger.warning(f"Translation failed: {e}. Returning original text.")
            return text
    
    def translate_food_safety_instructions(self, instructions: str, 
                                          target_language: str) -> Dict:
        """Translate food safety instructions to target language."""
        translated_instructions = self.translate(instructions, 'en', target_language)
        
        return {
            'original': instructions,
            'translated': translated_instructions,
            'source_language': 'en',
            'target_language': target_language
        }


class AIMatchingEngine:
    """Match donors to shelters using vector similarity and geospatial scoring."""
    
    def __init__(self, config: dict = None):
        """Initialize matching engine."""
        self.config = config or {}
        self.donor_embeddings = None
        self.shelter_embeddings = None
        
    def setup_embeddings(self):
        """Setup embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info("Setting up embedding model for matching engine")
            
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            logger.info("✓ Embedding model loaded")
            
        except ImportError:
            logger.error("sentence-transformers not available.")
            raise
    
    def calculate_distance(self, lat1: float, lon1: float, 
                          lat2: float, lon2: float) -> float:
        """Calculate distance between two points using Haversine formula."""
        from math import radians, sin, cos, sqrt, atan2
        
        R = 6371  # Earth's radius in km
        
        lat1_rad = radians(lat1)
        lon1_rad = radians(lon1)
        lat2_rad = radians(lat2)
        lon2_rad = radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = sin(dlat/2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        distance = R * c
        
        return distance
    
    def match_donors_to_shelters(self, donors: List[Dict], 
                                shelters: List[Dict]) -> List[Dict]:
        """Match donors to nearest shelters."""
        logger.info(f"Matching {len(donors)} donors to {len(shelters)} shelters")
        
        matches = []
        
        for donor in donors:
            best_matches = []
            
            for shelter in shelters:
                # Calculate distance
                distance = self.calculate_distance(
                    donor['latitude'], donor['longitude'],
                    shelter['latitude'], shelter['longitude']
                )
                
                # Simple matching score (lower distance = higher score)
                if distance < 50:  # 50 km threshold
                    match_score = 1 / (1 + distance)
                    
                    # Consider quantity match
                    quantity_match = min(
                        donor['quantity_kg'] / shelter['daily_requirement_kg'],
                        1.0
                    )
                    
                    final_score = 0.6 * match_score + 0.4 * quantity_match
                    
                    best_matches.append({
                        'donor_id': donor['location_id'],
                        'shelter_id': shelter['center_id'],
                        'distance_km': distance,
                        'score': final_score,
                        'quantity_match': quantity_match
                    })
            
            # Sort by score and take top matches
            best_matches.sort(key=lambda x: x['score'], reverse=True)
            matches.extend(best_matches[:3])  # Top 3 matches
        
        logger.info(f"✓ Generated {len(matches)} donor-shelter matches")
        return sorted(matches, key=lambda x: x['score'], reverse=True)
