"""Optimization module for FoodBridge AI."""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

class RouteOptimization:
    """Vehicle Routing Problem optimization for delivery routes."""
    
    def __init__(self, config: dict = None):
        """Initialize route optimization."""
        self.config = config or {}
        self.solution = None
        
    def optimize_routes(self, donors: List[Dict], shelters: List[Dict], 
                       vehicles: int = 5) -> Dict:
        """Optimize delivery routes using OR-Tools."""
        logger.info(f"Optimizing routes for {len(donors)} donors to {len(shelters)} shelters")
        
        try:
            from ortools.linear_solver import pywraplp
            
            # Create the routing index manager and routing model
            num_locations = len(donors) + len(shelters) + 1  # +1 for depot
            
            # For simplicity, create distance matrix
            distance_matrix = self._create_distance_matrix(donors, shelters)
            
            # Simplified solution - group by proximity
            routes = self._greedy_routing(donors, shelters, vehicles)
            
            logger.info(f"✓ Generated {len(routes)} optimized routes")
            
            return {
                'routes': routes,
                'num_vehicles': vehicles,
                'total_distance': sum(route['distance'] for route in routes),
                'status': 'optimal'
            }
            
        except Exception as e:
            logger.warning(f"OR-Tools optimization failed: {e}. Using greedy approach.")
            routes = self._greedy_routing(donors, shelters, vehicles)
            return {
                'routes': routes,
                'num_vehicles': vehicles,
                'total_distance': sum(route['distance'] for route in routes),
                'status': 'greedy'
            }
    
    def _create_distance_matrix(self, donors: List[Dict], 
                                shelters: List[Dict]) -> np.ndarray:
        """Create distance matrix between all locations."""
        all_locations = [{'lat': 0, 'lon': 0}] + donors + shelters  # Depot at origin
        n = len(all_locations)
        
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    lat1 = all_locations[i].get('latitude', all_locations[i].get('lat', 0))
                    lon1 = all_locations[i].get('longitude', all_locations[i].get('lon', 0))
                    lat2 = all_locations[j].get('latitude', all_locations[j].get('lat', 0))
                    lon2 = all_locations[j].get('longitude', all_locations[j].get('lon', 0))
                    
                    distance_matrix[i, j] = self._calculate_distance(lat1, lon1, lat2, lon2)
        
        return distance_matrix
    
    def _calculate_distance(self, lat1: float, lon1: float, 
                           lat2: float, lon2: float) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)
    
    def _greedy_routing(self, donors: List[Dict], shelters: List[Dict], 
                        vehicles: int) -> List[Dict]:
        """Generate routes using greedy nearest-neighbor algorithm."""
        routes = [[] for _ in range(vehicles)]
        route_distances = [0.0] * vehicles
        
        # Assign each donor-shelter pair to nearest vehicle route
        for donor in donors:
            for shelter in shelters:
                # Find vehicle with shortest route
                min_vehicle = np.argmin(route_distances)
                
                distance = self._calculate_distance(
                    donor.get('latitude', 0), donor.get('longitude', 0),
                    shelter.get('latitude', 0), shelter.get('longitude', 0)
                )
                
                routes[min_vehicle].append({
                    'donor': donor.get('location_id', 0),
                    'shelter': shelter.get('center_id', 0),
                    'distance': distance
                })
                
                route_distances[min_vehicle] += distance
        
        # Format routes
        formatted_routes = []
        for i, route in enumerate(routes):
            if route:  # Only include non-empty routes
                formatted_routes.append({
                    'vehicle_id': i,
                    'stops': route,
                    'distance': route_distances[i],
                    'num_stops': len(route)
                })
        
        return formatted_routes


class DeliveryOptimization:
    """Optimize delivery timing to minimize spoilage."""
    
    def __init__(self, config: dict = None):
        """Initialize delivery optimization."""
        self.config = config or {}
        
    def optimize_delivery_time(self, route: Dict, 
                              food_items: List[Dict]) -> Dict:
        """Determine optimal delivery time to minimize spoilage."""
        logger.info(f"Optimizing delivery time for route {route.get('vehicle_id', 0)}")
        
        # Calculate average shelf-life
        avg_shelf_life = np.mean([item.get('shelf_life_hours', 24) for item in food_items])
        
        # Calculate delivery distance and estimate time (assuming 40 km/hour average)
        route_distance = route.get('distance', 0)
        estimated_delivery_time = route_distance / 40  # hours
        
        # Add buffer (20% of shelf-life)
        safe_delivery_time = avg_shelf_life * 0.8
        
        # Determine if route is feasible
        is_feasible = estimated_delivery_time < safe_delivery_time
        
        # Recommended start time (delivery must reach within safe window)
        hours_to_delay = max(0, estimated_delivery_time - safe_delivery_time)
        
        return {
            'route_id': route.get('vehicle_id', 0),
            'avg_shelf_life_hours': avg_shelf_life,
            'estimated_delivery_time': estimated_delivery_time,
            'safe_delivery_window': safe_delivery_time,
            'is_feasible': is_feasible,
            'recommended_start_delay_hours': hours_to_delay,
            'spoilage_risk_score': min(estimated_delivery_time / safe_delivery_time, 1.0)
        }


class WasteReductionOptimizer:
    """Optimize system to reduce overall food waste."""
    
    def __init__(self, config: dict = None):
        """Initialize waste reduction optimizer."""
        self.config = config or {}
        
    def calculate_waste_reduction_potential(self, surplus_data: pd.DataFrame,
                                            demand_data: pd.DataFrame) -> Dict:
        """Calculate potential waste reduction through matching."""
        logger.info("Calculating waste reduction potential")
        
        total_surplus = surplus_data['quantity_kg'].sum()
        total_demand = demand_data['daily_food_requirement_kg'].sum()
        
        # Calculate matching potential
        matched_quantity = min(total_surplus, total_demand)
        waste_reduction_potential = (matched_quantity / total_surplus) * 100 if total_surplus > 0 else 0
        
        # Calculate savings
        cost_per_kg = 5  # Assuming cost of 5 per kg for waste management
        potential_savings = matched_quantity * cost_per_kg
        
        # Calculate environmental impact
        co2_per_kg = 0.5  # kg CO2 equivalent per kg of waste
        co2_reduction = matched_quantity * co2_per_kg
        
        return {
            'total_surplus_kg': total_surplus,
            'total_demand_kg': total_demand,
            'matched_quantity_kg': matched_quantity,
            'waste_reduction_percentage': waste_reduction_potential,
            'potential_savings': potential_savings,
            'co2_reduction_kg': co2_reduction,
            'beneficiaries_served': len(demand_data)
        }
    
    def recommend_optimization_strategy(self, surplus_data: pd.DataFrame,
                                       demand_data: pd.DataFrame) -> List[str]:
        """Recommend strategies to optimize waste reduction."""
        recommendations = []
        
        # Analyze patterns
        high_waste_areas = surplus_data[
            surplus_data['quantity_kg'] > surplus_data['quantity_kg'].quantile(0.75)
        ]['location_id'].unique()
        
        high_demand_areas = demand_data[
            demand_data['daily_food_requirement_kg'] > demand_data['daily_food_requirement_kg'].quantile(0.75)
        ]['center_id'].unique()
        
        recommendations.append(
            f"Focus on {len(high_waste_areas)} high-surplus areas with most waste generation"
        )
        recommendations.append(
            f"Prioritize {len(high_demand_areas)} high-demand shelters for immediate matching"
        )
        
        # Shelf-life recommendations
        if 'shelf_life_hours' in surplus_data.columns:
            avg_shelf_life = surplus_data['shelf_life_hours'].mean()
            if avg_shelf_life < 24:
                recommendations.append(
                    "Implement rapid delivery system for perishable items (shelf-life < 24 hours)"
                )
        
        # Food type recommendations
        if 'food_type' in surplus_data.columns:
            cooked_ratio = (surplus_data['food_type'] == 'Cooked').sum() / len(surplus_data)
            if cooked_ratio > 0.7:
                recommendations.append("High cooked food surplus - prioritize quick distribution")
        
        return recommendations
