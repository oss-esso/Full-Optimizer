#!/usr/bin/env python3
"""
Hybrid Travel Time Calculator - Fast Solving with Real-World Correction

This approach:
1. Uses Haversine distance for fast optimization solving
2. Samples a few real OSRM routes to calculate a correction factor
3. Applies the correction factor to estimate realistic travel times
4. Provides GPS map with real routing for visualization
"""

import requests
import time
import logging
import math
import random
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class HybridTravelCalculator:
    """Calculate travel times using Haversine + OSRM correction factor."""
    
    def __init__(self, osrm_base_url: str = "https://router.project-osrm.org"):
        self.osrm_base_url = osrm_base_url
        self.correction_factor = 1.5  # Default assumption: real roads are 50% longer
        self.sample_routes_calculated = 0
        
    def calculate_haversine_distance(self, from_coords: Tuple[float, float], to_coords: Tuple[float, float]) -> float:
        """Calculate Haversine distance in kilometers."""
        if from_coords == to_coords:
            return 0.0
            
        lat1, lon1 = math.radians(from_coords[0]), math.radians(from_coords[1])
        lat2, lon2 = math.radians(to_coords[0]), math.radians(to_coords[1])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        distance_km = 6371 * 2 * math.asin(math.sqrt(a))  # Earth radius in km
        return distance_km
    
    def sample_real_routes_for_correction(self, coordinates: List[Tuple[float, float]], max_samples: int = 10) -> float:
        """Sample a few real OSRM routes to calculate a correction factor."""
        logger.info(f"🔍 Sampling {max_samples} real routes to estimate correction factor...")
        
        n = len(coordinates)
        if n < 2:
            return 1.5  # Default factor
        
        # Select random pairs, but ensure we get diverse samples
        sample_pairs = []
        
        # Always include depot to some random locations
        depot_idx = 0
        for _ in range(min(3, max_samples // 2)):
            other_idx = random.randint(1, n-1)
            sample_pairs.append((depot_idx, other_idx))
        
        # Add some random pairs
        for _ in range(max_samples - len(sample_pairs)):
            i, j = random.sample(range(n), 2)
            sample_pairs.append((i, j))
        
        # Remove duplicates
        sample_pairs = list(set(sample_pairs))
        
        total_haversine_time = 0
        total_osrm_time = 0
        successful_samples = 0
        
        for i, j in sample_pairs:
            from_coords = coordinates[i]
            to_coords = coordinates[j]
            
            # Calculate Haversine time (assuming 40 km/h average speed)
            haversine_dist = self.calculate_haversine_distance(from_coords, to_coords)
            haversine_time = haversine_dist * 60 / 40  # minutes at 40 km/h
            
            if haversine_time == 0:
                continue
                
            # Get real OSRM time
            try:
                osrm_time = self._get_osrm_time_single(from_coords, to_coords)
                if osrm_time > 0:
                    total_haversine_time += haversine_time
                    total_osrm_time += osrm_time
                    successful_samples += 1
                    logger.info(f"  Sample {successful_samples}: {haversine_time:.1f}min → {osrm_time:.1f}min (factor: {osrm_time/haversine_time:.2f})")
                    
                    # Add small delay to avoid overwhelming OSRM
                    time.sleep(0.2)
                    
            except Exception as e:
                logger.warning(f"Failed to get OSRM time for sample: {e}")
                continue
        
        if successful_samples >= 3:
            correction_factor = total_osrm_time / total_haversine_time
            logger.info(f"✅ Calculated correction factor from {successful_samples} samples: {correction_factor:.2f}")
            logger.info(f"   Real roads are {(correction_factor-1)*100:.0f}% longer than straight-line distances")
        else:
            correction_factor = 1.5  # Conservative default
            logger.warning(f"⚠️  Only {successful_samples} samples successful, using default factor: {correction_factor}")
        
        self.correction_factor = correction_factor
        self.sample_routes_calculated = successful_samples
        return correction_factor
    
    def _get_osrm_time_single(self, from_coords: Tuple[float, float], to_coords: Tuple[float, float]) -> float:
        """Get OSRM travel time for a single route (used for sampling)."""
        from_lon, from_lat = from_coords[1], from_coords[0]
        to_lon, to_lat = to_coords[1], to_coords[0]
        
        url = f"{self.osrm_base_url}/route/v1/driving/{from_lon},{from_lat};{to_lon},{to_lat}"
        params = {
            'overview': 'false',
            'alternatives': 'false',
            'steps': 'false'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if data['code'] == 'Ok' and 'routes' in data and len(data['routes']) > 0:
            duration_seconds = data['routes'][0]['duration']
            return duration_seconds / 60  # Convert to minutes
        else:
            raise Exception(f"OSRM error: {data.get('message', 'Unknown error')}")
    
    def get_corrected_travel_time_matrix(self, coordinates: List[Tuple[float, float]]) -> List[List[int]]:
        """
        Get travel time matrix using Haversine + correction factor approach.
        Much faster than full OSRM matrix calculation.
        """
        n = len(coordinates)
        logger.info(f"🚀 Building hybrid travel time matrix for {n} locations...")
        
        # Step 1: Sample real routes to get correction factor
        correction_factor = self.sample_real_routes_for_correction(coordinates, max_samples=8)
        
        # Step 2: Build matrix using Haversine + correction factor
        matrix = [[0 for _ in range(n)] for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = 0
                else:
                    # Calculate Haversine distance
                    distance_km = self.calculate_haversine_distance(coordinates[i], coordinates[j])
                    
                    # Convert to time with realistic speed (40 km/h urban average)
                    base_time = distance_km * 60 / 40  # minutes
                    
                    # Apply correction factor for real road conditions
                    corrected_time = base_time * correction_factor
                    
                    # Ensure minimum 1 minute for non-zero distances
                    matrix[i][j] = max(1, int(corrected_time)) if distance_km > 0 else 0
        
        logger.info(f"✅ Hybrid travel time matrix built with correction factor {correction_factor:.2f}")
        self.print_matrix_summary(matrix, coordinates)
        
        return matrix
    
    def print_matrix_summary(self, matrix: List[List[int]], coordinates: List[Tuple[float, float]]):
        """Print a summary of the travel time matrix."""
        n = len(matrix)
        total_time = sum(sum(row) for row in matrix)
        non_zero_count = sum(1 for i in range(n) for j in range(n) if matrix[i][j] > 0)
        
        if non_zero_count > 0:
            avg_time = total_time / non_zero_count
            max_time = max(max(row) for row in matrix)
            min_time = min(min(val for val in row if val > 0) for row in matrix if any(val > 0 for val in row))
            
            logger.info(f"📊 Hybrid Travel Time Matrix Summary:")
            logger.info(f"  Locations: {n}")
            logger.info(f"  Total routes: {n * (n-1)}")
            logger.info(f"  Average travel time: {avg_time:.1f} minutes")
            logger.info(f"  Min travel time: {min_time} minutes")
            logger.info(f"  Max travel time: {max_time} minutes")
            logger.info(f"  Correction factor applied: {self.correction_factor:.2f}")
            logger.info(f"  Based on {self.sample_routes_calculated} real route samples")
        else:
            logger.info("Travel time matrix is empty")

# Global instance for easy access
hybrid_calculator = HybridTravelCalculator()
