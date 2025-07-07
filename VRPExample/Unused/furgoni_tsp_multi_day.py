#!/usr/bin/env python3
"""
Furgoni scenario adapted for TSP multi-day solver format.
This creates a TSP-compatible version of the furgoni delivery scenario.
"""

import math
import random
from typing import List, Tuple, Dict

# Furgoni scenario data adapted from vrp_scenarios.py
class FurgoniTSPData:
    """Furgoni delivery data in TSP multi-day solver format."""
    
    def __init__(self):
        """Initialize the furgoni scenario data."""
        # Depot location (Milan coordinates)
        self.depot_location = (44.5404, 8.1407, "Asti Distribution Center")
        
        # Delivery and pickup locations with time windows from Excel
        # Format: (location_id, address, lat, lon, tw_start_hours, tw_end_hours, service_time_minutes, is_pickup)
        self.locations_data = [
            # Depot (index 0)
            ("depot", "Asti Distribution Center", 44.5404, 8.1407, 0, 24, 5, False),
            # End depot (index 1) 
            ("depot_end", "Asti Distribution Center", 44.5404, 8.1407, 0, 24, 5, False),
            
            # Delivery locations (depots to destinations)
            ("schonaich_de", "gentile gusto carl Zeiss Str. 4 71101 Schonaich DE", 48.6667, 9.0000, 0, 24, 30, False),
            ("malmoe_sweden", "malmoe sweden menarini diag.slr", 55.6050, 13.0038, 0, 24, 30, False),
            ("salvazzano_pd", "Via Pelosa 20 35030 Salvazzano Dentro PD", 45.3833, 11.7833, 0, 24, 20, False),
            ("badia_polesine_ro", "Via L. Da Vinci 537 45021 Badia Polesine RO", 45.1167, 11.5000, 0, 24, 20, False),
            ("badia_polesine_ro_2", "Via L. Da Vinci 537 45021 Badia Polesine RO", 45.1167, 11.5000, 0, 24, 20, False),
            ("marostica_vi", "via Milano 1 36063 Marostica VI", 45.7500, 11.6500, 0, 24, 20, False),
            ("st_martin_crau", "Rue Gay Lussac 3 St. Martin de Crau 13310", 43.6333, 4.8000, 0, 24, 30, False),
            ("sant_olcese_ge", "Piazza Guglielmo Marconi 40 16010 Sant'Olcese GE", 44.5167, 8.9333, 0, 24, 45, False),
            ("casarza_ligure_ge", "Via Tangoni 30/32 I-16030 Casarza Ligure GE", 44.2833, 9.4667, 0, 24, 45, False),
            ("quinto_do_anjo", "Quinta De Marquesa 1CCI Portoga 10218 29050-557 Quinto Do Anjo", 38.5667, -8.9000, 0, 24, 30, False),
            ("quinto_do_anjo_2", "Quinta De Marquesa 1CCI Portoga 10218 29050-557 Quinto Do Anjo", 38.5667, -8.9000, 0, 24, 30, False),
            ("paris_fr", "Rue de L'Abbaye 10 PARIGI FR", 48.8566, 2.3522, 0, 24, 30, False),
            ("chiva_spagna", "Poligono I La Pamilla 196 46370 Chiva Spagna", 39.4667, -0.7167, 0, 24, 45, False),
            ("cerro_al_lambro_mi", "Via Autosole 7 20070 Cerro Al Lambro MI", 45.3333, 9.3000, 0, 24, 20, False),
            ("cormano_mi", "via dell'Artigianato 1 20032 Cormano MI", 45.5333, 9.1667, 0, 24, 20, False),
            ("ferno_va", "superstrada per Malpensa uscita Cargocity Torre D 5 piano 21010 Ferno VA", 45.6167, 8.7167, 0, 8.5, 15, False),  # Early morning pickup
            ("imperia_1", "via Filippo Airenti 2 18100 Imperia IM", 43.8833, 8.0333, 0, 24, 20, False),
            ("imperia_2", "via Nazionale 356/3 18100 Imperia IM", 43.8833, 8.0333, 0, 24, 20, False),
            ("villar_cuneo", "via I Maggio 43 Cuneo 12020 Villar San Costanzo", 44.3833, 7.5333, 0, 24, 20, False),
            ("villar_cuneo_2", "via I Maggio 43 Cuneo 12020 Villar San Costanzo", 44.3833, 7.5333, 0, 24, 20, False),
            ("castellalfero_at", "Via Statale 25/A Castell'Alfero 14033 AT", 44.9667, 8.2167, 0, 24, 15, False),
            ("osimo_an", "via Francesco Crispi 2 60027 Osimo AN", 43.4833, 13.4833, 0, 24, 20, False),
            ("castelfidardo", "via Jesina 27/P 60022 Castelfidardo", 43.4667, 13.5500, 0, 24, 15, False),
            ("somaglia_lo", "strada Provinciale 223 26867 Località Cantonale Somaglia LO", 45.1667, 9.6667, 0, 24, 20, False),
            ("capriate_bg", "via Bergamo 61/63 24042 Capriate San Gervasio BG", 45.6000, 9.5333, 0, 24, 20, False),
            ("cazzano_bg", "via Cavalier Pietro Radici 19 24026 Cazzano Sant'andrea BG", 45.7500, 9.8000, 0, 24, 15, False),
        ]
        
        # Build coordinate lookup
        self.coordinates = {}
        self.service_times = {}
        self.time_windows = {}
        
        for i, (loc_id, address, lat, lon, tw_start, tw_end, service_time, is_pickup) in enumerate(self.locations_data):
            self.coordinates[i] = (lat, lon)
            self.service_times[i] = service_time * 60  # Convert to seconds
            self.time_windows[i] = (tw_start * 3600, tw_end * 3600)  # Convert to seconds
        
        # Pre-calculate distance matrix using haversine distance
        self.distance_matrix = self._calculate_distance_matrix()
        
        print(f"🚚 Furgoni TSP Multi-Day Scenario Created:")
        print(f"   - {len(self.locations_data)} locations (including depot start/end)")
        print(f"   - Distance matrix: {len(self.distance_matrix)}x{len(self.distance_matrix[0])}")
        print(f"   - Service times: {min(self.service_times.values())//60}-{max(self.service_times.values())//60} minutes")
        print(f"   - Coverage: Europe-wide delivery network")
    
    def _calculate_distance_matrix(self) -> List[List[float]]:
        """Calculate distance matrix using haversine formula."""
        n = len(self.locations_data)
        matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = 0.0
                else:
                    lat1, lon1 = self.coordinates[i]
                    lat2, lon2 = self.coordinates[j]
                    distance_km = self._haversine_distance(lat1, lon1, lat2, lon2)
                    # Convert to travel time in seconds (assuming 80 km/h average speed)
                    travel_time_seconds = (distance_km / 80.0) * 3600
                    matrix[i][j] = travel_time_seconds
        
        return matrix
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance in kilometers."""
        R = 6371  # Earth's radius in kilometers
        
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c

# Global instance for the TSP solver interface
_furgoni_data = None

def get_furgoni_data():
    """Get or create the furgoni data instance."""
    global _furgoni_data
    if _furgoni_data is None:
        _furgoni_data = FurgoniTSPData()
    return _furgoni_data

# TSP Multi-Day Solver Interface Functions
def num_nodes():
    """Return the number of nodes in the problem."""
    data = get_furgoni_data()
    return len(data.locations_data)

def transit_callback(manager, day_end, night_nodes, morning_nodes, from_index, to_index):
    """Calculate transit cost between two nodes.
    
    This is the cost function used by OR-Tools for optimization.
    For TSP, this is typically the travel time or distance.
    """
    data = get_furgoni_data()
    
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    
    # Handle special virtual nodes (night/morning)
    if from_node in night_nodes or from_node in morning_nodes:
        if to_node in night_nodes or to_node in morning_nodes:
            return 0  # No cost between virtual nodes
        else:
            # Virtual to real node - use depot as reference
            from_node = 0  # depot
    
    if to_node in night_nodes or to_node in morning_nodes:
        # Real to virtual node - use depot as reference
        to_node = 0  # depot
    
    # Ensure nodes are within bounds
    if from_node >= len(data.distance_matrix) or to_node >= len(data.distance_matrix):
        return 0
    
    # Return travel time in seconds
    travel_time = data.distance_matrix[from_node][to_node]
    
    # Add a small cost to encourage visiting nodes
    base_cost = int(travel_time)
    return base_cost

def time_callback(manager, node_service_time, overnight_time, night_nodes, morning_nodes, from_index, to_index):
    """Calculate time taken to go from one node to another.
    
    This includes travel time and service time.
    """
    data = get_furgoni_data()
    
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    
    # Handle virtual nodes (night/morning)
    if from_node in night_nodes:
        if to_node in morning_nodes:
            # Night to morning transition - represents overnight stay
            return overnight_time
        else:
            # Night to regular node - start from depot location
            from_node = 0
    
    if from_node in morning_nodes:
        # Morning node to any other node - start from depot
        from_node = 0
    
    if to_node in night_nodes or to_node in morning_nodes:
        # Going to virtual node - use depot as destination
        to_node = 0
    
    # Ensure nodes are within bounds
    if from_node >= len(data.distance_matrix) or to_node >= len(data.distance_matrix):
        return 0
    
    # Travel time between real locations
    travel_time = data.distance_matrix[from_node][to_node]
    
    # Add service time at destination (unless it's depot or virtual node)
    service_time = 0
    if to_node not in [0, 1] and to_node not in night_nodes and to_node not in morning_nodes:
        service_time = data.service_times.get(to_node, node_service_time)
    
    total_time = int(travel_time + service_time)
    return total_time

def get_location_info(node_id: int) -> Dict:
    """Get information about a specific location."""
    data = get_furgoni_data()
    if 0 <= node_id < len(data.locations_data):
        loc_data = data.locations_data[node_id]
        return {
            'id': loc_data[0],
            'address': loc_data[1],
            'lat': loc_data[2],
            'lon': loc_data[3],
            'time_window': data.time_windows[node_id],
            'service_time': data.service_times[node_id]
        }
    return {}

def print_scenario_summary():
    """Print a summary of the furgoni scenario."""
    data = get_furgoni_data()
    
    print("\n" + "="*60)
    print("🚚 FURGONI TSP MULTI-DAY SCENARIO SUMMARY")
    print("="*60)
    
    print(f"📍 Locations: {len(data.locations_data)}")
    print(f"   - Depot: {data.locations_data[0][1]}")
    print(f"   - Delivery destinations: {len(data.locations_data) - 2}")
    
    print(f"\n🌍 Geographic Coverage:")
    countries = set()
    for loc_data in data.locations_data[2:]:  # Skip depot start/end
        if 'DE' in loc_data[1] or 'Germany' in loc_data[1]:
            countries.add('Germany')
        elif 'sweden' in loc_data[1].lower():
            countries.add('Sweden')
        elif 'FR' in loc_data[1] or 'PARIGI' in loc_data[1]:
            countries.add('France')
        elif 'Spagna' in loc_data[1]:
            countries.add('Spain')
        elif 'Portoga' in loc_data[1]:
            countries.add('Portugal')
        else:
            countries.add('Italy')
    
    for country in sorted(countries):
        country_locations = []
        for loc_data in data.locations_data[2:]:
            if country == 'Germany' and ('DE' in loc_data[1] or 'Germany' in loc_data[1]):
                country_locations.append(loc_data[0])
            elif country == 'Sweden' and 'sweden' in loc_data[1].lower():
                country_locations.append(loc_data[0])
            elif country == 'France' and ('FR' in loc_data[1] or 'PARIGI' in loc_data[1]):
                country_locations.append(loc_data[0])
            elif country == 'Spain' and 'Spagna' in loc_data[1]:
                country_locations.append(loc_data[0])
            elif country == 'Portugal' and 'Portoga' in loc_data[1]:
                country_locations.append(loc_data[0])
            elif country == 'Italy' and not any(x in loc_data[1] for x in ['DE', 'sweden', 'FR', 'PARIGI', 'Spagna', 'Portoga']):
                country_locations.append(loc_data[0])
        
        if country_locations:
            print(f"   - {country}: {len(country_locations)} locations")
    
    print(f"\n⏰ Service Times:")
    service_times_minutes = [t//60 for t in data.service_times.values()]
    print(f"   - Range: {min(service_times_minutes)}-{max(service_times_minutes)} minutes")
    print(f"   - Average: {sum(service_times_minutes)/len(service_times_minutes):.1f} minutes")
    
    print(f"\n🚗 Travel Estimates (based on distance):")
    # Calculate some sample distances
    depot_to_farthest = 0
    farthest_location = ""
    for i, loc_data in enumerate(data.locations_data[2:], 2):
        distance = data.distance_matrix[0][i] / 3600  # Convert seconds to hours
        if distance > depot_to_farthest:
            depot_to_farthest = distance
            farthest_location = loc_data[0]
    
    print(f"   - Farthest from depot: {farthest_location} (~{depot_to_farthest:.1f} hours)")
    print(f"   - Multi-day scheduling recommended for this scenario")
    
    print("="*60)

if __name__ == "__main__":
    # Test the interface
    print("Testing Furgoni TSP Multi-Day Interface...")
    print_scenario_summary()
    
    print(f"\nInterface Test:")
    print(f"  num_nodes(): {num_nodes()}")
    
    # Test some location info
    print(f"\nSample locations:")
    for i in [0, 1, 2, -1]:
        if i == -1:
            i = num_nodes() - 1
        info = get_location_info(i)
        print(f"  Node {i}: {info.get('id', 'N/A')} - {info.get('address', 'N/A')[:50]}...")
    
    print(f"\n✅ Furgoni TSP interface ready for multi-day solver!")
