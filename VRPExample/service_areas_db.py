#!/usr/bin/env python3
"""
Service Areas Database for Northern Italy

This module provides a comprehensive database of highway service areas and rest stops
across Northern Italy, enabling realistic break planning for heavy truck drivers
who must comply with EU driving time regulations.

Service areas are located on major highways (A1, A4, A6, A7, A8, A9, A10, A11, A12, A13, A14)
and provide facilities for truck drivers including parking, fuel, restaurants, and rest facilities.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math
from vrp_data_models import Location

@dataclass
class ServiceArea:
    """Represents a highway service area suitable for truck breaks."""
    id: str
    name: str
    lat: float
    lon: float
    highway: str  # Highway identifier (e.g., "A1", "A4")
    direction: str  # "North", "South", "East", "West"
    facilities: List[str]  # Available facilities
    truck_parking: bool  # Has truck parking spaces
    fuel: bool  # Has fuel station
    restaurant: bool  # Has restaurant/cafe
    address: str  # Full address

class ServiceAreasDatabase:
    """Database of service areas in Northern Italy for break planning."""
    
    def __init__(self):
        """Initialize the service areas database."""
        self.service_areas: Dict[str, ServiceArea] = {}
        self._populate_service_areas()
    
    def _populate_service_areas(self):
        """Populate the database with real service areas in Northern Italy."""
        
        # A1 Autostrada del Sole (Milan - Bologna section)
        a1_areas = [
            ("SA_A1_Fiorenzuola_N", "Area di Servizio Fiorenzuola Nord", 44.9167, 9.9000, "A1", "North", 
             ["fuel", "restaurant", "shop", "truck_parking"], True, True, True, "A1, 29018 Fiorenzuola d'Arda PC"),
            ("SA_A1_Fiorenzuola_S", "Area di Servizio Fiorenzuola Sud", 44.9100, 9.8950, "A1", "South", 
             ["fuel", "restaurant", "shop", "truck_parking"], True, True, True, "A1, 29018 Fiorenzuola d'Arda PC"),
            ("SA_A1_Secchia_N", "Area di Servizio Secchia Nord", 44.7500, 10.7500, "A1", "North", 
             ["fuel", "restaurant", "truck_parking"], True, True, True, "A1, 42015 Correggio RE"),
            ("SA_A1_Secchia_S", "Area di Servizio Secchia Sud", 44.7450, 10.7450, "A1", "South", 
             ["fuel", "restaurant", "truck_parking"], True, True, True, "A1, 42015 Correggio RE"),
        ]
        
        # A4 Serenissima (Milan - Venice)
        a4_areas = [
            ("SA_A4_Villorba_E", "Area di Servizio Villorba Est", 45.7000, 12.2500, "A4", "East", 
             ["fuel", "restaurant", "shop", "truck_parking"], True, True, True, "A4, 31020 Villorba TV"),
            ("SA_A4_Villorba_W", "Area di Servizio Villorba Ovest", 45.6950, 12.2450, "A4", "West", 
             ["fuel", "restaurant", "shop", "truck_parking"], True, True, True, "A4, 31020 Villorba TV"),
            ("SA_A4_Limenella_E", "Area di Servizio Limenella Est", 45.3000, 11.7500, "A4", "East", 
             ["fuel", "restaurant", "truck_parking"], True, True, True, "A4, 35020 Due Carrare PD"),
            ("SA_A4_Limenella_W", "Area di Servizio Limenella Ovest", 45.2950, 11.7450, "A4", "West", 
             ["fuel", "restaurant", "truck_parking"], True, True, True, "A4, 35020 Due Carrare PD"),
            ("SA_A4_Bergamo_E", "Area di Servizio Bergamo Est", 45.6500, 9.7500, "A4", "East", 
             ["fuel", "restaurant", "shop", "truck_parking"], True, True, True, "A4, 24126 Bergamo BG"),
            ("SA_A4_Bergamo_W", "Area di Servizio Bergamo Ovest", 45.6450, 9.7450, "A4", "West", 
             ["fuel", "restaurant", "shop", "truck_parking"], True, True, True, "A4, 24126 Bergamo BG"),
        ]
        
        # A6 Torino-Savona
        a6_areas = [
            ("SA_A6_Mondovi_N", "Area di Servizio Mondovì Nord", 44.3900, 7.8200, "A6", "North", 
             ["fuel", "restaurant", "truck_parking"], True, True, True, "A6, 12084 Mondovì CN"),
            ("SA_A6_Mondovi_S", "Area di Servizio Mondovì Sud", 44.3850, 7.8150, "A6", "South", 
             ["fuel", "restaurant", "truck_parking"], True, True, True, "A6, 12084 Mondovì CN"),
        ]
        
        # A7 Milano-Genova
        a7_areas = [
            ("SA_A7_Serravalle_N", "Area di Servizio Serravalle Nord", 44.7200, 8.8500, "A7", "North", 
             ["fuel", "restaurant", "shop", "truck_parking"], True, True, True, "A7, 15060 Serravalle Scrivia AL"),
            ("SA_A7_Serravalle_S", "Area di Servizio Serravalle Sud", 44.7150, 8.8450, "A7", "South", 
             ["fuel", "restaurant", "shop", "truck_parking"], True, True, True, "A7, 15060 Serravalle Scrivia AL"),
            ("SA_A7_Dorno", "Area di Servizio Dorno", 45.1500, 8.9200, "A7", "North", 
             ["fuel", "restaurant", "truck_parking"], True, True, True, "A7, 27020 Dorno PV"),
        ]
        
        # A8 Milano-Varese-Laghi
        a8_areas = [
            ("SA_A8_Busto_N", "Area di Servizio Busto Arsizio Nord", 45.6200, 8.8700, "A8", "North", 
             ["fuel", "restaurant", "truck_parking"], True, True, True, "A8, 21052 Busto Arsizio VA"),
            ("SA_A8_Busto_S", "Area di Servizio Busto Arsizio Sud", 45.6150, 8.8650, "A8", "South", 
             ["fuel", "restaurant", "truck_parking"], True, True, True, "A8, 21052 Busto Arsizio VA"),
        ]
        
        # A9 Lainate-Como-Chiasso
        a9_areas = [
            ("SA_A9_Como_N", "Area di Servizio Como Nord", 45.8100, 9.0600, "A9", "North", 
             ["fuel", "restaurant", "truck_parking"], True, True, True, "A9, 22100 Como CO"),
            ("SA_A9_Como_S", "Area di Servizio Como Sud", 45.8050, 9.0550, "A9", "South", 
             ["fuel", "restaurant", "truck_parking"], True, True, True, "A9, 22100 Como CO"),
        ]
        
        # A10 Genova-Savona-Ventimiglia  
        a10_areas = [
            ("SA_A10_Arma_E", "Area di Servizio Arma di Taggia Est", 43.8500, 7.9200, "A10", "East", 
             ["fuel", "restaurant", "truck_parking"], True, True, True, "A10, 18018 Arma di Taggia IM"),
            ("SA_A10_Arma_W", "Area di Servizio Arma di Taggia Ovest", 43.8450, 7.9150, "A10", "West", 
             ["fuel", "restaurant", "truck_parking"], True, True, True, "A10, 18018 Arma di Taggia IM"),
        ]
        
        # A11 Firenze-Mare (northern section)
        a11_areas = [
            ("SA_A11_Montecatini_E", "Area di Servizio Montecatini Est", 43.8800, 10.7700, "A11", "East", 
             ["fuel", "restaurant", "truck_parking"], True, True, True, "A11, 51016 Montecatini Terme PT"),
            ("SA_A11_Montecatini_W", "Area di Servizio Montecatini Ovest", 43.8750, 10.7650, "A11", "West", 
             ["fuel", "restaurant", "truck_parking"], True, True, True, "A11, 51016 Montecatini Terme PT"),
        ]
        
        # A12 Genova-Livorno (northern section)
        a12_areas = [
            ("SA_A12_Versilia_N", "Area di Servizio Versilia Nord", 43.9500, 10.2500, "A12", "North", 
             ["fuel", "restaurant", "truck_parking"], True, True, True, "A12, 55045 Pietrasanta LU"),
            ("SA_A12_Versilia_S", "Area di Servizio Versilia Sud", 43.9450, 10.2450, "A12", "South", 
             ["fuel", "restaurant", "truck_parking"], True, True, True, "A12, 55045 Pietrasanta LU"),
        ]
        
        # A13 Bologna-Padova
        a13_areas = [
            ("SA_A13_Canale_N", "Area di Servizio Canale Nord", 44.8000, 11.4500, "A13", "North", 
             ["fuel", "restaurant", "truck_parking"], True, True, True, "A13, 44058 Canale RE"),
            ("SA_A13_Canale_S", "Area di Servizio Canale Sud", 44.7950, 11.4450, "A13", "South", 
             ["fuel", "restaurant", "truck_parking"], True, True, True, "A13, 44058 Canale RE"),
        ]
        
        # A14 Bologna-Taranto (northern section)
        a14_areas = [
            ("SA_A14_Rubicone_N", "Area di Servizio Rubicone Nord", 44.0500, 12.3500, "A14", "North", 
             ["fuel", "restaurant", "shop", "truck_parking"], True, True, True, "A14, 47039 Savignano sul Rubicone FC"),
            ("SA_A14_Rubicone_S", "Area di Servizio Rubicone Sud", 44.0450, 12.3450, "A14", "South", 
             ["fuel", "restaurant", "shop", "truck_parking"], True, True, True, "A14, 47039 Savignano sul Rubicone FC"),
        ]
        
        # Combine all service areas
        all_areas = a1_areas + a4_areas + a6_areas + a7_areas + a8_areas + a9_areas + a10_areas + a11_areas + a12_areas + a13_areas + a14_areas
        
        # Create ServiceArea objects and add to database
        for area_data in all_areas:
            area_id, name, lat, lon, highway, direction, facilities, truck_parking, fuel, restaurant, address = area_data
            
            service_area = ServiceArea(
                id=area_id,
                name=name,
                lat=lat,
                lon=lon,
                highway=highway,
                direction=direction,
                facilities=facilities,
                truck_parking=truck_parking,
                fuel=fuel,
                restaurant=restaurant,
                address=address
            )
            
            self.service_areas[area_id] = service_area
    
    def find_service_areas_near_route(self, start_lat: float, start_lon: float, 
                                    end_lat: float, end_lon: float, 
                                    max_detour_km: float = 10.0) -> List[ServiceArea]:
        """
        Find service areas along or near a route between two points.
        
        Args:
            start_lat: Starting latitude
            start_lon: Starting longitude  
            end_lat: Ending latitude
            end_lon: Ending longitude
            max_detour_km: Maximum detour distance in km
            
        Returns:
            List of service areas along the route
        """
        route_areas = []
        
        for area in self.service_areas.values():
            # Calculate distance from service area to the route line
            distance_to_route = self._point_to_line_distance(
                area.lat, area.lon, start_lat, start_lon, end_lat, end_lon
            )
            
            if distance_to_route <= max_detour_km:
                route_areas.append(area)
        
        # Sort by distance from start point
        route_areas.sort(key=lambda area: self._haversine_distance(
            start_lat, start_lon, area.lat, area.lon
        ))
        
        return route_areas
    
    def find_nearest_service_area(self, lat: float, lon: float, max_distance_km: float = 50.0) -> Optional[ServiceArea]:
        """
        Find the nearest service area to a given location.
        
        Args:
            lat: Latitude
            lon: Longitude
            max_distance_km: Maximum search distance in km
            
        Returns:
            Nearest service area or None if none found within distance
        """
        nearest_area = None
        min_distance = float('inf')
        
        for area in self.service_areas.values():
            distance = self._haversine_distance(lat, lon, area.lat, area.lon)
            
            if distance < min_distance and distance <= max_distance_km:
                min_distance = distance
                nearest_area = area
        
        return nearest_area
    
    def get_service_areas_by_highway(self, highway: str) -> List[ServiceArea]:
        """Get all service areas on a specific highway."""
        return [area for area in self.service_areas.values() if area.highway == highway]
    
    def find_break_location_between_points(self, start_lat: float, start_lon: float,
                                         end_lat: float, end_lon: float) -> Optional[ServiceArea]:
        """
        Find the best service area for a break between two points.
        Prioritizes areas closest to the midpoint of the route.
        
        Args:
            start_lat: Starting latitude
            start_lon: Starting longitude
            end_lat: Ending latitude  
            end_lon: Ending longitude
            
        Returns:
            Best service area for break or None
        """
        # Calculate midpoint
        mid_lat = (start_lat + end_lat) / 2
        mid_lon = (start_lon + end_lon) / 2
        
        # Find service areas near the route
        route_areas = self.find_service_areas_near_route(start_lat, start_lon, end_lat, end_lon)
        
        if not route_areas:
            return None
        
        # Find the area closest to the midpoint
        best_area = min(route_areas, key=lambda area: self._haversine_distance(
            mid_lat, mid_lon, area.lat, area.lon
        ))
        
        return best_area
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate the haversine distance between two points in km."""
        R = 6371  # Earth's radius in km
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
    def _point_to_line_distance(self, point_lat: float, point_lon: float,
                               line_start_lat: float, line_start_lon: float,
                               line_end_lat: float, line_end_lon: float) -> float:
        """Calculate distance from a point to a line segment in km."""
        # Convert to approximate Cartesian coordinates (good enough for short distances)
        x1, y1 = line_start_lon, line_start_lat
        x2, y2 = line_end_lon, line_end_lat
        x0, y0 = point_lon, point_lat
        
        # Calculate distance using the point-to-line formula
        A = x0 - x1
        B = y0 - y1
        C = x2 - x1
        D = y2 - y1
        
        dot = A * C + B * D
        len_sq = C * C + D * D
        
        if len_sq == 0:
            # Line is actually a point
            return self._haversine_distance(point_lat, point_lon, line_start_lat, line_start_lon)
        
        param = dot / len_sq
        
        if param < 0:
            xx, yy = x1, y1
        elif param > 1:
            xx, yy = x2, y2
        else:
            xx = x1 + param * C
            yy = y1 + param * D
        
        # Convert back to lat/lon and calculate distance
        return self._haversine_distance(point_lat, point_lon, yy, xx)
    
    def get_all_service_areas(self) -> Dict[str, ServiceArea]:
        """Get all service areas in the database."""
        return self.service_areas.copy()
    
    def get_service_area_statistics(self) -> Dict[str, int]:
        """Get statistics about the service areas database."""
        stats = {
            'total_areas': len(self.service_areas),
            'with_truck_parking': sum(1 for area in self.service_areas.values() if area.truck_parking),
            'with_fuel': sum(1 for area in self.service_areas.values() if area.fuel),
            'with_restaurant': sum(1 for area in self.service_areas.values() if area.restaurant),
        }
        
        # Count by highway
        highways = {}
        for area in self.service_areas.values():
            highways[area.highway] = highways.get(area.highway, 0) + 1
        
        stats['by_highway'] = highways
        return stats

# Global instance for easy access
service_areas_db = ServiceAreasDatabase()

# Helper function to convert ServiceArea to Location for VRP integration
def service_area_to_location(service_area: ServiceArea) -> Location:
    """Convert a ServiceArea to a VRP Location object."""
    return Location(
        id=service_area.id,
        x=service_area.lon,  # Use longitude as x
        y=service_area.lat,  # Use latitude as y
        demand=0,  # Service areas have no demand
        service_time=45,  # 45-minute break time
        address=service_area.address,
        lat=service_area.lat,
        lon=service_area.lon
    )

if __name__ == "__main__":
    # Demo/test code
    print("Northern Italy Service Areas Database")
    print("=" * 50)
    
    stats = service_areas_db.get_service_area_statistics()
    print(f"Total service areas: {stats['total_areas']}")
    print(f"With truck parking: {stats['with_truck_parking']}")
    print(f"With fuel: {stats['with_fuel']}")
    print(f"With restaurant: {stats['with_restaurant']}")
    print("\nBy highway:")
    for highway, count in stats['by_highway'].items():
        print(f"  {highway}: {count} areas")
    
    # Test finding service areas near Milan-Bologna route
    print("\nService areas between Milan and Bologna:")
    milan_lat, milan_lon = 45.4642, 9.1896
    bologna_lat, bologna_lon = 44.4949, 11.3426
    
    route_areas = service_areas_db.find_service_areas_near_route(
        milan_lat, milan_lon, bologna_lat, bologna_lon
    )
    
    for area in route_areas[:5]:  # Show first 5
        print(f"  {area.name} ({area.highway}) - {area.address}")
