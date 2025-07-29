"""
Coordinate Validation Test

This script validates geocoded coordinates from the EPDT system by:
1. Loading all geocoded coordinates from the scenario
2. Creating a visual map of coordinates
3. Cross-referencing coordinates with web sources
4. Identifying potentially incorrect coordinates

Usage:
    python test_coordinate_validation.py
"""

import os
import sys
import json
from typing import Dict, List, Tuple, Optional

# Add necessary paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
heuristic_root = os.path.join(current_dir, '..')
src_dir = os.path.join(heuristic_root, 'src')
utils_dir = os.path.join(heuristic_root, 'utils')
sys.path.insert(0, heuristic_root)
sys.path.insert(0, src_dir)
sys.path.insert(0, utils_dir)

# Try to import folium for map creation
try:
    import folium
    FOLIUM_AVAILABLE = True
    print("✅ Folium available for map generation")
except ImportError:
    FOLIUM_AVAILABLE = False
    print("⚠️  Folium not available - will output coordinates only")

def load_geocoded_coordinates():
    """Load geocoded coordinates from the scenario."""
    try:
        # Try multiple import approaches
        try:
            from scenario_creator import create_scenario_from_excel
        except ImportError:
            # Try from src directory
            sys.path.insert(0, os.path.join(heuristic_root, 'src'))
            from scenario_creator import create_scenario_from_excel
        
        excel_path = os.path.join(src_dir, 'furgoni.xlsx')
        print(f"📁 Loading coordinates from: {excel_path}")
        
        orders, vehicles, drivers = create_scenario_from_excel(excel_path)
        
        coordinates = []
        depot_coords = None
        
        print(f"Debug: Found {len(orders)} orders and {len(vehicles)} vehicles")
        
        # Extract depot coordinates
        # Since vehicles don't have depot_location, we'll need to find the depot from tasks
        # Let's check if we can find a depot task or use a default depot location
        for i, vehicle in enumerate(vehicles[:3]):  # Just first 3 for debugging
            print(f"Debug: Vehicle {i} has attributes: {[attr for attr in dir(vehicle) if not attr.startswith('_')]}")
            if hasattr(vehicle, 'depot_id'):
                print(f"Debug: Vehicle depot_id: {vehicle.depot_id}")
                # We'll set a default depot location for now, or try to find it from tasks
                # For now, let's see if we can extract it from the geocoding cache or use a fallback
                break
        
        # Extract order coordinates
        for i, order in enumerate(orders[:3]):  # Just first 3 for debugging
            print(f"Debug: Order {i} has attributes: {[attr for attr in dir(order) if not attr.startswith('_')]}")
            
            # Try to find the order ID attribute
            order_id = getattr(order, 'id', getattr(order, 'order_id', f'order_{i}'))
            print(f"Debug: Order ID: {order_id}")
            
            # Check pickup tasks
            if hasattr(order, 'pickup_tasks') and order.pickup_tasks:
                print(f"Debug: Order has {len(order.pickup_tasks)} pickup tasks")
                for task in order.pickup_tasks:
                    print(f"Debug: Pickup task has attributes: {[attr for attr in dir(task) if not attr.startswith('_')]}")
                    if hasattr(task, 'lat') and hasattr(task, 'lon'):
                        print(f"Debug: Pickup coordinates: ({task.lat}, {task.lon}), location_id: {task.location_id}")
                        coordinates.append({
                            'name': f'PICKUP_{order_id}',
                            'address': str(task.location_id),
                            'lat': task.lat,
                            'lon': task.lon,
                            'type': 'pickup',
                            'order_id': order_id
                        })
            
            # Check delivery tasks
            if hasattr(order, 'delivery_tasks') and order.delivery_tasks:
                print(f"Debug: Order has {len(order.delivery_tasks)} delivery tasks")
                for task in order.delivery_tasks:
                    print(f"Debug: Delivery task has attributes: {[attr for attr in dir(task) if not attr.startswith('_')]}")
                    if hasattr(task, 'lat') and hasattr(task, 'lon'):
                        print(f"Debug: Delivery coordinates: ({task.lat}, {task.lon}), location_id: {task.location_id}")
                        coordinates.append({
                            'name': f'DELIVERY_{order_id}',
                            'address': str(task.location_id),
                            'lat': task.lat,
                            'lon': task.lon,
                            'type': 'delivery',
                            'order_id': order_id
                        })
        
        # Add all orders if we found some coordinates from the first 3
        if coordinates:
            coordinates = []  # Reset and do all orders
            for i, order in enumerate(orders):
                order_id = getattr(order, 'id', getattr(order, 'order_id', f'order_{i}'))
                
                # Check pickup tasks
                if hasattr(order, 'pickup_tasks') and order.pickup_tasks:
                    for task in order.pickup_tasks:
                        if hasattr(task, 'lat') and hasattr(task, 'lon'):
                            coordinates.append({
                                'name': f'PICKUP_{order_id}',
                                'address': str(task.location_id),
                                'lat': task.lat,
                                'lon': task.lon,
                                'type': 'pickup',
                                'order_id': order_id
                            })
                
                # Check delivery tasks
                if hasattr(order, 'delivery_tasks') and order.delivery_tasks:
                    for task in order.delivery_tasks:
                        if hasattr(task, 'lat') and hasattr(task, 'lon'):
                            coordinates.append({
                                'name': f'DELIVERY_{order_id}',
                                'address': str(task.location_id),
                                'lat': task.lat,
                                'lon': task.lon,
                                'type': 'delivery',
                                'order_id': order_id
                            })
        
        print(f"Debug: Extracted {len(coordinates)} coordinates")
        return coordinates, depot_coords
        
    except Exception as e:
        print(f"❌ Error loading coordinates: {e}")
        import traceback
        traceback.print_exc()
        return [], None

def validate_coordinate_bounds(coordinates: List[Dict]) -> List[Dict]:
    """Validate that coordinates are within reasonable bounds for Italy/Europe."""
    invalid_coords = []
    
    # Rough bounds for Italy and surrounding areas
    ITALY_BOUNDS = {
        'lat_min': 35.0,  # Southern Sicily
        'lat_max': 47.0,  # Northern Alps
        'lon_min': 6.0,   # Western borders
        'lon_max': 19.0   # Eastern borders
    }
    
    for coord in coordinates:
        lat, lon = coord['lat'], coord['lon']
        
        issues = []
        if lat < ITALY_BOUNDS['lat_min'] or lat > ITALY_BOUNDS['lat_max']:
            issues.append(f"Latitude {lat} outside Italy bounds ({ITALY_BOUNDS['lat_min']}-{ITALY_BOUNDS['lat_max']})")
        
        if lon < ITALY_BOUNDS['lon_min'] or lon > ITALY_BOUNDS['lon_max']:
            issues.append(f"Longitude {lon} outside Italy bounds ({ITALY_BOUNDS['lon_min']}-{ITALY_BOUNDS['lon_max']})")
        
        # Check for obviously wrong coordinates (0,0 or very precise repeating decimals)
        if lat == 0.0 and lon == 0.0:
            issues.append("Coordinates are (0,0) - likely geocoding failed")
        
        if issues:
            invalid_coords.append({
                **coord,
                'issues': issues
            })
    
    return invalid_coords

def create_coordinate_map(coordinates: List[Dict], depot_coords: Dict, output_path: str):
    """Create an interactive map with all coordinates."""
    if not FOLIUM_AVAILABLE:
        print("⚠️  Cannot create map - folium not available")
        return False
    
    # Calculate center point (roughly central Italy)
    if depot_coords:
        center_lat = depot_coords['lat']
        center_lon = depot_coords['lon']
    else:
        center_lat = 42.0  # Central Italy
        center_lon = 12.0
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles='OpenStreetMap'
    )
    
    # Add depot
    if depot_coords:
        folium.Marker(
            [depot_coords['lat'], depot_coords['lon']],
            popup=f"🏢 {depot_coords['name']}",
            tooltip=depot_coords['address'],
            icon=folium.Icon(color='red', icon='home')
        ).add_to(m)
    
    # Add coordinates with different colors for different types
    color_map = {
        'pickup': 'blue',
        'delivery': 'green',
        'depot': 'red'
    }
    
    icon_map = {
        'pickup': 'arrow-up',
        'delivery': 'arrow-down',
        'depot': 'home'
    }
    
    for coord in coordinates:
        color = color_map.get(coord['type'], 'gray')
        icon = icon_map.get(coord['type'], 'info-sign')
        
        folium.Marker(
            [coord['lat'], coord['lon']],
            popup=f"{coord['type'].title()}: {coord['name']}<br>{coord['address']}",
            tooltip=f"{coord['address']} ({coord['lat']:.4f}, {coord['lon']:.4f})",
            icon=folium.Icon(color=color, icon=icon)
        ).add_to(m)
    
    # Save map
    m.save(output_path)
    print(f"🗺️  Interactive map saved to: {output_path}")
    return True

def print_coordinate_summary(coordinates: List[Dict], depot_coords: Dict):
    """Print a summary of all coordinates."""
    print("\n" + "="*80)
    print("📍 COORDINATE VALIDATION SUMMARY")
    print("="*80)
    
    if depot_coords:
        print(f"\n🏢 DEPOT COORDINATES:")
        print(f"   📍 {depot_coords['address']}")
        print(f"   🌍 Latitude: {depot_coords['lat']:.6f}")
        print(f"   🌍 Longitude: {depot_coords['lon']:.6f}")
    
    print(f"\n📦 ORDER LOCATIONS ({len(coordinates)} total):")
    print("-" * 80)
    
    # Group by address to avoid duplicates
    unique_addresses = {}
    for coord in coordinates:
        addr = coord['address']
        if addr not in unique_addresses:
            unique_addresses[addr] = coord
    
    for i, (address, coord) in enumerate(unique_addresses.items(), 1):
        print(f"{i:2d}. {address}")
        print(f"     🌍 Lat: {coord['lat']:9.6f} | Lon: {coord['lon']:9.6f}")
        print(f"     📋 Type: {coord['type']}")
        print()

def main():
    """Main coordinate validation function."""
    print("🔍 EPDT COORDINATE VALIDATION TEST")
    print("="*50)
    
    # Load coordinates
    coordinates, depot_coords = load_geocoded_coordinates()
    
    if not coordinates and not depot_coords:
        print("❌ No coordinates loaded. Exiting.")
        return
    
    print(f"✅ Loaded {len(coordinates)} location coordinates")
    if depot_coords:
        print(f"✅ Loaded depot coordinates")
    
    # Print coordinate summary
    print_coordinate_summary(coordinates, depot_coords)
    
    # Validate coordinate bounds
    invalid_coords = validate_coordinate_bounds(coordinates + ([depot_coords] if depot_coords else []))
    
    if invalid_coords:
        print(f"\n⚠️  POTENTIAL COORDINATE ISSUES FOUND ({len(invalid_coords)} locations):")
        print("-" * 80)
        for coord in invalid_coords:
            print(f"❌ {coord['address']}")
            print(f"   🌍 Coordinates: ({coord['lat']:.6f}, {coord['lon']:.6f})")
            for issue in coord['issues']:
                print(f"   ⚠️  {issue}")
            print()
    else:
        print("\n✅ All coordinates appear to be within reasonable bounds for Italy/Europe")
    
    # Create map if possible
    if FOLIUM_AVAILABLE:
        output_path = os.path.join(current_dir, 'coordinate_validation_map.html')
        create_coordinate_map(coordinates, depot_coords, output_path)
    
    # Save coordinates to JSON for external validation
    all_coords = coordinates + ([depot_coords] if depot_coords else [])
    output_json = os.path.join(current_dir, 'extracted_coordinates.json')
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_coords, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Coordinates saved to: {output_json}")
    print("   (You can use this file to cross-reference with external geocoding services)")
    
    print("\n🎯 NEXT STEPS FOR VALIDATION:")
    print("1. Review the generated map for obviously misplaced locations")
    print("2. Cross-reference suspicious coordinates with Google Maps")
    print("3. Check addresses with unusual coordinate patterns")
    print("4. Verify international locations (France, San Marino) are correctly placed")

if __name__ == "__main__":
    main()
