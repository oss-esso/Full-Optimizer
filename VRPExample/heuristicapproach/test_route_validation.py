#!/usr/bin/env python3
"""
Test script to validate that route provider correctly handles coordinate changes.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'algo'))

from route_provider import RouteProvider

def test_coordinate_validation():
    """Test that the route provider validates coordinates correctly."""
    
    print("🧪 Testing Route Provider Coordinate Validation")
    print("=" * 60)
    
    # Initialize route provider
    route_provider = RouteProvider(db_path="test_route_validation.db")
    
    # Test coordinates - Milan vs Asti
    milan_coords = (9.1896, 45.4642)  # Old Milan coordinates
    asti_coords = (8.2057, 44.9009)   # New Asti coordinates
    
    # Test location
    test_destination = (11.5000, 45.1167)  # Badia Polesine
    
    print(f"1️⃣  Testing route from Milan depot to destination...")
    print(f"   Milan coords: {milan_coords}")
    print(f"   Destination: {test_destination}")
    
    # Get route from Milan depot
    route1 = route_provider.get_route_details(
        start_node_id="depot",
        end_node_id="badia_polesine_ro",
        start_coords=milan_coords,
        end_coords=test_destination
    )
    
    if route1:
        print(f"   ✅ Route calculated: {route1['distance_km']:.2f} km, {route1['duration_minutes']:.1f} min")
    else:
        print(f"   ❌ Failed to get route")
    
    print(f"\n2️⃣  Testing route from Asti depot to same destination...")
    print(f"   Asti coords: {asti_coords}")
    print(f"   Destination: {test_destination}")
    
    # Get route from Asti depot (should trigger coordinate validation)
    route2 = route_provider.get_route_details(
        start_node_id="depot",
        end_node_id="badia_polesine_ro", 
        start_coords=asti_coords,
        end_coords=test_destination
    )
    
    if route2:
        print(f"   ✅ Route calculated: {route2['distance_km']:.2f} km, {route2['duration_minutes']:.1f} min")
    else:
        print(f"   ❌ Failed to get route")
    
    # Compare routes
    if route1 and route2:
        distance_diff = abs(route1['distance_km'] - route2['distance_km'])
        time_diff = abs(route1['duration_minutes'] - route2['duration_minutes'])
        
        print(f"\n📊 Route Comparison:")
        print(f"   Distance difference: {distance_diff:.2f} km")
        print(f"   Time difference: {time_diff:.1f} minutes")
        
        if distance_diff > 10:  # Significant difference expected
            print(f"   ✅ Coordinate validation working - routes are different!")
        else:
            print(f"   ⚠️  Routes are very similar - validation may not be triggered")
    
    print(f"\n3️⃣  Testing cached route retrieval with wrong coordinates...")
    
    # Try to get the cached route with original coordinates (should find it)
    cached_route_correct = route_provider.get_route_details(
        start_node_id="depot",
        end_node_id="badia_polesine_ro",
        start_coords=asti_coords,  # Correct current coordinates
        end_coords=test_destination
    )
    
    # Try to get with wrong coordinates (should trigger recalculation if different)
    wrong_coords = (10.0, 45.0)  # Some other coordinates
    cached_route_wrong = route_provider.get_route_details(
        start_node_id="depot", 
        end_node_id="test_destination",
        start_coords=wrong_coords,
        end_coords=test_destination
    )
    
    print("✅ Test completed!")
    
    # Clean up test database
    try:
        os.remove("test_route_validation.db")
        print("🧹 Cleaned up test database")
    except:
        pass

if __name__ == "__main__":
    test_coordinate_validation()
