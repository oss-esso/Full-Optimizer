#!/usr/bin/env python3
"""
Test the enhanced sequential optimizer with a small dataset.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vrp_multiday_sequential import SequentialMultiDayVRP

def test_small_enhanced_sequential():
    """Test with a small set of locations to verify the enhancement works."""
    print("🧪 Testing Enhanced Sequential Optimizer (Small Dataset)")
    print("=" * 60)
    
    # Small test locations around Zurich area (realistic coordinates)
    locations = [
        {'id': 'depot', 'x': 47.3769, 'y': 8.5417, 'demand': 0, 'service_time': 0, 'address': 'Zurich Depot'},
        {'id': 'winterthur', 'x': 47.5009, 'y': 8.7240, 'demand': 100, 'service_time': 20, 'address': 'Winterthur'},
        {'id': 'basel', 'x': 47.5596, 'y': 7.5886, 'demand': 120, 'service_time': 25, 'address': 'Basel'},
        {'id': 'bern', 'x': 46.9481, 'y': 7.4474, 'demand': 110, 'service_time': 22, 'address': 'Bern'},
        {'id': 'lucerne', 'x': 47.0502, 'y': 8.3093, 'demand': 90, 'service_time': 18, 'address': 'Lucerne'},
        {'id': 'st_gallen', 'x': 47.4245, 'y': 9.3767, 'demand': 80, 'service_time': 15, 'address': 'St. Gallen'},
    ]
    
    vehicles = [
        {'id': 'truck_1', 'capacity': 300, 'volume_capacity': 10.0, 'cost_per_km': 1.50, 'max_time': 480},
        {'id': 'truck_2', 'capacity': 250, 'volume_capacity': 8.0, 'cost_per_km': 1.20, 'max_time': 480},
    ]
    
    print(f"📍 Small test: {len(locations)} locations, {len(vehicles)} vehicles")
    print(f"📦 Total demand: {sum(loc.get('demand', 0) for loc in locations)} units")
    
    # Test with road composition enabled
    print(f"\n🛣️ Testing with road composition enabled...")
    sequential_vrp = SequentialMultiDayVRP(
        vehicles, 
        locations, 
        use_road_composition=True,
        max_route_distance_km=300  # Limit routes to reduce API calls
    )
    
    # Show some sample route data
    if hasattr(sequential_vrp, 'route_db') and sequential_vrp.route_db:
        sample_route = sequential_vrp.route_db.get_route_with_composition(locations[0], locations[1])
        if sample_route:
            print(f"\n📊 Sample route (depot → winterthur):")
            print(f"   Distance: {sample_route['distance_km']:.1f} km")
            print(f"   Car time: {sample_route['car_time_minutes']:.1f} min")
            print(f"   Truck time: {sample_route['truck_time_minutes']:.1f} min")
            print(f"   Truck ratio: {sample_route['truck_ratio']:.3f}")
            if sample_route['composition']:
                print(f"   Road composition:")
                for road_type, distance in sample_route['composition'].items():
                    percentage = (distance / sample_route['distance_km']) * 100
                    print(f"     {road_type}: {distance:.1f} km ({percentage:.1f}%)")
    
    print(f"\n✅ Enhanced Sequential Optimizer test completed!")
    return sequential_vrp

def test_comparison():
    """Compare road composition vs simple calculation."""
    print(f"\n🔄 COMPARISON TEST: Road Composition vs Simple Calculation")
    print("=" * 60)
    
    # Same small dataset
    locations = [
        {'id': 'depot', 'x': 47.3769, 'y': 8.5417, 'demand': 0, 'service_time': 0, 'address': 'Zurich Depot'},
        {'id': 'basel', 'x': 47.5596, 'y': 7.5886, 'demand': 120, 'service_time': 25, 'address': 'Basel'},
        {'id': 'bern', 'x': 46.9481, 'y': 7.4474, 'demand': 110, 'service_time': 22, 'address': 'Bern'},
        {'id': 'lucerne', 'x': 47.0502, 'y': 8.3093, 'demand': 90, 'service_time': 18, 'address': 'Lucerne'},
    ]
    
    vehicles = [
        {'id': 'truck_1', 'capacity': 500, 'volume_capacity': 15.0, 'cost_per_km': 1.50, 'max_time': 480},
    ]
    
    # Test with road composition
    print(f"\n🛣️ Method 1: Road Composition Analysis")
    vrp_composition = SequentialMultiDayVRP(vehicles, locations, use_road_composition=True, max_route_distance_km=200)
    
    print(f"\n📐 Method 2: Simple Calculation")
    vrp_simple = SequentialMultiDayVRP(vehicles, locations, use_road_composition=False)
    
    # Compare matrices
    print(f"\n📊 DISTANCE COMPARISON (km):")
    print(f"{'Route':<20} {'Composition':<12} {'Simple':<10} {'Difference':<12}")
    print("-" * 55)
    
    route_names = ['Depot→Basel', 'Depot→Bern', 'Depot→Lucerne', 'Basel→Bern']
    route_pairs = [(0,1), (0,2), (0,3), (1,2)]
    
    for name, (i,j) in zip(route_names, route_pairs):
        if i < len(locations) and j < len(locations):
            comp_dist = vrp_composition.distance_matrix[i,j]
            simple_dist = vrp_simple.distance_matrix[i,j]
            diff = comp_dist - simple_dist
            print(f"{name:<20} {comp_dist:>8.1f} km   {simple_dist:>6.1f} km   {diff:>+7.1f} km")
    
    print(f"\n⏱️  TIME COMPARISON (minutes):")
    print(f"{'Route':<20} {'Composition':<12} {'Simple':<10} {'Difference':<12}")
    print("-" * 55)
    
    for name, (i,j) in zip(route_names, route_pairs):
        if i < len(locations) and j < len(locations):
            comp_time = vrp_composition.time_matrix[i,j]
            simple_time = vrp_simple.time_matrix[i,j]
            diff = comp_time - simple_time
            print(f"{name:<20} {comp_time:>8.1f} min   {simple_time:>6.1f} min   {diff:>+7.1f} min")
    
    return vrp_composition, vrp_simple

if __name__ == "__main__":
    # Test the enhanced sequential optimizer
    vrp = test_small_enhanced_sequential()
    
    # Run comparison test
    comp_vrp, simple_vrp = test_comparison()
