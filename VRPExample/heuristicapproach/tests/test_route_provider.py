"""
Test Advanced Travel Time Calculation

This script tests the OSRM route provider integration and vehicle-specific
travel time calculations for the EPDT algorithm.

Tests:
1. RouteProvider database caching functionality
2. Vehicle-specific travel time calculations  
3. Integration with EPDT data structures
4. Fallback behavior when OSRM is unavailable

Usage:
    python test_route_provider.py
"""

import os
import sys
import sqlite3
import json
from pathlib import Path

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
algo_dir = os.path.join(current_dir, '..', 'algo')
sys.path.insert(0, algo_dir)

# Import the modules we're testing
try:
    from route_provider import RouteProvider, calculate_travel_time_between_tasks
    from epdt_data_structures import Task, Vehicle, TaskType
    print("✅ Successfully imported route_provider and epdt_data_structures")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def test_route_provider_basic():
    """Test basic RouteProvider functionality."""
    print("\n🧪 Testing RouteProvider Basic Functionality")
    
    # Create a test route provider with a temporary database
    test_db = "test_routes.db"
    if os.path.exists(test_db):
        os.remove(test_db)
    
    provider = RouteProvider(db_path=test_db)
    
    # Test Milan to Rome route (coordinates from the road_composition.py example)
    milan_coords = (9.18951, 45.46427)  # (lon, lat)
    rome_coords = (12.49637, 41.90278)
    
    print(f"📍 Testing route: Milan {milan_coords} → Rome {rome_coords}")
    
    # Test fetching route details
    route_data = provider.get_route_details(
        "milan", "rome", milan_coords, rome_coords
    )
    
    if route_data:
        print(f"✅ Route data retrieved:")
        print(f"   Distance: {route_data['distance_km']:.2f} km")
        print(f"   Base duration: {route_data['base_duration_minutes']:.2f} minutes")
        print(f"   Road composition: {len(route_data['road_composition'])} road types")
        
        # Test caching - should be faster second time
        route_data_cached = provider.get_route_details("milan", "rome")
        if route_data_cached:
            print("✅ Route data successfully retrieved from cache")
        else:
            print("❌ Failed to retrieve from cache")
    else:
        print("❌ Failed to retrieve route data (might be network issue)")
        return False
    
    # Clean up test database
    if os.path.exists(test_db):
        os.remove(test_db)
    
    return True


def test_vehicle_specific_calculations():
    """Test vehicle-specific travel time calculations."""
    print("\n🚛 Testing Vehicle-Specific Travel Time Calculations")
    
    # Create test tasks
    task1 = Task(
        id="pickup_milan",
        location_id="milan",
        task_type=TaskType.PICKUP,
        order_id="order1",
        lat=45.46427,
        lon=9.18951,
        service_time=10.0,
        demand=100.0
    )
    
    task2 = Task(
        id="delivery_rome", 
        location_id="rome",
        task_type=TaskType.DELIVERY,
        order_id="order1",
        lat=41.90278,
        lon=12.49637,
        service_time=15.0,
        demand=-100.0
    )
    
    # Create different vehicle types
    car_vehicle = Vehicle(
        id="car1",
        depot_id="depot1",
        weight_capacity=500,
        volume_capacity=5,
        vehicle_type="car"
    )
    
    standard_truck = Vehicle(
        id="truck1", 
        depot_id="depot1",
        weight_capacity=1500,
        volume_capacity=15,
        vehicle_type="standard"
    )
    
    heavy_truck = Vehicle(
        id="heavy1",
        depot_id="depot1", 
        weight_capacity=3000,
        volume_capacity=30,
        vehicle_type="heavy"
    )
    
    vehicles = [
        ("Car", car_vehicle),
        ("Standard Truck", standard_truck), 
        ("Heavy Truck", heavy_truck)
    ]
    
    print(f"📦 Testing travel time from {task1.id} to {task2.id}")
    
    travel_times = {}
    for vehicle_name, vehicle in vehicles:
        try:
            travel_time = calculate_travel_time_between_tasks(task1, task2, vehicle)
            travel_times[vehicle_name] = travel_time
            print(f"   {vehicle_name:15}: {travel_time:.2f} minutes")
        except Exception as e:
            print(f"   ❌ {vehicle_name:15}: Error - {e}")
            travel_times[vehicle_name] = None
    
    # Verify that heavy trucks take longer than standard trucks, which take longer than cars
    if all(t is not None for t in travel_times.values()):
        car_time = travel_times["Car"]
        standard_time = travel_times["Standard Truck"] 
        heavy_time = travel_times["Heavy Truck"]
        
        if car_time <= standard_time <= heavy_time:
            print("✅ Vehicle travel times follow expected order (car ≤ standard ≤ heavy)")
        else:
            print(f"⚠️  Unexpected travel time order: car={car_time:.1f}, standard={standard_time:.1f}, heavy={heavy_time:.1f}")
    
    return travel_times


def test_database_caching():
    """Test the SQLite database caching functionality."""
    print("\n💾 Testing Database Caching")
    
    test_db = "test_cache.db"
    if os.path.exists(test_db):
        os.remove(test_db)
    
    provider = RouteProvider(db_path=test_db)
    
    # Manually insert test data
    conn = sqlite3.connect(test_db)
    cursor = conn.cursor()
    
    test_route_data = {
        'distance_km': 150.5,
        'base_duration_minutes': 120.0,
        'road_composition': {'motorway': 85.0, 'primary': 15.0},
        'route_geometry': {'type': 'LineString', 'coordinates': [[9.0, 45.0], [12.0, 41.0]]}
    }
    
    cursor.execute("""
        INSERT INTO routes 
        (start_node_id, end_node_id, distance_km, base_duration_minutes, 
         road_composition_json, route_geometry_json)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        "test_start", "test_end",
        test_route_data['distance_km'],
        test_route_data['base_duration_minutes'],
        json.dumps(test_route_data['road_composition']),
        json.dumps(test_route_data['route_geometry'])
    ))
    
    conn.commit()
    conn.close()
    
    # Test retrieval from cache
    cached_data = provider.get_route_details("test_start", "test_end")
    
    if cached_data:
        print("✅ Successfully retrieved data from cache:")
        print(f"   Distance: {cached_data['distance_km']} km")
        print(f"   Duration: {cached_data['base_duration_minutes']} minutes")
        print(f"   Road composition: {cached_data['road_composition']}")
        
        # Verify data integrity
        if (cached_data['distance_km'] == test_route_data['distance_km'] and
            cached_data['base_duration_minutes'] == test_route_data['base_duration_minutes']):
            print("✅ Cache data integrity verified")
        else:
            print("❌ Cache data mismatch")
    else:
        print("❌ Failed to retrieve from cache")
    
    # Clean up
    if os.path.exists(test_db):
        os.remove(test_db)
    
    return cached_data is not None


def test_fallback_behavior():
    """Test fallback behavior when OSRM is unavailable."""
    print("\n🔄 Testing Fallback Behavior")
    
    # Create tasks with coordinates but no external network access
    task1 = Task(
        id="test1", location_id="loc1", task_type=TaskType.PICKUP,
        order_id="order1", lat=45.0, lon=9.0, service_time=5.0
    )
    
    task2 = Task(
        id="test2", location_id="loc2", task_type=TaskType.DELIVERY, 
        order_id="order1", lat=46.0, lon=10.0, service_time=5.0
    )
    
    vehicle = Vehicle(
        id="test_vehicle", depot_id="depot1",
        weight_capacity=1000, volume_capacity=10
    )
    
    # Test with unavailable route data (no coordinates provided to force cache miss)
    provider = RouteProvider(db_path="empty_test.db")
    
    # This should fallback to Euclidean distance
    try:
        travel_time = calculate_travel_time_between_tasks(task1, task2, vehicle)
        print(f"✅ Fallback calculation successful: {travel_time:.2f} minutes")
        
        if travel_time >= 1.0:  # Should have minimum 1 minute
            print("✅ Minimum travel time constraint respected")
        else:
            print("❌ Minimum travel time constraint violated")
            
    except Exception as e:
        print(f"❌ Fallback calculation failed: {e}")
        return False
    
    # Clean up
    if os.path.exists("empty_test.db"):
        os.remove("empty_test.db")
    
    return True


def test_edge_cases():
    """Test edge cases and error conditions."""
    print("\n🎯 Testing Edge Cases")
    
    all_passed = True
    
    # Test same location (zero distance)
    task_same = Task(
        id="same", location_id="same_loc", task_type=TaskType.PICKUP,
        order_id="order1", lat=45.0, lon=9.0, service_time=5.0
    )
    
    vehicle = Vehicle(
        id="test", depot_id="depot1", 
        weight_capacity=1000, volume_capacity=10
    )
    
    try:
        travel_time = calculate_travel_time_between_tasks(task_same, task_same, vehicle)
        print(f"✅ Same location test: {travel_time:.2f} minutes")
        
        if travel_time >= 0:
            print("✅ Zero distance handled correctly")
        else:
            print("❌ Negative travel time returned")
            all_passed = False
    except Exception as e:
        print(f"❌ Same location test failed: {e}")
        all_passed = False
    
    # Test with missing attributes
    minimal_task1 = Task(
        id="minimal1", location_id="loc1", task_type=TaskType.PICKUP,
        order_id="order1", lat=0, lon=0, service_time=0
    )
    
    minimal_task2 = Task(
        id="minimal2", location_id="loc2", task_type=TaskType.DELIVERY,
        order_id="order1", lat=1, lon=1, service_time=0
    )
    
    try:
        travel_time = calculate_travel_time_between_tasks(minimal_task1, minimal_task2, vehicle)
        print(f"✅ Minimal attributes test: {travel_time:.2f} minutes")
        
        if travel_time >= 1.0:  # Should have minimum 1 minute
            print("✅ Minimal attributes constraints respected")
        else:
            print("❌ Minimal travel time constraint violated")
            all_passed = False
    except Exception as e:
        print(f"❌ Minimal attributes test failed: {e}")
        all_passed = False
    
    return all_passed


def main():
    """Run all tests for the route provider functionality."""
    print("=" * 80)
    print("🚀 ADVANCED TRAVEL TIME CALCULATION TEST SUITE")
    print("=" * 80)
    
    tests = [
        ("Basic RouteProvider", test_route_provider_basic),
        ("Vehicle-Specific Calculations", test_vehicle_specific_calculations),
        ("Database Caching", test_database_caching),
        ("Fallback Behavior", test_fallback_behavior),
        ("Edge Cases", test_edge_cases),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:30}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Route provider is ready for production use.")
    else:
        print("⚠️  Some tests failed. Please review the implementation.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
