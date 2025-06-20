#!/usr/bin/env python3
"""
Test the updated scenarios with [0, close_time] time windows and fractional cargo loads.
"""

import sys
import os

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

from vrp_scenarios import get_all_scenarios
from vrp_optimizer_fixed import VRPQuantumOptimizer
from vrp_data_models import VRPObjective

def analyze_scenario(scenario_name):
    """Analyze and test a scenario."""
    print(f"\n{'='*40}")
    print(f"ANALYZING SCENARIO: {scenario_name}")
    print(f"{'='*40}")
    
    # Load scenario
    scenarios = get_all_scenarios()
    scenario = scenarios[scenario_name]
    
    # Basic scenario info
    print(f"\nScenario Details:")
    print(f"- Total locations: {len(scenario.locations)}")
    print(f"- Vehicles: {len(scenario.vehicles)}")
    print(f"- Requests: {len(scenario.ride_requests)}")
    
    # Vehicle details
    vehicle = list(scenario.vehicles.values())[0]
    print(f"\nVehicle Details:")
    print(f"- Capacity: {vehicle.capacity}")
    print(f"- Max time: {vehicle.max_time} minutes ({vehicle.max_time/60} hours)")
    
    # Time window analysis
    locations_with_tw = [(loc_id, loc.time_window_start, loc.time_window_end) 
                         for loc_id, loc in scenario.locations.items() 
                         if hasattr(loc, 'time_window_start') and loc.time_window_start is not None]
    
    print(f"\nTime Window Analysis:")
    print(f"- Locations with time windows: {len(locations_with_tw)}/{len(scenario.locations)}")
    print(f"\nTime Window Examples (first 5):")
    for loc_id, tw_start, tw_end in locations_with_tw[:5]:
        print(f"- {loc_id}: [{tw_start}, {tw_end}]")
    
    # Cargo load analysis
    print(f"\nCargo Load Analysis:")
    fractional_loads = [req.passengers for req in scenario.ride_requests if req.passengers % 1 != 0]
    print(f"- Total requests: {len(scenario.ride_requests)}")
    print(f"- Requests with fractional loads: {len(fractional_loads)}")
    
    print(f"\nCargo Load Examples (first 5):")
    for req in scenario.ride_requests[:5]:
        print(f"- Request {req.id}: {req.passengers} units")
    
    # Speed validation
    location_ids = list(scenario.locations.keys())
    if len(location_ids) >= 2:
        loc1_id, loc2_id = location_ids[0], location_ids[1]
        
        # For accurate speed calculation, we need to get the speed directly from the model
        print(f"\nSpeed Validation:")
        print(f"- Using fixed speed of 50 km/h as specified in vrp_data_models.py")
        
        # Verify the get_duration method is working correctly by checking the calculation
        loc1 = scenario.locations[loc1_id]
        loc2 = scenario.locations[loc2_id]
        
        # Calculate distance in coordinate units
        distance_units = scenario.get_distance(loc1_id, loc2_id)
        
        # Calculate the expected duration based on 50 km/h speed
        distance_km = distance_units * 111  # approximate GPS to km conversion
        expected_hours = distance_km / 50.0  # 50 km/h speed
        expected_seconds = expected_hours * 3600
        
        # Get the actual duration from the model
        actual_seconds = scenario.get_duration(loc1_id, loc2_id)
        
        print(f"- Distance: {distance_units:.4f} degrees ≈ {distance_km:.4f} km")
        print(f"- Expected duration at 50 km/h: {expected_seconds:.2f} seconds")
        print(f"- Actual duration from model: {actual_seconds:.2f} seconds")
        
        # Verify they match within a small tolerance
        tolerance = 0.1  # 10% tolerance
        ratio = abs(expected_seconds - actual_seconds) / expected_seconds if expected_seconds > 0 else float('inf')
        print(f"- Speed calculation {'✅ accurate' if ratio < tolerance else '❌ inaccurate'} (within {tolerance*100:.0f}% tolerance)")
    
    # Run optimization
    print(f"\nRunning Optimization...")
    try:
        optimizer = VRPQuantumOptimizer(scenario, VRPObjective.MINIMIZE_DISTANCE)
        import time
        start_time = time.time()
        result = optimizer.optimize_with_ortools()
        solve_time = time.time() - start_time
        
        # Check if solved
        solved = result.routes is not None and len(result.routes) > 0
        if solved:
            vehicles_used = len([r for r in result.routes.values() if len(r) > 2])
            solved = vehicles_used > 0
        else:
            vehicles_used = 0
        
        print(f"- Result: {'✅ SOLVED' if solved else '❌ NOT SOLVED'}")
        print(f"- Solve time: {solve_time:.3f}s")
        print(f"- Status: {result.status}")
        
        if solved:
            print(f"- Objective value: {result.objective_value:.2f}")
            print(f"- Vehicles used: {vehicles_used}/{len(scenario.vehicles)}")
            
            # Quick route analysis
            total_stops = sum(len(route) for route in result.routes.values())
            print(f"- Total stops across all routes: {total_stops}")
            
            # Show sample route
            active_routes = [v_id for v_id, route in result.routes.items() if len(route) > 2]
            if active_routes:
                sample_v_id = active_routes[0]
                sample_route = result.routes[sample_v_id]
                print(f"\nSample Route (vehicle {sample_v_id}):")
                print(f"- Stops: {len(sample_route)}")
                print(f"- Route: {' -> '.join(sample_route)}")
        
    except Exception as e:
        print(f"❌ ERROR during optimization: {str(e)}")
    
    print(f"\n{'='*40}\n")

def main():
    """Test main entry point."""
    print("=" * 80)
    print("TESTING UPDATED SCENARIOS WITH [0, close_time] TIME WINDOWS AND FRACTIONAL CARGO LOADS")
    print("=" * 80)
    
    try:
        # Test the small scenario first
        analyze_scenario('MODA_small')
        
        # Then try the larger scenario
        analyze_scenario('MODA_first')
        
    except Exception as e:
        print(f"ERROR: {str(e)}")

if __name__ == "__main__":
    main()
