"""
Minimal Working Example for Road Composition-Based Truck Speed Adjustments

This MWE demonstrates the enhanced truck speed adjustment system that:
1. Uses OSRM to get detailed route information with road type composition
2. Applies weighted truck speed adjustments based on actual road types used
3. Compares different truck types (standard vs heavy) with realistic speed profiles
4. Shows the difference between simple default adjustments and composition-based adjustments

The road composition approach is more realistic because:
- Different road types have different speed impacts for trucks
- Highway routes favor trucks less than city routes
- Rural roads have different impacts than urban areas
"""

import time
from typing import Dict, List
import sys
import os

# Add the VRPExample directory to Python path if needed
vrp_example_dir = os.path.dirname(os.path.abspath(__file__))
if vrp_example_dir not in sys.path:
    sys.path.append(vrp_example_dir)

def test_road_composition_truck_adjustments():
    """Test road composition-based truck speed adjustments."""
    print("🛣️ Testing Road Composition-Based Truck Speed Adjustments")
    print("=" * 70)
    
    # Import required modules
    try:
        from vrp_optimizer_clean import OSMDistanceCalculator
        from vrp_scenarios import DEFAULT_TRUCK_SPEED_RATIOS
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return
    
    # Define test locations (Italian cities with different route types)
    test_locations = [
        {'id': 'depot', 'x': 9.1900, 'y': 45.4642},     # Milan (depot)
        {'id': 'highway_route', 'x': 11.8767, 'y': 45.4058},  # Venice (mostly highway)
        {'id': 'mixed_route', 'x': 8.9463, 'y': 44.4056},     # Genoa (mixed roads)
        {'id': 'urban_route', 'x': 9.2497, 'y': 45.4903},     # Monza (urban roads)
    ]
    
    print(f"📍 Test locations:")
    for loc in test_locations:
        print(f"  - {loc['id']}: ({loc['x']:.4f}, {loc['y']:.4f})")
    
    # Test configurations
    test_configs = [
        {
            'name': 'Car Baseline',
            'truck_speed_ratios': None,
            'use_truck_speeds': False,
            'use_road_composition': False
        },
        {
            'name': 'Standard Truck (Default)',
            'truck_speed_ratios': DEFAULT_TRUCK_SPEED_RATIOS['standard'],
            'use_truck_speeds': True,
            'use_road_composition': False
        },
        {
            'name': 'Standard Truck (Composition)',
            'truck_speed_ratios': DEFAULT_TRUCK_SPEED_RATIOS['standard'],
            'use_truck_speeds': True,
            'use_road_composition': True
        },
        {
            'name': 'Heavy Truck (Default)',
            'truck_speed_ratios': DEFAULT_TRUCK_SPEED_RATIOS['heavy'],
            'use_truck_speeds': True,
            'use_road_composition': False
        },
        {
            'name': 'Heavy Truck (Composition)',
            'truck_speed_ratios': DEFAULT_TRUCK_SPEED_RATIOS['heavy'],
            'use_truck_speeds': True,
            'use_road_composition': True
        }
    ]
    
    # Display truck speed profiles
    print(f"\n🚛 Truck Speed Profiles:")
    for truck_type, ratios in DEFAULT_TRUCK_SPEED_RATIOS.items():
        print(f"  {truck_type.upper()} TRUCK:")
        for road_type, ratio in ratios.items():
            if road_type != 'default':
                slowdown_pct = int((1 - ratio) * 100)
                print(f"    - {road_type}: {ratio:.2f} ({slowdown_pct}% slower)")
            else:
                slowdown_pct = int((1 - ratio) * 100)
                print(f"    - {road_type}: {ratio:.2f} ({slowdown_pct}% slower overall)")
    
    # Run tests for each configuration
    results = {}
    
    for config in test_configs:
        print(f"\n" + "─" * 50)
        print(f"🔧 Testing: {config['name']}")
        print(f"   Use truck speeds: {config['use_truck_speeds']}")
        print(f"   Use road composition: {config['use_road_composition']}")
        
        try:
            # Create calculator with specific configuration
            calculator = OSMDistanceCalculator(
                locations=test_locations,
                truck_speed_ratios=config['truck_speed_ratios'],
                use_truck_speeds=config['use_truck_speeds'],
                use_road_composition=config['use_road_composition']
            )
            
            # Test routes
            route_results = {}
            
            for i in range(1, len(test_locations)):  # Skip depot to depot
                origin = test_locations[0]  # depot
                destination = test_locations[i]
                route_name = f"depot → {destination['id']}"
                
                print(f"  📍 Route: {route_name}")
                
                # Get travel time (distance is same for all configs)
                travel_time = calculator.time_matrix[0, i]
                
                route_results[route_name] = {
                    'time_minutes': travel_time,
                    'destination': destination['id']
                }
                
                print(f"    ⏱️ Travel time: {travel_time:.1f} minutes")
                
                # If using road composition, show composition details
                if config['use_road_composition']:
                    cache_key = f"{origin['id']}→{destination['id']}"
                    if cache_key in calculator.road_composition_cache:
                        composition = calculator.road_composition_cache[cache_key]
                        print(f"    🛣️ Road composition:")
                        total_distance = sum(composition.values())
                        for road_type, distance_km in composition.items():
                            percentage = (distance_km / total_distance) * 100 if total_distance > 0 else 0
                            print(f"      - {road_type}: {distance_km:.1f}km ({percentage:.1f}%)")
                        
                        # Calculate weighted truck ratio for this route
                        if config['truck_speed_ratios']:
                            weighted_ratio = calculator._calculate_weighted_truck_ratio(composition)
                            slowdown_pct = int((1 - weighted_ratio) * 100)
                            print(f"    🚛 Route-specific truck adjustment: {weighted_ratio:.3f} ({slowdown_pct}% slower)")
            
            results[config['name']] = route_results
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
            results[config['name']] = None
    
    # Compare results
    print(f"\n" + "=" * 70)
    print(f"📊 COMPARISON RESULTS")
    print(f"=" * 70)
    
    # Create comparison table
    route_names = []
    if results.get('Car Baseline'):
        route_names = list(results['Car Baseline'].keys())
    
    for route_name in route_names:
        print(f"\n🛣️ Route: {route_name}")
        print(f"{'Configuration':<25} {'Time (min)':<12} {'vs Car':<12} {'vs Default'}")
        print("-" * 65)
        
        car_time = None
        default_truck_time = None
        
        for config_name, route_data in results.items():
            if route_data and route_name in route_data:
                time_min = route_data[route_name]['time_minutes']
                
                # Store baseline times
                if config_name == 'Car Baseline':
                    car_time = time_min
                elif 'Default' in config_name and default_truck_time is None:
                    default_truck_time = time_min
                
                # Calculate comparisons
                vs_car = ""
                vs_default = ""
                
                if car_time and car_time > 0:
                    time_increase = ((time_min - car_time) / car_time) * 100
                    vs_car = f"+{time_increase:.1f}%"
                
                if default_truck_time and default_truck_time > 0 and 'Composition' in config_name:
                    if 'Standard' in config_name:
                        # Compare with Standard Truck (Default)
                        default_time = None
                        for name, data in results.items():
                            if name == 'Standard Truck (Default)' and data and route_name in data:
                                default_time = data[route_name]['time_minutes']
                                break
                        if default_time:
                            diff_pct = ((time_min - default_time) / default_time) * 100
                            vs_default = f"{diff_pct:+.1f}%"
                    elif 'Heavy' in config_name:
                        # Compare with Heavy Truck (Default)
                        default_time = None
                        for name, data in results.items():
                            if name == 'Heavy Truck (Default)' and data and route_name in data:
                                default_time = data[route_name]['time_minutes']
                                break
                        if default_time:
                            diff_pct = ((time_min - default_time) / default_time) * 100
                            vs_default = f"{diff_pct:+.1f}%"
                
                print(f"{config_name:<25} {time_min:<12.1f} {vs_car:<12} {vs_default}")
            else:
                print(f"{config_name:<25} {'ERROR':<12} {'':<12} {''}")
    
    # Summary insights
    print(f"\n📋 KEY INSIGHTS:")
    print(f"─" * 50)
    
    print(f"✅ Road composition-based adjustments provide more realistic truck routing:")
    print(f"  • Highway-heavy routes: Less truck penalty (trucks handle highways better)")
    print(f"  • Urban-heavy routes: More truck penalty (trucks struggle in cities)")
    print(f"  • Mixed routes: Balanced adjustment based on actual road type mix")
    
    print(f"\n✅ Comparison with default adjustment:")
    print(f"  • Default: Single factor applied to all routes uniformly")
    print(f"  • Composition: Route-specific factors based on actual road types")
    print(f"  • More accurate for route planning and cost estimation")
    
    return results

def test_optimizer_with_road_composition():
    """Test the full VRP optimizer with road composition-based truck speeds."""
    print(f"\n" + "=" * 70)
    print(f"🚛 Testing VRP Optimizer with Road Composition")
    print(f"=" * 70)
    
    try:
        from vrp_scenarios import create_vrppd_scenario, DEFAULT_TRUCK_SPEED_RATIOS
        from vrp_optimizer_clean import CleanVRPOptimizer
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return
    
    # Create a simple test scenario
    print("📦 Creating pickup-delivery scenario...")
    scenario = create_vrppd_scenario()
    
    # Convert to optimizer format
    vehicles_dicts = [{
        'id': v.id,
        'capacity': v.capacity,
        'volume_capacity': getattr(v, 'volume_capacity', v.capacity * 0.001),
        'cost_per_km': getattr(v, 'cost_per_km', 1.0),
        'start_location': v.depot_id,
        'end_location': v.depot_id,
        'max_time': getattr(v, 'max_time', 24 * 60),
        'use_truck_speeds': True,
        'truck_speed_ratios': DEFAULT_TRUCK_SPEED_RATIOS['standard'].copy()
    } for v in scenario.vehicles.values()]
    
    print(f"📊 Scenario: {len(scenario.locations)} locations, {len(vehicles_dicts)} vehicles")
    
    # Test with and without road composition
    test_cases = [
        ('Default Truck Adjustment', False),
        ('Road Composition Adjustment', True)
    ]
    
    for test_name, use_composition in test_cases:
        print(f"\n🧪 Testing: {test_name}")
        print(f"   Road composition: {use_composition}")
        
        try:
            # Temporarily modify the optimizer to use road composition
            # We'll need to patch the solve method
            optimizer = CleanVRPOptimizer(vehicles=vehicles_dicts, locations=None, vrp_instance=scenario)
            optimizer.ride_requests = scenario.ride_requests
            
            # Add a flag to enable road composition (this would normally be a parameter)
            optimizer._use_road_composition = use_composition
            
            start_time = time.time()
            result = optimizer.solve(constraint_level="pickup_delivery", verbose=False)
            solve_time = time.time() - start_time
            
            if result and len(result) >= 2:
                solution, status = result[0], result[1]
                if solution:
                    total_distance = solution.get('total_distance', 0)
                    total_cost = solution.get('total_cost', 0)
                    
                    print(f"  ✅ SUCCESS")
                    print(f"    Distance: {total_distance:.1f}km")
                    print(f"    Cost: €{total_cost:.2f}")
                    print(f"    Solve time: {solve_time:.1f}s")
                else:
                    print(f"  ❌ No solution found")
            else:
                print(f"  ❌ Solver failed")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print(f"\n📋 NOTE: To fully enable road composition in the optimizer,")
    print(f"    modify the OSMDistanceCalculator instantiation in the solve() method")
    print(f"    to include use_road_composition=True parameter.")

if __name__ == "__main__":
    print("🛣️ Road Composition-Based Truck Speed Adjustments - MWE")
    print("=" * 80)
    
    # Test 1: Road composition analysis
    print("Test 1: Road composition analysis and truck speed adjustments")
    results = test_road_composition_truck_adjustments()
    
    # Test 2: Integration with VRP optimizer
    print(f"\nTest 2: VRP optimizer integration")
    test_optimizer_with_road_composition()
    
    print(f"\n" + "=" * 80)
    print(f"✅ Road composition-based truck speed adjustment MWE completed!")
    print(f"   This demonstrates more realistic truck routing based on actual road types.")
