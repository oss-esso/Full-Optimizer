"""
Test file to compare the two VRP optimizers on the overnight test scenario.

This test compares:
1. CleanVRPOptimizer (original from vrp_optimizer_clean.py)
2. CleanVRPOptimizer (OSM-enhanced copy from vrp_optimizer_clean_copy.py)

Both are tested on the overnight test scenario to evaluate:
- Distance calculation differences (Manhattan vs OSM)
- Solution quality and feasibility
- Capacity constraint handling
- Performance metrics
"""

import time
import logging
from typing import Dict, Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_overnight_scenario_comparison():
    """Compare both optimizers on the overnight test scenario."""
    print("🧪 VRP Optimizers Comparison Test on Overnight Test Scenario")
    print("=" * 80)
    
    # Import scenario creation function
    try:
        from vrp_scenarios import create_overnight_test_scenario
        print("📦 Using overnight test scenario")
    except ImportError:
        print("❌ Error: Could not import create_overnight_test_scenario")
        return
    
    # Create the test scenario
    print("\n📦 Creating overnight test scenario...")
    scenario = create_overnight_test_scenario()
    
    print(f"\n📊 Scenario Overview:")
    print(f"  - Name: {scenario.name}")
    print(f"  - Locations: {len(scenario.locations)}")
    print(f"  - Vehicles: {len(scenario.vehicles)}")
    print(f"  - Ride requests: {len(scenario.ride_requests)}")
    print(f"  - Is realistic: {getattr(scenario, 'is_realistic', False)}")
    
    # Convert vehicles to dict format for both optimizers
    vehicles_dicts = [{
        'id': v.id,
        'capacity': v.capacity,
        'volume_capacity': getattr(v, 'volume_capacity', v.capacity * 0.001),  # Estimate volume
        'cost_per_km': getattr(v, 'cost_per_km', 1.0),
        'start_location': v.depot_id,
        'end_location': v.depot_id,
        'max_time': getattr(v, 'max_time', 24 * 60)
    } for v in scenario.vehicles.values()]
    
    print(f"\n🚛 Vehicle Configuration:")
    for vehicle in vehicles_dicts:
        print(f"  - {vehicle['id']}: {vehicle['capacity']}kg, {vehicle['volume_capacity']:.2f}m³, "
              f"€{vehicle['cost_per_km']:.2f}/km, {vehicle['max_time']}min max")
    
    # Test constraint levels to evaluate
    constraint_levels = ["none", "capacity", "pickup_delivery", "time_windows", "full"]
    
    results = {}
    
    # Test Original Optimizer (Manhattan distance)
    print("\n" + "="*80)
    print("🔵 Testing Original Optimizer (Manhattan distance)")
    print("="*80)
    
    try:
        from vrp_optimizer_clean import CleanVRPOptimizer as OriginalOptimizer
        
        for level in constraint_levels:
            print(f"\n--- Testing constraint level: {level} ---")
            
            optimizer = OriginalOptimizer(vehicles=vehicles_dicts, locations=None, vrp_instance=scenario)
            optimizer.ride_requests = scenario.ride_requests
            
            start_time = time.time()
            try:
                result = optimizer.solve(constraint_level=level, verbose=False)
                solve_time = time.time() - start_time
                
                if result and len(result) >= 2:
                    solution, status = result[0], result[1]
                    success = solution is not None
                    total_distance = solution.get('total_distance', 0) if solution else 0
                    total_cost = solution.get('total_cost', 0) if solution else 0
                    
                    results[f"original_{level}"] = {
                        'success': success,
                        'status': status,
                        'solve_time': solve_time,
                        'total_distance': total_distance,
                        'total_cost': total_cost,
                        'optimizer': 'original',
                        'level': level
                    }
                    
                    print(f"✅ SUCCESS - Distance: {total_distance}km, Cost: €{total_cost:.2f}, Time: {solve_time:.1f}s")
                else:
                    results[f"original_{level}"] = {
                        'success': False,
                        'status': 'FAILED',
                        'solve_time': solve_time,
                        'optimizer': 'original',
                        'level': level
                    }
                    print(f"❌ FAILED - Time: {solve_time:.1f}s")
                    
            except Exception as e:
                solve_time = time.time() - start_time
                results[f"original_{level}"] = {
                    'success': False,
                    'status': f'ERROR: {str(e)}',
                    'solve_time': solve_time,
                    'optimizer': 'original',
                    'level': level
                }
                print(f"❌ ERROR: {e} - Time: {solve_time:.1f}s")
                
    except ImportError as e:
        print(f"❌ Could not import original optimizer: {e}")
    
    # Test OSM-Enhanced Optimizer 
    print("\n" + "="*80)
    print("🟢 Testing OSM-Enhanced Optimizer (Real routing)")
    print("="*80)
    
    # Print comparison summary
    print_comparison_summary(results, constraint_levels)

def print_comparison_summary(results: Dict, constraint_levels: list):
    """Print a comprehensive comparison summary."""
    print("\n" + "="*80)
    print("📊 COMPARISON SUMMARY")
    print("="*80)
    
    print(f"\n{'Constraint Level':<20} {'Original':<25} {'OSM-Enhanced':<25} {'Winner'}")
    print("-" * 90)
    
    for level in constraint_levels:
        original_key = f"original_{level}"
        osm_key = f"osm_{level}"
        
        # Get results for this level
        orig_result = results.get(original_key, {})
        osm_result = results.get(osm_key, {})
        
        # Format original result
        if orig_result.get('success', False):
            orig_str = f"✅ {orig_result.get('total_distance', 0):.1f}km €{orig_result.get('total_cost', 0):.1f}"
        else:
            orig_str = f"❌ {orig_result.get('status', 'UNKNOWN')}"
        
        # Format OSM result  
        if osm_result.get('success', False):
            osm_str = f"✅ {osm_result.get('total_distance', 0):.1f}km €{osm_result.get('total_cost', 0):.1f}"
        else:
            osm_str = f"❌ {osm_result.get('status', 'UNKNOWN')}"
        
        # Determine winner
        winner = "TIE"
        if orig_result.get('success', False) and osm_result.get('success', False):
            orig_cost = orig_result.get('total_cost', float('inf'))
            osm_cost = osm_result.get('total_cost', float('inf'))
            if orig_cost < osm_cost:
                winner = "Original"
            elif osm_cost < orig_cost:
                winner = "OSM"
        elif orig_result.get('success', False):
            winner = "Original"
        elif osm_result.get('success', False):
            winner = "OSM"
        
        print(f"{level:<20} {orig_str:<25} {osm_str:<25} {winner}")
    
    # Print timing comparison
    print(f"\n⏱️  TIMING COMPARISON")
    print("-" * 50)
    for level in constraint_levels:
        orig_time = results.get(f"original_{level}", {}).get('solve_time', 0)
        osm_time = results.get(f"osm_{level}", {}).get('solve_time', 0)
        
        faster = "TIE"
        if orig_time > 0 and osm_time > 0:
            if orig_time < osm_time:
                faster = "Original"
            elif osm_time < orig_time:
                faster = "OSM"
        elif orig_time > 0:
            faster = "Original"
        elif osm_time > 0:
            faster = "OSM"
        
        print(f"{level:<20} {orig_time:.2f}s vs {osm_time:.2f}s → {faster}")
    
    # Print success rate comparison
    print(f"\n🎯 SUCCESS RATE COMPARISON")
    print("-" * 50)
    
    orig_successes = sum(1 for level in constraint_levels if results.get(f"original_{level}", {}).get('success', False))
    osm_successes = sum(1 for level in constraint_levels if results.get(f"osm_{level}", {}).get('success', False))
    total_tests = len(constraint_levels)
    
    print(f"Original Optimizer:    {orig_successes}/{total_tests} ({orig_successes/total_tests*100:.1f}%)")
    print(f"OSM-Enhanced Optimizer: {osm_successes}/{total_tests} ({osm_successes/total_tests*100:.1f}%)")
    
    # Print key differences
    print(f"\n🔍 KEY DIFFERENCES")
    print("-" * 50)
    print("Original Optimizer:")
    print("  ✅ Fast Manhattan distance calculation")
    print("  ✅ Consistent geometric distance model")
    print("  ❌ Less realistic distance estimates")
    print("  ❌ No real-world routing factors")
    
    print("\nOSM-Enhanced Optimizer:")
    print("  ✅ Real-world OSM routing with actual road networks")
    print("  ✅ Truck-specific speed adjustments")
    print("  ✅ More accurate distance and time estimates")
    print("  ❌ Slower due to API calls and complex calculations")
    print("  ❌ Dependent on external OSRM service")

def test_single_constraint_level():
    """Quick test with just one constraint level for debugging."""
    print("🔬 Quick Single Constraint Test (pickup_delivery)")
    print("=" * 60)
    
    try:
        from vrp_scenarios import create_overnight_test_scenario
    except ImportError:
        print("❌ Error: Could not import create_overnight_test_scenario")
        return
    
    scenario = create_overnight_test_scenario()
    
    # Convert vehicles
    vehicles_dicts = [{
        'id': v.id,
        'capacity': v.capacity,
        'volume_capacity': getattr(v, 'volume_capacity', v.capacity * 0.001),
        'cost_per_km': getattr(v, 'cost_per_km', 1.0),
        'start_location': v.depot_id,
        'end_location': v.depot_id,
        'max_time': getattr(v, 'max_time', 24 * 60)
    } for v in scenario.vehicles.values()]
    
    # Test both optimizers with pickup_delivery constraint
    constraint_level = "pickup_delivery"
    
    print(f"\n🔵 Testing Original Optimizer...")
    try:
        from vrp_optimizer_clean import CleanVRPOptimizer as OriginalOptimizer
        optimizer1 = OriginalOptimizer(vehicles=vehicles_dicts, locations=None, vrp_instance=scenario)
        optimizer1.ride_requests = scenario.ride_requests
        result1 = optimizer1.solve(constraint_level=constraint_level, verbose=True)
        print(f"Original result: {type(result1)} - {result1 is not None}")
    except Exception as e:
        print(f"❌ Original optimizer error: {e}")
    
    print(f"\n🟢 Testing OSM-Enhanced Optimizer...")
    try:
        from vrp_optimizer_clean_copy import CleanVRPOptimizer as OSMOptimizer
        optimizer2 = OSMOptimizer(vehicles=vehicles_dicts, locations=None, vrp_instance=scenario)
        optimizer2.ride_requests = scenario.ride_requests
        result2 = optimizer2.solve(constraint_level=constraint_level, verbose=True)
        print(f"OSM result: {type(result2)} - {result2 is not None}")
    except Exception as e:
        print(f"❌ OSM optimizer error: {e}")

def test_overnight_node_creation():
    """Test the sequential multi-day VRP optimizer on the realistic MODA furgoni scenario.
    
    This tests the sequential multi-day VRP solver on a complex, realistic scenario with:
    - Multiple vehicles with different capacities
    - Real pickup and delivery requests
    - Realistic time constraints and routing
    - Multi-day planning when daily limits are exceeded
    
    The solver should create overnight nodes when needed and ensure all vehicles return to depot.
    """
    print("\n" + "="*80)
    print("🚛 Testing Sequential Multi-Day VRP on MODA Furgoni Scenario")
    print("="*80)
    
    try:
        import sys
        import os
        import importlib.util
        from vrp_scenarios import create_furgoni_scenario
        
        print("📦 Using MODA furgoni scenario for multiday test")
    except ImportError as e:
        print(f"❌ Error importing required modules: {e}")
        return
    
    # Create the test scenario
    print("\n📦 Creating MODA furgoni scenario...")
    scenario = create_furgoni_scenario()
    
    # Analyze the scenario
    print(f"\n📊 Furgoni Scenario Analysis:")
    print(f"  - Total locations: {len(scenario.locations)}")
    print(f"  - Total vehicles: {len(scenario.vehicles)}")
    print(f"  - Total requests: {len(scenario.ride_requests)}")
    
    # Show some key locations
    depot_found = False
    pickup_locations = []
    delivery_locations = []
    
    for loc_id, loc in scenario.locations.items():
        if 'depot' in str(loc_id).lower():
            depot_found = True
            print(f"  ✅ Depot found: {loc_id} at ({loc.x}, {loc.y})")
            if hasattr(loc, 'address'):
                print(f"      Address: {loc.address}")
        elif any(keyword in str(loc_id).lower() for keyword in ['pickup', 'via', 'source']):
            pickup_locations.append(loc_id)
        else:
            delivery_locations.append(loc_id)
    
    print(f"  - Pickup locations: {len(pickup_locations)}")
    print(f"  - Delivery locations: {len(delivery_locations)}")
    
    if not depot_found:
        print("❌ No depot found in scenario locations.")
        print("Available locations:")
        for loc_id, loc in scenario.locations.items():
            print(f" - Location ID: {loc_id}")
            if hasattr(loc, 'address'):
                print(f"   Address: {loc.address}")
        return
    
    # Convert scenario data to format expected by SequentialMultiDayVRP
    
    # Get locations in format needed for SequentialMultiDayVRP
    locations = []
    for loc_id, loc in scenario.locations.items():
        location = {
            'id': str(loc_id),
            'x': float(loc.x),
            'y': float(loc.y),
            'demand': getattr(loc, 'demand', 0),
            'service_time': getattr(loc, 'service_time', 15),
            'address': getattr(loc, 'name', str(loc_id))
        }
        locations.append(location)
    
    # Convert vehicles to format needed for SequentialMultiDayVRP
    vehicles = []
    for v_id, v in scenario.vehicles.items():
        vehicle = {
            'id': str(v_id),
            'capacity': v.capacity,
            'volume_capacity': getattr(v, 'volume_capacity', v.capacity * 0.001),
            'cost_per_km': getattr(v, 'cost_per_km', 1.0),
            'max_daily_km': getattr(v, 'max_daily_km', 600),
            'max_time': getattr(v, 'max_time', 24 * 60)  # Make sure max_time is included
        }
        vehicles.append(vehicle)
    
    print(f"\n📊 Converted Data for Sequential Multi-Day VRP:")
    print(f"  - Locations: {len(locations)}")
    print(f"  - Vehicles: {len(vehicles)}")
    
    # Import and run the sequential multi-day VRP solver
    try:
        spec = importlib.util.spec_from_file_location("vrp_multiday", 
                                                    os.path.join(os.path.dirname(__file__), 
                                                                "vrp_multiday_sequential.py"))
        vrp_multiday = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(vrp_multiday)
        
        print("\n📋 Testing Sequential Multi-Day VRP with overnight nodes...")
        
        # Create sequential VRP solver
        sequential_vrp = vrp_multiday.SequentialMultiDayVRP(vehicles, locations)
        
        # Use max_time from scenario vehicles instead of hardcoding
        # Get max_time from the first vehicle (assuming all vehicles use the same max_time)
        if vehicles and 'max_time' in vehicles[0]:
            sequential_vrp.daily_time_limit_minutes = vehicles[0]['max_time']
            print(f"  - Using vehicle max_time: {sequential_vrp.daily_time_limit_minutes} minutes per day")
        else:
            # Fall back to default
            sequential_vrp.daily_time_limit_minutes = 24 * 60  # Default to 24 hours
            print(f"  - Using default max_time: {sequential_vrp.daily_time_limit_minutes} minutes per day")
        
        # Solve the multi-day problem
        print("\n🚀 Solving sequential multi-day VRP...")
        solution = sequential_vrp.solve_sequential_multiday(max_days=7)
        
        if solution:
            print("\n✅ Sequential Multi-Day VRP solution found!")
            
            # Check if there are any overnight stays
            total_overnight_stays = 0
            overnight_locations = []
            
            for vehicle_id, route_data in solution['vehicle_routes'].items():
                if 'daily_routes' in route_data:
                    for day, day_route in route_data['daily_routes'].items():
                        if 'overnight_location' in day_route and day_route['overnight_location']:
                            overnight_location = day_route['overnight_location']
                            overnight_locations.append(overnight_location)
                            total_overnight_stays += 1
                            print(f"  🛏️ Vehicle {vehicle_id} - Day {day}: Overnight at {overnight_location}")
            
            print(f"\n📊 Total overnight stays: {total_overnight_stays}")
            
            # Analyze overnight positions in the solution
            print("\n📊 Analyzing overnight positions and route patterns:")
            
            # Get depot coordinates
            depot_coords = None
            for loc_id, loc in scenario.locations.items():
                if 'depot' in loc_id.lower():
                    depot_coords = (loc.x, loc.y)
                    print(f"  - Depot coordinates: {depot_coords}")
                    break
            
            if depot_coords:
                # Analyze all overnight positions
                all_overnight_positions = 0
                overnight_positions = []
                
                # Extract overnight positions from the solution
                for vehicle_id, route_data in solution['vehicle_routes'].items():
                    # Look for overnight positions in various formats
                    if 'overnight_positions' in route_data:
                        for day, pos in route_data['overnight_positions'].items():
                            x, y = pos
                            overnight_positions.append((day, vehicle_id, x, y))
                    
                    # Check for daily routes with overnight stops
                    if 'daily_routes' in route_data:
                        for day, day_route in route_data['daily_routes'].items():
                            if 'overnight_position' in day_route and day_route['overnight_position']:
                                pos = day_route['overnight_position']
                                if isinstance(pos, tuple) and len(pos) == 2:
                                    x, y = pos
                                    overnight_positions.append((day, vehicle_id, x, y))
                                elif isinstance(pos, dict) and 'x' in pos and 'y' in pos:
                                    x, y = pos['x'], pos['y']
                                    overnight_positions.append((day, vehicle_id, x, y))
                    
                    # Check for stops that are marked as overnight
                    # This handles the case where overnight positions are stored in the route stops
                    if 'stops' in route_data:
                        for stop in route_data['stops']:
                            if stop.get('is_overnight', False) and 'coordinates' in stop:
                                x, y = stop['coordinates']
                                day = stop.get('day', 'unknown')
                                overnight_positions.append((day, vehicle_id, x, y))
                                
                # Additionally, check all day solutions for overnight stops
                for day_num, day_solution in solution.get('daily_solutions', {}).items():
                    for vehicle_id, route in day_solution.get('routes', {}).items():
                        for stop in route.get('stops', []):
                            if stop.get('is_overnight', False) and 'coordinates' in stop:
                                x, y = stop['coordinates']
                                overnight_positions.append((day_num, vehicle_id, x, y))
                
                # Check if any overnight positions exist
                print("\n🛏️ Overnight positions found:")
                for day, vehicle_id, x, y in overnight_positions:
                    # Calculate distance from depot
                    if depot_coords:
                        distance_from_depot = ((x - depot_coords[0])**2 + 
                                             (y - depot_coords[1])**2)**0.5
                        
                        all_overnight_positions += 1
                        print(f"  • Day {day}, {vehicle_id}: ({x:.4f}, {y:.4f})")
                        print(f"    Distance from depot: {distance_from_depot:.2f} km")
                
                print(f"\n📊 Found {all_overnight_positions} overnight positions across all days")
            else:
                print("❌ Could not perform route analysis: missing depot coordinates")
                
            # Plot the solution
            try:
                plot_filename = sequential_vrp.plot_sequential_solution(solution, 
                                                               "Sequential Multi-Day VRP - MODA Furgoni Scenario")
                print(f"\n📊 Solution plotted and saved as: {plot_filename}")
            except Exception as plot_error:
                print(f"❌ Error plotting solution: {plot_error}")
            
            # Create interactive HTML map visualization
            try:
                print("\n🗺️ Creating interactive HTML map visualization...")
                html_map_path = create_interactive_vrp_map(scenario, solution, sequential_vrp)
                if html_map_path:
                    print(f"📊 Interactive map saved as: {html_map_path}")
                else:
                    print("❌ Could not create interactive map")
            except Exception as map_error:
                print(f"❌ Error creating interactive map: {map_error}")
                import traceback
                traceback.print_exc()
                
            return solution
        else:
            print("❌ Failed to solve sequential multi-day VRP")
            return None
        
    except ImportError as e:
        print(f"❌ Could not import Sequential Multi-Day VRP solver: {e}")
    except Exception as e:
        print(f"❌ Error in Sequential Multi-Day VRP test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Choose which test to run
    test_type = "overnight"  # Options: "full", "quick", "overnight"
    
    if test_type == "full":
        test_overnight_scenario_comparison()
    elif test_type == "quick":
        test_single_constraint_level()
    elif test_type == "overnight":
        test_overnight_node_creation()
    else:
        print("Available tests: 'full', 'quick', 'overnight'")
        print("Set test_type variable to choose which test to run.")
