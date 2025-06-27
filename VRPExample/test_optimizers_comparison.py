"""
Test file to compare the two VRP optimizers on the Cormano scenario.

This test compares:
1. CleanVRPOptimizer (original from vrp_optimizer_clean.py)
2. CleanVRPOptimizer (OSM-enhanced copy from vrp_optimizer_clean_copy.py)

Both are tested on the Cormano scenario to evaluate:
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

def test_cormano_scenario_comparison():
    """Compare both optimizers on the Cormano scenario."""
    print("🧪 VRP Optimizers Comparison Test on Cormano Scenario")
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
    
    try:
        from vrp_optimizer_clean_copy import CleanVRPOptimizer as OSMOptimizer
        
        for level in constraint_levels:
            print(f"\n--- Testing constraint level: {level} ---")
            
            optimizer = OSMOptimizer(vehicles=vehicles_dicts, locations=None, vrp_instance=scenario)
            optimizer.ride_requests = scenario.ride_requests
            
            start_time = time.time()
            try:
                result = optimizer.solve(constraint_level=level, verbose=False)
                solve_time = time.time() - start_time
                
                if result:
                    solution = result
                    success = solution is not None
                    total_distance = solution.get('total_distance', 0) if solution else 0
                    total_cost = solution.get('total_cost', 0) if solution else 0
                    
                    results[f"osm_{level}"] = {
                        'success': success,
                        'status': 'SUCCESS',
                        'solve_time': solve_time,
                        'total_distance': total_distance,
                        'total_cost': total_cost,
                        'optimizer': 'osm',
                        'level': level
                    }
                    
                    print(f"✅ SUCCESS - Distance: {total_distance}km, Cost: €{total_cost:.2f}, Time: {solve_time:.1f}s")
                else:
                    results[f"osm_{level}"] = {
                        'success': False,
                        'status': 'FAILED',
                        'solve_time': solve_time,
                        'optimizer': 'osm',
                        'level': level
                    }
                    print(f"❌ FAILED - Time: {solve_time:.1f}s")
                    
            except Exception as e:
                solve_time = time.time() - start_time
                error_msg = str(e)
                if "Weight" in error_msg and "Map key not found" in error_msg:
                    error_msg = "Missing Weight dimension (OSM optimizer implementation issue)"
                
                results[f"osm_{level}"] = {
                    'success': False,
                    'status': f'ERROR: {error_msg}',
                    'solve_time': solve_time,
                    'optimizer': 'osm',
                    'level': level
                }
                print(f"❌ ERROR: {error_msg} - Time: {solve_time:.1f}s")
                
    except ImportError as e:
        print(f"❌ Could not import OSM optimizer: {e}")
        # Fill in failed results for all levels
        for level in constraint_levels:
            results[f"osm_{level}"] = {
                'success': False,
                'status': f'IMPORT ERROR: {str(e)}',
                'solve_time': 0,
                'optimizer': 'osm',
                'level': level
            }
    
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
        from vrp_scenarios import create_cormano_scenario
    except ImportError:
        print("❌ Error: Could not import create_cormano_scenario")
        return
    
    scenario = create_cormano_scenario()
    
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

if __name__ == "__main__":
    # Choose which test to run
    test_type = "full"  # Options: "full", "quick"
    
    if test_type == "full":
        test_cormano_scenario_comparison()
    elif test_type == "quick":
        test_single_constraint_level()
    else:
        print("Available tests: 'full', 'quick'")
        print("Set test_type variable to choose which test to run.")
