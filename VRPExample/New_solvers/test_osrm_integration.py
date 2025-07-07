#!/usr/bin/env python3
"""
Test script for the OSRM-integrated sequential multiday solver.
This tests the enhanced solver with correct overnight logic and OSRM route caching.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vrp_multiday_sequential import SequentialMultiDayVRP
import time

def test_osrm_integrated_solver():
    """Test the OSRM-integrated sequential solver with a small scenario."""
    print("🧪 TESTING OSRM-INTEGRATED SEQUENTIAL SOLVER")
    print("=" * 60)
    
    # Create a simple test scenario with realistic Swiss locations
    locations = [
        {'id': 'depot', 'x': 8.5417, 'y': 47.3769, 'demand': 0, 'service_time': 0, 'address': 'Main Depot Zurich'},
        {'id': 'basel', 'x': 7.5886, 'y': 47.5596, 'demand': 120, 'service_time': 25, 'address': 'Basel'},
        {'id': 'bern', 'x': 7.4474, 'y': 46.9481, 'demand': 110, 'service_time': 22, 'address': 'Bern'},
        {'id': 'geneva', 'x': 6.1432, 'y': 46.2044, 'demand': 150, 'service_time': 30, 'address': 'Geneva'},
        {'id': 'lausanne', 'x': 6.6323, 'y': 46.5197, 'demand': 90, 'service_time': 20, 'address': 'Lausanne'},
        {'id': 'lucerne', 'x': 8.3093, 'y': 47.0502, 'demand': 80, 'service_time': 18, 'address': 'Lucerne'},
    ]
    
    # Create vehicles with different types
    vehicles = [
        {
            'id': 'furgone_1',
            'capacity': 1000,
            'volume_capacity': 15.0,
            'cost_per_km': 1.2,
            'max_time': 480,  # 8 hours per day
            'vehicle_type': 'furgone'
        },
        {
            'id': 'furgone_2', 
            'capacity': 1000,
            'volume_capacity': 15.0,
            'cost_per_km': 1.2,
            'max_time': 480,
            'vehicle_type': 'furgone'
        }
    ]
    
    print(f"📊 Test Scenario:")
    print(f"  - Locations: {len(locations)} (including depot)")
    print(f"  - Vehicles: {len(vehicles)}")
    print(f"  - Total demand: {sum(loc.get('demand', 0) for loc in locations)} kg")
    
    # Initialize the solver with OSRM caching
    print(f"\n🚛 Initializing OSRM-integrated solver...")
    solver = SequentialMultiDayVRP(
        vehicles=vehicles,
        locations=locations,
        use_truck_speeds=True,
        db_path="moda_routes.db"  # Use the existing cache
    )
    
    # Show cache statistics after initialization
    if hasattr(solver.distance_calculator, 'route_db'):
        cache_stats = solver.distance_calculator.route_db.get_cache_stats()
        print(f"📊 Route Cache: {cache_stats['total_routes']} routes, {cache_stats['database_size_mb']:.2f} MB")
    
    # Solve the problem
    print(f"\n🚀 Solving sequential multi-day VRP...")
    start_time = time.time()
    
    try:
        solution = solver.solve_sequential_multiday(max_days=3)
        solve_time = time.time() - start_time
        
        if solution:
            print(f"\n✅ Solution found in {solve_time:.2f} seconds!")
            
            # Print solution summary
            total_days = len(solution.get('daily_solutions', {}))
            total_distance = solution.get('total_distance', 0)
            total_cost = solution.get('total_cost', 0)
            
            print(f"📊 Solution Summary:")
            print(f"  - Days required: {total_days}")
            print(f"  - Total distance: {total_distance:.2f} km")
            print(f"  - Total cost: €{total_cost:.2f}")
            
            # Check for overnight stops
            overnight_count = 0
            for day_num, day_data in solution.get('daily_solutions', {}).items():
                overnight_locations = day_data.get('overnight_locations', {})
                overnight_count += len(overnight_locations)
            
            print(f"  - Overnight stops: {overnight_count}")
            
            # Verify the overnight logic: vehicles with no deliveries shouldn't have unnecessary overnight stops
            print(f"\n🔍 Verifying overnight logic...")
            for day_num, day_data in solution.get('daily_solutions', {}).items():
                routes = day_data.get('routes', {})
                for vehicle_id, route_data in routes.items():
                    stops = route_data.get('stops', [])
                    
                    # Count actual customer deliveries (not depot, not overnight)
                    deliveries = sum(1 for stop in stops 
                                   if not stop.get('location_id', '').startswith('depot') 
                                   and not stop.get('is_overnight_node', False)
                                   and not stop.get('is_road_position', False))
                    
                    overnight_stops = sum(1 for stop in stops 
                                        if stop.get('is_overnight_node', False) or stop.get('is_road_position', False))
                    
                    if deliveries == 0 and overnight_stops > 0:
                        print(f"  ⚠️ Day {day_num}, {vehicle_id}: 0 deliveries but {overnight_stops} overnight stops")
                    elif deliveries > 0:
                        print(f"  ✅ Day {day_num}, {vehicle_id}: {deliveries} deliveries, {overnight_stops} overnight stops")
                    else:
                        print(f"  ✅ Day {day_num}, {vehicle_id}: 0 deliveries, 0 overnight stops")
            
            print(f"\n🎯 Test completed successfully!")
            
        else:
            print(f"\n❌ Failed to find solution in {solve_time:.2f} seconds")
            
    except Exception as e:
        solve_time = time.time() - start_time
        print(f"\n❌ Error during solving: {e}")
        import traceback
        traceback.print_exc()
    
    # Show final cache statistics
    if hasattr(solver.distance_calculator, 'route_db'):
        final_stats = solver.distance_calculator.route_db.get_cache_stats()
        print(f"\n📊 Final Cache Stats: {final_stats['total_routes']} routes, {final_stats['database_size_mb']:.2f} MB")

if __name__ == "__main__":
    test_osrm_integrated_solver()
