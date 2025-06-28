"""
Test Cached OSRM Sequential VRP
==============================

This test demonstrates the new cached OSRM routing functionality for the sequential VRP.
It shows how the route database is built and reused for faster subsequent runs.
"""

import time
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vrp_multiday_sequential import SequentialMultiDayVRP
from route_database import RouteDatabase
from vrp_scenarios import DEFAULT_TRUCK_SPEED_RATIOS

def test_cached_osrm_sequential():
    """Test the cached OSRM sequential VRP with a realistic scenario."""
    print("🧪 Testing Cached OSRM Sequential Multi-Day VRP")
    print("=" * 60)
    
    # Create a manageable test scenario (Italian cities)
    locations = [
        {'id': 'depot_milan', 'x': 9.1900, 'y': 45.4642, 'demand': 0, 'service_time': 0, 'address': 'Main Depot Milan'},
        
        # Northern destinations
        {'id': 'venice', 'x': 12.3155, 'y': 45.4408, 'demand': 150, 'volume_demand': 4.2, 'service_time': 30, 'address': 'Venice'},
        {'id': 'verona', 'x': 10.9916, 'y': 45.4384, 'demand': 120, 'volume_demand': 3.1, 'service_time': 25, 'address': 'Verona'},
        {'id': 'bergamo', 'x': 9.6696, 'y': 45.6983, 'demand': 85, 'volume_demand': 2.3, 'service_time': 20, 'address': 'Bergamo'},
        
        # Southern destinations  
        {'id': 'bologna', 'x': 11.3426, 'y': 44.4949, 'demand': 180, 'volume_demand': 4.8, 'service_time': 35, 'address': 'Bologna'},
        {'id': 'florence', 'x': 11.2558, 'y': 43.7696, 'demand': 200, 'volume_demand': 5.2, 'service_time': 40, 'address': 'Florence'},
        {'id': 'genoa', 'x': 8.9463, 'y': 44.4056, 'demand': 160, 'volume_demand': 4.1, 'service_time': 32, 'address': 'Genoa'},
        
        # Distant destinations
        {'id': 'rome', 'x': 12.4964, 'y': 41.9028, 'demand': 250, 'volume_demand': 6.1, 'service_time': 45, 'address': 'Rome'},
        {'id': 'naples', 'x': 14.2681, 'y': 40.8518, 'demand': 170, 'volume_demand': 4.4, 'service_time': 35, 'address': 'Naples'},
    ]
    
    vehicles = [
        {'id': 'truck_heavy_1', 'capacity': 500, 'volume_capacity': 12.0, 'cost_per_km': 1.80, 'max_daily_km': 600, 
         'truck_speed_ratios': DEFAULT_TRUCK_SPEED_RATIOS['heavy']},
        {'id': 'truck_standard_1', 'capacity': 350, 'volume_capacity': 10.0, 'cost_per_km': 1.20, 'max_daily_km': 700,
         'truck_speed_ratios': DEFAULT_TRUCK_SPEED_RATIOS['standard']},
        {'id': 'van_light_1', 'capacity': 250, 'volume_capacity': 8.0, 'cost_per_km': 0.95, 'max_daily_km': 750,
         'truck_speed_ratios': DEFAULT_TRUCK_SPEED_RATIOS['standard']},
    ]
    
    print(f"📍 Test scenario: {len(locations)} locations, {len(vehicles)} vehicles")
    print(f"📦 Total demand: {sum(loc.get('demand', 0) for loc in locations)} units")
    print(f"📊 Total volume: {sum(loc.get('volume_demand', 0) for loc in locations):.1f} m³")
    
    # Database filename for this test
    db_path = "test_italy_routes.db"
    
    # Show initial database stats
    if os.path.exists(db_path):
        print(f"\n📁 Using existing route database: {db_path}")
        route_db = RouteDatabase(db_path)
        route_db._print_cache_stats()
    else:
        print(f"\n📁 Creating new route database: {db_path}")
        route_db = RouteDatabase(db_path)
        route_db._print_cache_stats()
    
    # Test 1: First run (will populate database)
    print(f"\n" + "=" * 60)
    print(f"🔄 First Run: Building Route Database")
    print(f"=" * 60)
    
    start_time = time.time()
    sequential_vrp = SequentialMultiDayVRP(vehicles, locations, use_truck_speeds=True, db_path=db_path)
    first_run_time = time.time() - start_time
    
    print(f"\n⏱️ First run initialization time: {first_run_time:.1f} seconds")
    
    # Get cache stats after first run
    cache_stats = sequential_vrp.distance_calculator.get_cache_stats()
    print(f"📊 Routes in database after first run: {cache_stats['total_routes']}")
    
    # Test 2: Second run (should be much faster due to caching)
    print(f"\n" + "=" * 60)
    print(f"🚀 Second Run: Using Cached Routes")
    print(f"=" * 60)
    
    start_time = time.time()
    sequential_vrp_2 = SequentialMultiDayVRP(vehicles, locations, use_truck_speeds=True, db_path=db_path)
    second_run_time = time.time() - start_time
    
    print(f"\n⏱️ Second run initialization time: {second_run_time:.1f} seconds")
    print(f"🚀 Speed improvement: {first_run_time/second_run_time:.1f}x faster!")
    
    # Test 3: Actually solve the VRP problem
    print(f"\n" + "=" * 60)
    print(f"🧮 Solving Sequential Multi-Day VRP")
    print(f"=" * 60)
    
    start_time = time.time()
    solution = sequential_vrp_2.solve_sequential_multiday(max_days=5)
    solve_time = time.time() - start_time
    
    if solution:
        print(f"\n✅ Sequential Multi-Day VRP solution found!")
        print(f"⏱️ Total solve time: {solve_time:.1f} seconds")
        print(f"📅 Days used: {solution['total_days']}")
        
        # Print vehicle routes summary
        print(f"\n🚛 Vehicle Routes Summary:")
        total_distance = 0
        total_cost = 0
        
        for vehicle_id, route_data in solution['vehicle_routes'].items():
            distance = route_data['total_distance']
            overnight_stays = route_data['total_overnight_stays']
            days_active = route_data['days_active']
            
            # Calculate cost
            vehicle = next((v for v in vehicles if v['id'] == vehicle_id), None)
            cost = distance * vehicle['cost_per_km'] if vehicle else 0
            
            total_distance += distance
            total_cost += cost
            
            print(f"  {vehicle_id}: {distance:.1f}km, €{cost:.2f}, {days_active} days, {overnight_stays} overnight stays")
        
        print(f"\n📊 Total Distance: {total_distance:.1f}km")
        print(f"💰 Total Cost: €{total_cost:.2f}")
        
        # Show final database stats
        final_stats = sequential_vrp_2.distance_calculator.get_cache_stats()
        print(f"\n📁 Final database stats:")
        print(f"  - Total routes cached: {final_stats['total_routes']}")
        print(f"  - Database size: {final_stats['database_size_mb']:.2f} MB")
        
        # Try plotting if possible
        try:
            plot_filename = sequential_vrp_2.plot_sequential_solution(solution, 
                                                                   "Cached OSRM Sequential VRP - Italy Test")
            print(f"📊 Solution plotted: {plot_filename}")
        except Exception as e:
            print(f"⚠️ Could not create plot: {e}")
        
        return solution
    else:
        print("❌ Failed to solve sequential multi-day VRP")
        return None

def test_database_growth():
    """Test how the database grows with different scenarios."""
    print(f"\n" + "=" * 60)
    print(f"📈 Testing Database Growth with Multiple Scenarios")
    print(f"=" * 60)
    
    db_path = "multi_scenario_routes.db"
    
    # Scenario 1: Northern Italy
    print(f"\n🇮🇹 Scenario 1: Northern Italy")
    locations_north = [
        {'id': 'depot_milan', 'x': 9.1900, 'y': 45.4642, 'demand': 0, 'service_time': 0},
        {'id': 'turin', 'x': 7.6869, 'y': 45.0703, 'demand': 100, 'service_time': 25},
        {'id': 'venice', 'x': 12.3155, 'y': 45.4408, 'demand': 120, 'service_time': 30},
        {'id': 'verona', 'x': 10.9916, 'y': 45.4384, 'demand': 90, 'service_time': 20},
    ]
    
    vehicles_small = [
        {'id': 'van_1', 'capacity': 200, 'volume_capacity': 8.0, 'cost_per_km': 1.0,
         'truck_speed_ratios': DEFAULT_TRUCK_SPEED_RATIOS['standard']}
    ]
    
    vrp_north = SequentialMultiDayVRP(vehicles_small, locations_north, use_truck_speeds=True, db_path=db_path)
    stats_after_north = vrp_north.distance_calculator.get_cache_stats()
    print(f"  Routes after Northern Italy: {stats_after_north['total_routes']}")
    
    # Scenario 2: Central Italy  
    print(f"\n🏛️ Scenario 2: Central Italy")
    locations_central = [
        {'id': 'depot_florence', 'x': 11.2558, 'y': 43.7696, 'demand': 0, 'service_time': 0},
        {'id': 'rome', 'x': 12.4964, 'y': 41.9028, 'demand': 150, 'service_time': 35},
        {'id': 'bologna', 'x': 11.3426, 'y': 44.4949, 'demand': 110, 'service_time': 25},
        {'id': 'perugia', 'x': 12.3889, 'y': 43.1122, 'demand': 80, 'service_time': 20},
    ]
    
    vrp_central = SequentialMultiDayVRP(vehicles_small, locations_central, use_truck_speeds=True, db_path=db_path)
    stats_after_central = vrp_central.distance_calculator.get_cache_stats()
    print(f"  Routes after Central Italy: {stats_after_central['total_routes']}")
    
    # Scenario 3: Southern Italy
    print(f"\n🌋 Scenario 3: Southern Italy")  
    locations_south = [
        {'id': 'depot_naples', 'x': 14.2681, 'y': 40.8518, 'demand': 0, 'service_time': 0},
        {'id': 'bari', 'x': 16.8719, 'y': 41.1171, 'demand': 130, 'service_time': 30},
        {'id': 'palermo', 'x': 13.3614, 'y': 38.1157, 'demand': 140, 'service_time': 35},
        {'id': 'catania', 'x': 15.0870, 'y': 37.5079, 'demand': 100, 'service_time': 25},
    ]
    
    vrp_south = SequentialMultiDayVRP(vehicles_small, locations_south, use_truck_speeds=True, db_path=db_path)
    stats_final = vrp_south.distance_calculator.get_cache_stats()
    print(f"  Routes after Southern Italy: {stats_final['total_routes']}")
    
    print(f"\n📈 Database Growth Summary:")
    print(f"  - Started with: 0 routes")
    print(f"  - After Northern Italy: {stats_after_north['total_routes']} routes")
    print(f"  - After Central Italy: {stats_after_central['total_routes']} routes")  
    print(f"  - After Southern Italy: {stats_final['total_routes']} routes")
    print(f"  - Final database size: {stats_final['database_size_mb']:.2f} MB")
    print(f"  - Total unique routes: {stats_final['total_routes']}")

if __name__ == "__main__":
    print("🛣️ Cached OSRM Sequential VRP Test Suite")
    print("=" * 80)
    
    # Test 1: Basic cached OSRM functionality
    solution = test_cached_osrm_sequential()
    
    # Test 2: Database growth across scenarios  
    test_database_growth()
    
    print(f"\n" + "=" * 80)
    print(f"✅ Cached OSRM Sequential VRP testing completed!")
    print(f"   Route databases have been created and will speed up future runs.")
    print(f"   The more scenarios you test, the more comprehensive the database becomes.")
