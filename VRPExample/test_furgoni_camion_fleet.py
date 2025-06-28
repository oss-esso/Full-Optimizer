"""
Test Mixed Fleet: Furgoni vs Camion
==================================

This test demonstrates the mixed vehicle fleet functionality using Italian naming conventions:
- "furgone" (pl. "furgoni") = standard light truck with standard speed ratios
- "camion" (pl. "camion") = heavy truck with reduced speed ratios due to weight restrictions

The test shows how each vehicle type applies different speed adjustments based on road composition.
"""

import time
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vrp_multiday_sequential import SequentialMultiDayVRP
from route_database import RouteDatabase
from vrp_scenarios import DEFAULT_TRUCK_SPEED_RATIOS

def test_furgoni_vs_camion_fleet():
    """Test mixed fleet with Italian vehicle naming: furgoni vs camion."""
    print("🇮🇹 Testing Mixed Fleet: Furgoni vs Camion")
    print("=" * 60)
    
    # Create a realistic Italian delivery scenario
    locations = [
        {'id': 'depot_milano', 'x': 9.1900, 'y': 45.4642, 'demand': 0, 'service_time': 0, 'address': 'Deposito Milano'},
        
        # Urban deliveries (good for furgoni)
        {'id': 'bergamo', 'x': 9.6696, 'y': 45.6983, 'demand': 120, 'volume_demand': 3.2, 'service_time': 25, 'address': 'Bergamo Centro'},
        {'id': 'brescia', 'x': 10.2287, 'y': 45.5416, 'demand': 90, 'volume_demand': 2.8, 'service_time': 20, 'address': 'Brescia'},
        {'id': 'monza', 'x': 9.2744, 'y': 45.5845, 'demand': 85, 'volume_demand': 2.5, 'service_time': 18, 'address': 'Monza'},
        
        # Medium distance deliveries
        {'id': 'verona', 'x': 10.9916, 'y': 45.4384, 'demand': 180, 'volume_demand': 4.8, 'service_time': 35, 'address': 'Verona'},
        {'id': 'padova', 'x': 11.8767, 'y': 45.4064, 'demand': 160, 'volume_demand': 4.2, 'service_time': 30, 'address': 'Padova'},
        
        # Long distance / heavy deliveries (good for camion)
        {'id': 'bologna', 'x': 11.3426, 'y': 44.4949, 'demand': 320, 'volume_demand': 8.5, 'service_time': 45, 'address': 'Bologna Deposito'},
        {'id': 'firenze', 'x': 11.2558, 'y': 43.7696, 'demand': 280, 'volume_demand': 7.8, 'service_time': 40, 'address': 'Firenze Centro'},
        {'id': 'genova', 'x': 8.9463, 'y': 44.4056, 'demand': 350, 'volume_demand': 9.2, 'service_time': 50, 'address': 'Porto di Genova'},
    ]
    
    # Mixed fleet: furgoni for light/urban, camion for heavy/long-distance
    vehicles = [
        # Light delivery vans (furgoni)
        {
            'id': 'furgone_leggero_1', 
            'capacity': 250, 
            'volume_capacity': 8.0, 
            'cost_per_km': 0.85, 
            'max_daily_km': 800,
            'truck_speed_ratios': DEFAULT_TRUCK_SPEED_RATIOS['standard'],
            'type': 'furgone'
        },
        {
            'id': 'furgone_medio_1', 
            'capacity': 400, 
            'volume_capacity': 12.0, 
            'cost_per_km': 1.10, 
            'max_daily_km': 750,
            'truck_speed_ratios': DEFAULT_TRUCK_SPEED_RATIOS['standard'],
            'type': 'furgone'
        },
        
        # Heavy trucks (camion)
        {
            'id': 'camion_pesante_1', 
            'capacity': 800, 
            'volume_capacity': 20.0, 
            'cost_per_km': 1.95, 
            'max_daily_km': 550,
            'truck_speed_ratios': DEFAULT_TRUCK_SPEED_RATIOS['heavy'],
            'type': 'camion'
        },
        {
            'id': 'camion_articolato_1', 
            'capacity': 1200, 
            'volume_capacity': 28.0, 
            'cost_per_km': 2.35, 
            'max_daily_km': 500,
            'truck_speed_ratios': DEFAULT_TRUCK_SPEED_RATIOS['heavy'],
            'type': 'camion'
        },
    ]
    
    print(f"📍 Scenario: {len(locations)} locations in Northern Italy")
    print(f"🚛 Fleet composition:")
    furgoni = [v for v in vehicles if v['type'] == 'furgone']
    camion = [v for v in vehicles if v['type'] == 'camion']
    print(f"  - {len(furgoni)} furgoni (standard speed ratios)")
    print(f"  - {len(camion)} camion (heavy truck speed ratios)")
    
    print(f"\n📦 Total demand: {sum(loc.get('demand', 0) for loc in locations)} units")
    print(f"📊 Total volume: {sum(loc.get('volume_demand', 0) for loc in locations):.1f} m³")
    
    # Use the same shared database as main tests
    db_path = "test_italy_routes.db"
    
    print(f"\n🗄️ Using shared route database: {db_path}")
    print(f"   (This will add any missing routes to the shared database)")
    
    # Create VRP instance
    print(f"\n" + "=" * 60)
    print(f"🔧 Initializing Mixed Fleet VRP")
    print(f"=" * 60)
    
    start_time = time.time()
    sequential_vrp = SequentialMultiDayVRP(vehicles, locations, use_truck_speeds=True, db_path=db_path)
    init_time = time.time() - start_time
    
    print(f"⏱️ Initialization time: {init_time:.1f} seconds")
    
    # Show speed ratio differences
    print(f"\n🚗 Vehicle Speed Ratio Comparison:")
    print(f"Standard ratios (furgoni): {DEFAULT_TRUCK_SPEED_RATIOS['standard']}")
    print(f"Heavy ratios (camion): {DEFAULT_TRUCK_SPEED_RATIOS['heavy']}")
    
    # Test travel time differences for the same route with different vehicles
    print(f"\n⏱️ Travel Time Comparison (Milano → Bologna):")
    
    # Get Milano and Bologna indices
    milano_idx = next(i for i, loc in enumerate(locations) if loc['id'] == 'depot_milano')
    bologna_idx = next(i for i, loc in enumerate(locations) if loc['id'] == 'bologna')
    
    for vehicle in vehicles:
        travel_time = sequential_vrp.get_vehicle_travel_time(milano_idx, bologna_idx, vehicle['id'])
        distance = sequential_vrp.distance_calculator.distance_matrix[milano_idx][bologna_idx]
        avg_speed = distance / (travel_time / 60) if travel_time > 0 else 0
        
        print(f"  {vehicle['id']}: {travel_time:.1f}min ({distance:.1f}km, avg {avg_speed:.1f}km/h)")
    
    # Solve the VRP
    print(f"\n" + "=" * 60)
    print(f"🧮 Solving Mixed Fleet Sequential VRP")
    print(f"=" * 60)
    
    start_time = time.time()
    solution = sequential_vrp.solve_sequential_multiday(max_days=3)
    solve_time = time.time() - start_time
    
    if solution:
        print(f"\n✅ Mixed Fleet VRP solution found!")
        print(f"⏱️ Total solve time: {solve_time:.1f} seconds")
        print(f"📅 Days used: {solution['total_days']}")
        
        # Analyze solution by vehicle type
        print(f"\n📊 Solution Analysis by Vehicle Type:")
        print(f"Solution keys: {list(solution.keys())}")
        
        # Check if we have the expected structure
        if 'vehicle_routes' not in solution:
            print("⚠️ No vehicle_routes found in solution")
            return solution
        
        furgoni_stats = {'total_distance': 0, 'total_cost': 0, 'vehicles_used': 0, 'deliveries': 0}
        camion_stats = {'total_distance': 0, 'total_cost': 0, 'vehicles_used': 0, 'deliveries': 0}
        
        for vehicle_id, route_data in solution['vehicle_routes'].items():
            vehicle = next((v for v in vehicles if v['id'] == vehicle_id), None)
            if not vehicle:
                continue
                
            print(f"Route data keys for {vehicle_id}: {list(route_data.keys())}")
            
            distance = route_data.get('total_distance', 0)
            cost = distance * vehicle['cost_per_km']
            
            # Count deliveries based on available data structure
            deliveries = 0
            if 'daily_routes' in route_data:
                deliveries = len([stop for day in route_data['daily_routes'] for stop in day if stop != milano_idx])
            elif 'full_route' in route_data:
                # Count unique customer visits (exclude depot visits)
                deliveries = len([stop for stop in route_data['full_route'] if stop != milano_idx])
            elif 'route' in route_data:
                deliveries = len([stop for stop in route_data['route'] if stop != milano_idx])
            
            if vehicle['type'] == 'furgone':
                furgoni_stats['total_distance'] += distance
                furgoni_stats['total_cost'] += cost
                furgoni_stats['vehicles_used'] += 1
                furgoni_stats['deliveries'] += deliveries
            else:  # camion
                camion_stats['total_distance'] += distance
                camion_stats['total_cost'] += cost
                camion_stats['vehicles_used'] += 1
                camion_stats['deliveries'] += deliveries
            
            print(f"  {vehicle_id} ({vehicle['type']}): {distance:.1f}km, €{cost:.2f}, {deliveries} deliveries")
        
        print(f"\n🚛 Fleet Performance Summary:")
        total_distance = furgoni_stats['total_distance'] + camion_stats['total_distance']
        total_cost = furgoni_stats['total_cost'] + camion_stats['total_cost']
        
        if furgoni_stats['vehicles_used'] > 0:
            print(f"  Furgoni ({furgoni_stats['vehicles_used']} vehicles):")
            print(f"    - Distance: {furgoni_stats['total_distance']:.1f}km ({100*furgoni_stats['total_distance']/total_distance:.1f}%)")
            print(f"    - Cost: €{furgoni_stats['total_cost']:.2f} ({100*furgoni_stats['total_cost']/total_cost:.1f}%)")
            print(f"    - Deliveries: {furgoni_stats['deliveries']}")
            print(f"    - Avg cost/km: €{furgoni_stats['total_cost']/furgoni_stats['total_distance']:.2f}")
        
        if camion_stats['vehicles_used'] > 0:
            print(f"  Camion ({camion_stats['vehicles_used']} vehicles):")
            print(f"    - Distance: {camion_stats['total_distance']:.1f}km ({100*camion_stats['total_distance']/total_distance:.1f}%)")
            print(f"    - Cost: €{camion_stats['total_cost']:.2f} ({100*camion_stats['total_cost']/total_cost:.1f}%)")
            print(f"    - Deliveries: {camion_stats['deliveries']}")
            if camion_stats['total_distance'] > 0:
                print(f"    - Avg cost/km: €{camion_stats['total_cost']/camion_stats['total_distance']:.2f}")
            else:
                print(f"    - Avg cost/km: €0.00 (no distance traveled)")
        
        print(f"\n💰 Total Fleet Cost: €{total_cost:.2f}")
        print(f"🛣️ Total Distance: {total_distance:.1f}km")
        
        # Show database stats
        cache_stats = sequential_vrp.distance_calculator.get_cache_stats()
        print(f"\n🗄️ Route Database Stats:")
        print(f"  - Cached routes: {cache_stats['total_routes']}")
        print(f"  - Database size: {cache_stats['database_size_mb']:.2f} MB")
        
        # Try to create plot
        try:
            plot_filename = sequential_vrp.plot_sequential_solution(solution, 
                                                                 "Mixed Fleet: Furgoni vs Camion - Italy")
            print(f"📊 Solution plotted: {plot_filename}")
        except Exception as e:
            print(f"⚠️ Could not create plot: {e}")
        
        return solution
        
    else:
        print("❌ Failed to solve mixed fleet VRP")
        return None

def analyze_speed_ratio_impact():
    """Analyze the impact of different speed ratios on route planning."""
    print(f"\n" + "=" * 60)
    print(f"🔍 Analyzing Speed Ratio Impact")
    print(f"=" * 60)
    
    # Simple 3-city route for clear comparison
    locations = [
        {'id': 'depot', 'x': 9.1900, 'y': 45.4642, 'demand': 0, 'service_time': 0},
        {'id': 'city_a', 'x': 11.3426, 'y': 44.4949, 'demand': 100, 'service_time': 30},  # Bologna
        {'id': 'city_b', 'x': 12.4964, 'y': 41.9028, 'demand': 100, 'service_time': 30},  # Rome
    ]
    
    # Compare same capacity vehicles with different speed ratios
    test_vehicles = [
        {
            'id': 'furgone_test', 
            'capacity': 500, 
            'volume_capacity': 15.0, 
            'cost_per_km': 1.50, 
            'max_daily_km': 600,
            'truck_speed_ratios': DEFAULT_TRUCK_SPEED_RATIOS['standard']
        },
        {
            'id': 'camion_test', 
            'capacity': 500, 
            'volume_capacity': 15.0, 
            'cost_per_km': 1.50, 
            'max_daily_km': 600,
            'truck_speed_ratios': DEFAULT_TRUCK_SPEED_RATIOS['heavy']
        },
    ]
    
    vrp = SequentialMultiDayVRP(test_vehicles, locations, use_truck_speeds=True, 
                               db_path="speed_ratio_test.db")
    
    print(f"📋 Route Analysis: Depot → Bologna → Rome → Depot")
    print(f"{'Vehicle':<15} {'Total Time':<12} {'Speed Penalty':<15} {'Efficiency':<12}")
    print(f"-" * 55)
    
    depot_idx = 0
    bologna_idx = 1
    rome_idx = 2
    
    for vehicle in test_vehicles:
        vehicle_id = vehicle['id']
        
        # Calculate route times
        depot_to_bologna = vrp.get_vehicle_travel_time(depot_idx, bologna_idx, vehicle_id)
        bologna_to_rome = vrp.get_vehicle_travel_time(bologna_idx, rome_idx, vehicle_id)
        rome_to_depot = vrp.get_vehicle_travel_time(rome_idx, depot_idx, vehicle_id)
        
        total_time = depot_to_bologna + bologna_to_rome + rome_to_depot
        
        # Calculate distances for comparison
        depot_to_bologna_dist = vrp.distance_calculator.distance_matrix[depot_idx][bologna_idx]
        bologna_to_rome_dist = vrp.distance_calculator.distance_matrix[bologna_idx][rome_idx]
        rome_to_depot_dist = vrp.distance_calculator.distance_matrix[rome_idx][depot_idx]
        
        total_distance = depot_to_bologna_dist + bologna_to_rome_dist + rome_to_depot_dist
        avg_speed = total_distance / (total_time / 60) if total_time > 0 else 0
        
        # Compare to baseline (assuming 50 km/h average without penalties)
        baseline_time = (total_distance / 50) * 60  # minutes
        speed_penalty = ((total_time - baseline_time) / baseline_time) * 100 if baseline_time > 0 else 0
        
        efficiency = 100 - speed_penalty
        
        vehicle_type = "Furgone" if "furgone" in vehicle_id else "Camion"
        print(f"{vehicle_type:<15} {total_time:.1f}min{'':<6} {speed_penalty:+.1f}%{'':<9} {efficiency:.1f}%")
    
    print(f"\n💡 Speed Ratio Impact:")
    print(f"  - Furgoni use standard speed ratios (better on highways)")
    print(f"  - Camion use heavy truck ratios (slower on all road types)")
    print(f"  - Road composition affects both, but camion are penalized more")

if __name__ == "__main__":
    print("🇮🇹 Mixed Fleet Test: Furgoni vs Camion")
    print("=" * 80)
    
    # Test 1: Mixed fleet optimization
    solution = test_furgoni_vs_camion_fleet()
    
    # Test 2: Speed ratio impact analysis
    analyze_speed_ratio_impact()
    
    print(f"\n" + "=" * 80)
    print(f"✅ Mixed Fleet Testing Completed!")
    print(f"   This demonstrates how furgoni and camion use different speed ratios")
    print(f"   based on vehicle weight and road restrictions in Italy.")
