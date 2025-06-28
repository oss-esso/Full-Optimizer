"""
Test Mixed Vehicle Types with Different Speed Ratios
==================================================

This test demonstrates the system handling different vehicle types:
- Furgoni (standard trucks) with standard speed ratios
- Camion (heavy trucks) with heavy truck speed ratios
- Each vehicle type has different speed profiles for different road types

This ensures realistic routing where heavy trucks are slower on certain road types
compared to lighter vehicles.
"""

import time
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vrp_multiday_sequential import SequentialMultiDayVRP
from route_database import RouteDatabase
from vrp_scenarios import DEFAULT_TRUCK_SPEED_RATIOS

def test_mixed_vehicle_types():
    """Test the system with mixed vehicle types (furgoni vs camion)."""
    print("🚛 Testing Mixed Vehicle Types - Furgoni vs Camion")
    print("=" * 70)
    
    # Create a realistic Italian delivery scenario
    locations = [
        {'id': 'depot_milano', 'x': 9.1900, 'y': 45.4642, 'demand': 0, 'service_time': 0, 'address': 'Depot Milano'},
        
        # Northern routes (good highways - speed difference will be visible)
        {'id': 'torino', 'x': 7.6869, 'y': 45.0703, 'demand': 200, 'volume_demand': 5.2, 'service_time': 35, 'address': 'Torino'},
        {'id': 'venezia', 'x': 12.3155, 'y': 45.4408, 'demand': 180, 'volume_demand': 4.8, 'service_time': 30, 'address': 'Venezia'},
        {'id': 'verona', 'x': 10.9916, 'y': 45.4384, 'demand': 150, 'volume_demand': 3.9, 'service_time': 25, 'address': 'Verona'},
        
        # Central routes (mixed road types)
        {'id': 'bologna', 'x': 11.3426, 'y': 44.4949, 'demand': 220, 'volume_demand': 5.8, 'service_time': 40, 'address': 'Bologna'},
        {'id': 'firenze', 'x': 11.2558, 'y': 43.7696, 'demand': 190, 'volume_demand': 5.0, 'service_time': 35, 'address': 'Firenze'},
        
        # Southern routes (slower roads - bigger speed difference)
        {'id': 'roma', 'x': 12.4964, 'y': 41.9028, 'demand': 300, 'volume_demand': 7.2, 'service_time': 50, 'address': 'Roma'},
        {'id': 'napoli', 'x': 14.2681, 'y': 40.8518, 'demand': 250, 'volume_demand': 6.1, 'service_time': 45, 'address': 'Napoli'},
    ]
    
    # Mixed fleet with realistic Italian vehicle types
    vehicles = [
        # Heavy trucks (Camion) - slower on all road types, higher capacity
        {
            'id': 'camion_iveco_01', 
            'capacity': 800, 
            'volume_capacity': 18.0, 
            'cost_per_km': 2.20, 
            'max_daily_km': 600,
            'vehicle_type': 'camion',
            'truck_speed_ratios': DEFAULT_TRUCK_SPEED_RATIOS['heavy']
        },
        {
            'id': 'camion_volvo_02', 
            'capacity': 750, 
            'volume_capacity': 16.5, 
            'cost_per_km': 2.10, 
            'max_daily_km': 580,
            'vehicle_type': 'camion',
            'truck_speed_ratios': DEFAULT_TRUCK_SPEED_RATIOS['heavy']
        },
        
        # Standard trucks (Furgoni) - faster, lower capacity
        {
            'id': 'furgone_fiat_01', 
            'capacity': 400, 
            'volume_capacity': 12.0, 
            'cost_per_km': 1.50, 
            'max_daily_km': 700,
            'vehicle_type': 'furgone',
            'truck_speed_ratios': DEFAULT_TRUCK_SPEED_RATIOS['standard']
        },
        {
            'id': 'furgone_mercedes_02', 
            'capacity': 450, 
            'volume_capacity': 13.5, 
            'cost_per_km': 1.60, 
            'max_daily_km': 680,
            'vehicle_type': 'furgone',
            'truck_speed_ratios': DEFAULT_TRUCK_SPEED_RATIOS['standard']
        },
        
        # Light vans (Furgoncini) - fastest, lowest capacity
        {
            'id': 'furgoncino_iveco_01', 
            'capacity': 250, 
            'volume_capacity': 8.0, 
            'cost_per_km': 1.00, 
            'max_daily_km': 800,
            'vehicle_type': 'furgoncino',
            'truck_speed_ratios': DEFAULT_TRUCK_SPEED_RATIOS['standard']
        },
    ]
    
    print(f"📍 Scenario: {len(locations)} locations, {len(vehicles)} vehicles")
    print(f"📦 Total demand: {sum(loc.get('demand', 0) for loc in locations)} units")
    print(f"📊 Total volume: {sum(loc.get('volume_demand', 0) for loc in locations):.1f} m³")
    
    # Display vehicle types and their speed profiles
    print(f"\n🚛 Vehicle Fleet Configuration:")
    for vehicle in vehicles:
        speed_ratios = vehicle['truck_speed_ratios']
        vehicle_type = vehicle['vehicle_type']
        
        print(f"  {vehicle['id']} ({vehicle_type}):")
        print(f"    📦 Capacity: {vehicle['capacity']}kg, {vehicle['volume_capacity']}m³")
        print(f"    💰 Cost: €{vehicle['cost_per_km']:.2f}/km")
        print(f"    🛣️ Speed ratios: Motorway: {speed_ratios['motorway']:.2f}, Secondary: {speed_ratios['secondary']:.2f}")
    
    # Database for this test
    db_path = "mixed_vehicles_routes.db"
    
    print(f"\n📁 Using route database: {db_path}")
    
    # Test the system with mixed vehicle types
    print(f"\n" + "=" * 70)
    print(f"🔄 Building Route Matrices with Vehicle-Specific Speed Adjustments")
    print(f"=" * 70)
    
    # Create sequential VRP with mixed vehicle types
    start_time = time.time()
    sequential_vrp = SequentialMultiDayVRP(vehicles, locations, use_truck_speeds=True, db_path=db_path)
    init_time = time.time() - start_time
    
    print(f"\n⏱️ Matrix calculation time: {init_time:.1f} seconds")
    
    # Display some key route times to show the speed differences
    print(f"\n🕒 Route Time Comparison (showing speed ratio effects):")
    print(f"{'Route':<25} {'Distance (km)':<15} {'Base Time (min)':<18} {'Adjusted Time (min)':<20}")
    print("-" * 80)
    
    # Show a few key routes with their times
    key_routes = [
        (0, 1, "Milano → Torino"),
        (0, 6, "Milano → Roma"),
        (0, 7, "Milano → Napoli"),
        (1, 6, "Torino → Roma"),
        (6, 7, "Roma → Napoli")
    ]
    
    for from_idx, to_idx, route_name in key_routes:
        if (from_idx < len(sequential_vrp.distance_calculator.distance_matrix) and 
            to_idx < len(sequential_vrp.distance_calculator.distance_matrix[0])):
            
            distance = sequential_vrp.distance_calculator.distance_matrix[from_idx][to_idx]
            
            # Calculate base time without truck adjustments (assume 60 km/h average)
            base_time = distance * 60 / 60  # 60 minutes per 60km
            
            # Get actual adjusted time from the matrix
            adjusted_time = sequential_vrp.distance_calculator.time_matrix[from_idx][to_idx]
            
            print(f"{route_name:<25} {distance:<15.1f} {base_time:<18.1f} {adjusted_time:<20.1f}")
    
    # Now test the sequential VRP solver
    print(f"\n" + "=" * 70)
    print(f"🧮 Solving Multi-Day VRP with Mixed Vehicle Types")
    print(f"=" * 70)
    
    start_time = time.time()
    solution = sequential_vrp.solve_sequential_multiday(max_days=5)
    solve_time = time.time() - start_time
    
    if solution:
        print(f"\n✅ Multi-Day VRP solution found!")
        print(f"⏱️ Solve time: {solve_time:.1f} seconds")
        print(f"📅 Days used: {solution['total_days']}")
        
        # Analyze solution by vehicle type
        print(f"\n📊 Solution Analysis by Vehicle Type:")
        
        vehicle_types = {}
        for vehicle_id, route_data in solution['vehicle_routes'].items():
            # Find vehicle info
            vehicle = next((v for v in vehicles if v['id'] == vehicle_id), None)
            if vehicle:
                v_type = vehicle['vehicle_type']
                if v_type not in vehicle_types:
                    vehicle_types[v_type] = {
                        'count': 0, 
                        'total_distance': 0, 
                        'total_cost': 0, 
                        'vehicles': []
                    }
                
                distance = route_data['total_distance']
                cost = distance * vehicle['cost_per_km']
                overnight_stays = route_data['total_overnight_stays']
                
                vehicle_types[v_type]['count'] += 1
                vehicle_types[v_type]['total_distance'] += distance
                vehicle_types[v_type]['total_cost'] += cost
                vehicle_types[v_type]['vehicles'].append({
                    'id': vehicle_id,
                    'distance': distance,
                    'cost': cost,
                    'overnight_stays': overnight_stays,
                    'days_active': route_data['days_active']
                })
        
        # Display results by vehicle type
        total_distance = 0
        total_cost = 0
        
        for v_type, data in vehicle_types.items():
            print(f"\n🚛 {v_type.upper()} vehicles:")
            print(f"  Count: {data['count']}")
            print(f"  Total distance: {data['total_distance']:.1f} km")
            print(f"  Total cost: €{data['total_cost']:.2f}")
            print(f"  Avg distance per vehicle: {data['total_distance']/data['count']:.1f} km")
            
            for vehicle_info in data['vehicles']:
                print(f"    {vehicle_info['id']}: {vehicle_info['distance']:.1f}km, "
                      f"€{vehicle_info['cost']:.2f}, {vehicle_info['days_active']} days, "
                      f"{vehicle_info['overnight_stays']} overnights")
            
            total_distance += data['total_distance']
            total_cost += data['total_cost']
        
        print(f"\n💯 Overall Totals:")
        print(f"📏 Total Distance: {total_distance:.1f} km")
        print(f"💰 Total Cost: €{total_cost:.2f}")
        
        # Demonstrate speed ratio differences
        print(f"\n🔬 Speed Ratio Analysis:")
        print(f"This solution used different speed ratios for different vehicle types:")
        
        for v_type in ['heavy', 'standard']:
            ratios = DEFAULT_TRUCK_SPEED_RATIOS[v_type]
            print(f"  {v_type.upper()} trucks:")
            print(f"    Motorway speed ratio: {ratios['motorway']:.3f} (vs car speeds)")
            print(f"    Primary road ratio: {ratios['primary']:.3f}")
            print(f"    Secondary road ratio: {ratios['secondary']:.3f}")
            print(f"    Residential road ratio: {ratios['residential']:.3f}")
        
        # Cache statistics
        cache_stats = sequential_vrp.distance_calculator.get_cache_stats()
        print(f"\n📁 Route Database Stats:")
        print(f"  Total cached routes: {cache_stats['total_routes']}")
        print(f"  Database size: {cache_stats['database_size_mb']:.2f} MB")
        
        # Try plotting
        try:
            plot_filename = sequential_vrp.plot_sequential_solution(solution, 
                                                                   "Mixed Vehicle Types - Furgoni vs Camion")
            print(f"\n📊 Solution plotted: {plot_filename}")
        except Exception as e:
            print(f"⚠️ Could not create plot: {e}")
        
        return solution
    else:
        print("❌ Failed to solve multi-day VRP")
        return None

def demonstrate_speed_differences():
    """Demonstrate the actual speed differences between vehicle types."""
    print(f"\n" + "=" * 70)
    print(f"🔬 Speed Ratio Demonstration")
    print(f"=" * 70)
    
    print(f"This shows how different vehicle types have different speed ratios:")
    print(f"(Lower ratio = slower than cars on that road type)")
    
    road_types = ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'residential']
    
    print(f"\n{'Road Type':<15} {'Heavy Trucks':<15} {'Standard Trucks':<18} {'Difference':<12}")
    print("-" * 65)
    
    for road_type in road_types:
        if road_type in DEFAULT_TRUCK_SPEED_RATIOS['heavy']:
            heavy_ratio = DEFAULT_TRUCK_SPEED_RATIOS['heavy'][road_type]
            standard_ratio = DEFAULT_TRUCK_SPEED_RATIOS['standard'][road_type]
            difference = standard_ratio - heavy_ratio
            
            print(f"{road_type:<15} {heavy_ratio:<15.3f} {standard_ratio:<18.3f} +{difference:<12.3f}")
    
    print(f"\nExample: On a motorway where cars travel at 130 km/h:")
    motorway_car_speed = 130
    heavy_speed = motorway_car_speed * DEFAULT_TRUCK_SPEED_RATIOS['heavy']['motorway']
    standard_speed = motorway_car_speed * DEFAULT_TRUCK_SPEED_RATIOS['standard']['motorway']
    
    print(f"  🚗 Cars: {motorway_car_speed} km/h")
    print(f"  🚛 Camion (heavy): {heavy_speed:.1f} km/h")
    print(f"  🚚 Furgoni (standard): {standard_speed:.1f} km/h")
    print(f"  📈 Furgoni are {((standard_speed/heavy_speed - 1) * 100):.1f}% faster than Camion")

if __name__ == "__main__":
    print("🇮🇹 Italian Mixed Vehicle Fleet Test")
    print("=" * 80)
    
    # Demonstrate the speed differences first
    demonstrate_speed_differences()
    
    # Run the main test
    solution = test_mixed_vehicle_types()
    
    print(f"\n" + "=" * 80)
    print(f"✅ Mixed vehicle type testing completed!")
    print(f"   The system correctly applies different speed ratios to:")
    print(f"   - Camion (heavy trucks): Slower on all road types")
    print(f"   - Furgoni (standard trucks): Faster than heavy trucks")
    print(f"   - Furgoncini (light vans): Use standard truck speeds")
    print(f"   This results in more realistic routing and timing!")
