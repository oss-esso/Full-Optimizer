#!/usr/bin/env python3
"""
Enhanced Test for Sequential Multi-Day VRP with Bigger Example
============================================================

This creates a comprehensive test scenario with more locations and vehicles
to thoroughly test both VRP implementations.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_large_test_scenario():
    """Create a large test scenario with many customers spread across a region."""
    
    # Main depot
    locations = [
        {'id': 'depot', 'x': 47.3769, 'y': 8.5417, 'demand': 0, 'service_time': 0, 'address': 'Main Depot Zurich'},
    ]
    
    # Regional customers - spread across Switzerland-like area
    customers = [
        # North region
        {'id': 'basel_north', 'x': 47.5596, 'y': 7.5886, 'demand': 120, 'volume_demand': 3.2, 'service_time': 25, 'address': 'Basel North'},
        {'id': 'schaffhausen', 'x': 47.6979, 'y': 8.6308, 'demand': 95, 'volume_demand': 2.1, 'service_time': 20, 'address': 'Schaffhausen'},
        {'id': 'winterthur', 'x': 47.5009, 'y': 8.7240, 'demand': 140, 'volume_demand': 3.8, 'service_time': 30, 'address': 'Winterthur'},
        {'id': 'st_gallen', 'x': 47.4245, 'y': 9.3767, 'demand': 180, 'volume_demand': 4.5, 'service_time': 35, 'address': 'St. Gallen'},
        
        # East region  
        {'id': 'chur', 'x': 46.8480, 'y': 9.5330, 'demand': 110, 'volume_demand': 2.8, 'service_time': 25, 'address': 'Chur'},
        {'id': 'davos', 'x': 46.8098, 'y': 9.8368, 'demand': 75, 'volume_demand': 1.9, 'service_time': 20, 'address': 'Davos'},
        {'id': 'st_moritz', 'x': 46.4908, 'y': 9.8355, 'demand': 65, 'volume_demand': 1.7, 'service_time': 18, 'address': 'St. Moritz'},
        
        # South region
        {'id': 'lugano', 'x': 46.0037, 'y': 8.9511, 'demand': 200, 'volume_demand': 5.2, 'service_time': 40, 'address': 'Lugano'},
        {'id': 'locarno', 'x': 46.1712, 'y': 8.7994, 'demand': 85, 'volume_demand': 2.3, 'service_time': 22, 'address': 'Locarno'},
        {'id': 'bellinzona', 'x': 46.1944, 'y': 9.0175, 'demand': 130, 'volume_demand': 3.4, 'service_time': 28, 'address': 'Bellinzona'},
        
        # West region
        {'id': 'geneva', 'x': 46.2044, 'y': 6.1432, 'demand': 250, 'volume_demand': 6.1, 'service_time': 45, 'address': 'Geneva'},
        {'id': 'lausanne', 'x': 46.5197, 'y': 6.6323, 'demand': 190, 'volume_demand': 4.8, 'service_time': 38, 'address': 'Lausanne'},
        {'id': 'montreux', 'x': 46.4312, 'y': 6.9123, 'demand': 100, 'volume_demand': 2.6, 'service_time': 24, 'address': 'Montreux'},
        {'id': 'sion', 'x': 46.2280, 'y': 7.3598, 'demand': 115, 'volume_demand': 3.0, 'service_time': 26, 'address': 'Sion'},
        
        # Central region
        {'id': 'lucerne', 'x': 47.0502, 'y': 8.3093, 'demand': 160, 'volume_demand': 4.1, 'service_time': 32, 'address': 'Lucerne'},
        {'id': 'interlaken', 'x': 46.6863, 'y': 7.8632, 'demand': 90, 'volume_demand': 2.4, 'service_time': 23, 'address': 'Interlaken'},
        {'id': 'bern', 'x': 46.9481, 'y': 7.4474, 'demand': 210, 'volume_demand': 5.4, 'service_time': 42, 'address': 'Bern'},
        {'id': 'fribourg', 'x': 46.8057, 'y': 7.1608, 'demand': 125, 'volume_demand': 3.3, 'service_time': 27, 'address': 'Fribourg'},
        
        # Additional challenging locations
        {'id': 'zermatt', 'x': 46.0207, 'y': 7.7491, 'demand': 60, 'volume_demand': 1.5, 'service_time': 15, 'address': 'Zermatt'},
        {'id': 'appenzell', 'x': 47.3319, 'y': 9.4108, 'demand': 70, 'volume_demand': 1.8, 'service_time': 18, 'address': 'Appenzell'},
        {'id': 'aarau', 'x': 47.3911, 'y': 8.0431, 'demand': 135, 'volume_demand': 3.5, 'service_time': 29, 'address': 'Aarau'},
    ]
    
    locations.extend(customers)
    
    # Fleet of diverse vehicles
    vehicles = [
        {'id': 'truck_40t_alpha', 'capacity': 800, 'volume_capacity': 18.0, 'cost_per_km': 2.20, 'max_daily_km': 600},
        {'id': 'truck_24t_beta', 'capacity': 500, 'volume_capacity': 12.0, 'cost_per_km': 1.80, 'max_daily_km': 650},
        {'id': 'van_7t_gamma', 'capacity': 350, 'volume_capacity': 10.0, 'cost_per_km': 1.20, 'max_daily_km': 700},
        {'id': 'van_4t_delta', 'capacity': 250, 'volume_capacity': 8.0, 'cost_per_km': 0.95, 'max_daily_km': 750},
        {'id': 'sprinter_3t_epsilon', 'capacity': 180, 'volume_capacity': 6.0, 'cost_per_km': 0.75, 'max_daily_km': 800},
        # Added five additional vehicles for expanded testing
        {'id': 'truck_18t_zeta', 'capacity': 400, 'volume_capacity': 9.0, 'cost_per_km': 1.50, 'max_daily_km': 700},
        {'id': 'van_5t_eta', 'capacity': 300, 'volume_capacity': 7.0, 'cost_per_km': 1.00, 'max_daily_km': 800},
        {'id': 'truck_electric_theta', 'capacity': 600, 'volume_capacity': 15.0, 'cost_per_km': 0.50, 'max_daily_km': 500},
        {'id': 'van_electric_iota', 'capacity': 200, 'volume_capacity': 5.0, 'cost_per_km': 0.40, 'max_daily_km': 900},
        {'id': 'bike_delivery_mu', 'capacity': 50,  'volume_capacity': 1.0, 'cost_per_km': 0.10, 'max_daily_km': 100},
    ]
    
    print(f"🏗️  Created large test scenario:")
    print(f"   📍 Locations: {len(locations)} (1 depot + {len(customers)} customers)")
    print(f"   🚛 Vehicles: {len(vehicles)}")
    print(f"   📦 Total demand: {sum(loc.get('demand', 0) for loc in locations)} units")
    print(f"   📊 Total volume: {sum(loc.get('volume_demand', 0) for loc in locations):.1f} m³")
    print(f"   🕒 Total service time: {sum(loc.get('service_time', 0) for loc in locations)} minutes")
    
    return locations, vehicles


def test_both_implementations():
    """Test both VRP implementations with the large scenario."""
    print("\n" + "="*80)
    print("🧪 COMPREHENSIVE TEST: BOTH VRP IMPLEMENTATIONS")
    print("="*80)
    
    # Create large test scenario
    locations, vehicles = create_large_test_scenario()
    
    # Import both implementations
    try:
        # Test original implementation
        print(f"\n{'='*60}")
        print("🚀 TESTING ORIGINAL IMPLEMENTATION")
        print(f"{'='*60}")
        
        import vrp_multiday_sequential
        
        original_vrp = vrp_multiday_sequential.SequentialMultiDayVRP(vehicles, locations)
        original_solution = original_vrp.solve_sequential_multiday(max_days=7)
        
        if original_solution:
            original_plot = original_vrp.plot_sequential_solution(
                original_solution, 
                "Original Sequential Multi-Day VRP - Large Scenario"
            )
            print(f"✅ Original implementation completed successfully")
            
            # Analyze customer coverage for original implementation
            print_customer_analysis(original_solution, locations, "Original Implementation")
        else:
            print(f"❌ Original implementation failed")
            
    except Exception as e:
        print(f"❌ Error testing original implementation: {e}")
        original_solution = None
        original_plot = None
    
    try:
        # Test new implementation
        print(f"\n{'='*60}")
        print("🚀 TESTING NEW IMPLEMENTATION")
        print(f"{'='*60}")
        
        import vrp_multiday_sequential_new
        
        new_vrp = vrp_multiday_sequential_new.SequentialMultiDayVRP(vehicles, locations)
        new_solution = new_vrp.solve_sequential_multiday(max_days=7)
        
        if new_solution:
            new_plot = new_vrp.plot_sequential_solution(
                new_solution, 
                "New Sequential Multi-Day VRP - Large Scenario"
            )
            print(f"✅ New implementation completed successfully")
            
            # Analyze customer coverage for new implementation
            print_customer_analysis(new_solution, locations, "New Implementation")
        else:
            print(f"❌ New implementation failed")
            
    except Exception as e:
        print(f"❌ Error testing new implementation: {e}")
        new_solution = None
        new_plot = None
    
    # Comparison summary
    print(f"\n{'='*60}")
    print("📊 COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    if original_solution and new_solution:
        print("✅ Both implementations completed successfully")
        
        # Compare key metrics
        orig_days = original_solution.get('total_days', 0)
        new_days = new_solution.get('total_days', 0)
        
        orig_distance = sum(route.get('total_distance', 0) for route in original_solution.get('vehicle_routes', {}).values())
        new_distance = sum(route.get('total_distance', 0) for route in new_solution.get('vehicle_routes', {}).values())
        
        orig_overnight = sum(route.get('total_overnight_stays', 0) for route in original_solution.get('vehicle_routes', {}).values())
        new_overnight = sum(route.get('total_overnight_stays', 0) for route in new_solution.get('vehicle_routes', {}).values())
        
        print(f"\n📈 Metrics Comparison:")
        print(f"   Days required:    Original: {orig_days:>3} | New: {new_days:>3}")
        print(f"   Total distance:   Original: {orig_distance:>7.1f} km | New: {new_distance:>7.1f} km")
        print(f"   Overnight stays:  Original: {orig_overnight:>3} | New: {new_overnight:>3}")
        
        if orig_distance > 0 and new_distance > 0:
            improvement = ((orig_distance - new_distance) / orig_distance) * 100
            print(f"   Distance improvement: {improvement:+.1f}%")
        
        # Compare customer coverage between implementations
        compare_customer_coverage(original_solution, new_solution, locations)
            
    elif original_solution:
        print("⚠️  Only original implementation completed")
    elif new_solution:
        print("⚠️  Only new implementation completed")
    else:
        print("❌ Both implementations failed")
    
    return original_solution, new_solution


def analyze_customer_coverage(solution, locations):
    """Analyze which customers are reached and which are not."""
    if not solution or solution.get('status') != 'SUCCESS':
        return None, None, None
    
    # Get all real customer locations (excluding depot and any overnight pseudo-customers)
    # Filter out depot and any location that looks like an overnight stop
    all_customers = {}
    for loc in locations:
        loc_id = loc['id']
        # Only include real customers - exclude depot and overnight pseudo-locations
        if (loc_id != 'depot' and 
            not loc_id.startswith('road_overnight') and
            not loc_id.startswith('overnight_') and
            not loc.get('is_overnight_stop', False) and
            not loc.get('type', '').lower() in ['overnight', 'depot']):
            all_customers[loc_id] = loc
    
    served_customers = set()
    unserved_customers = set()
    
    # Track customers served by each vehicle
    vehicle_customers = {}
    
    # Extract served customers from vehicle routes
    for vehicle_id, vehicle_data in solution.get('vehicle_routes', {}).items():
        vehicle_customers[vehicle_id] = []
        full_route = vehicle_data.get('full_route', [])
        
        # Keep track of unique customers served by this vehicle
        vehicle_served_customers = set()
        
        for stop in full_route:
            location_id = stop.get('location_id', '')
            # Only count real customers (already filtered in all_customers)
            if (location_id in all_customers and 
                not stop.get('is_day_marker', False)):
                served_customers.add(location_id)
                vehicle_served_customers.add(location_id)
        
        # Convert to list for the vehicle_customers dict
        vehicle_customers[vehicle_id] = list(vehicle_served_customers)
    
    # Find unserved customers
    unserved_customers = set(all_customers.keys()) - served_customers
    
    return served_customers, unserved_customers, vehicle_customers


def print_customer_analysis(solution, locations, implementation_name):
    """Print detailed customer coverage analysis."""
    served, unserved, vehicle_customers = analyze_customer_coverage(solution, locations)
    
    if served is None:
        print(f"❌ {implementation_name}: No valid solution to analyze")
        return
    
    # Count real customers only (same filtering as in analyze_customer_coverage)
    real_customers = [loc for loc in locations 
                     if (loc['id'] != 'depot' and 
                         not loc['id'].startswith('road_overnight') and
                         not loc['id'].startswith('overnight_') and
                         not loc.get('is_overnight_stop', False) and
                         not loc.get('type', '').lower() in ['overnight', 'depot'])]
    
    total_customers = len(real_customers)
    served_count = len(served)
    unserved_count = len(unserved)
    coverage_percentage = (served_count / total_customers) * 100 if total_customers > 0 else 0
    
    print(f"\n📋 {implementation_name.upper()} - CUSTOMER COVERAGE ANALYSIS")
    print("=" * 60)
    print(f"📊 Coverage Summary:")
    print(f"   Total Customers: {total_customers}")
    print(f"   Served: {served_count} ({coverage_percentage:.1f}%)")
    print(f"   Unserved: {unserved_count} ({100-coverage_percentage:.1f}%)")
    
    if served:
        print(f"\n✅ SERVED CUSTOMERS ({len(served)}):")
        served_list = sorted(list(served))
        for i, customer in enumerate(served_list, 1):
            customer_info = next((loc for loc in locations if loc['id'] == customer), {})
            demand = customer_info.get('demand', 0)
            volume = customer_info.get('volume_demand', 0)
            service_time = customer_info.get('service_time', 0)
            address = customer_info.get('address', 'Unknown')
            
            # Find which vehicle serves this customer
            serving_vehicle = None
            for vehicle_id, customers in vehicle_customers.items():
                if customer in customers:
                    serving_vehicle = vehicle_id
                    break
            
            print(f"   {i:2d}. {customer:<15} | {address:<25} | Vehicle: {serving_vehicle:<20} | "
                  f"Demand: {demand:3d} | Volume: {volume:.1f}m³ | Service: {service_time:2d}min")
    
    if unserved:
        print(f"\n❌ UNSERVED CUSTOMERS ({len(unserved)}):")
        unserved_list = sorted(list(unserved))
        total_missed_demand = 0
        total_missed_volume = 0
        total_missed_service_time = 0
        
        for i, customer in enumerate(unserved_list, 1):
            customer_info = next((loc for loc in locations if loc['id'] == customer), {})
            demand = customer_info.get('demand', 0)
            volume = customer_info.get('volume_demand', 0)
            service_time = customer_info.get('service_time', 0)
            address = customer_info.get('address', 'Unknown')
            
            total_missed_demand += demand
            total_missed_volume += volume
            total_missed_service_time += service_time
            
            print(f"   {i:2d}. {customer:<15} | {address:<25} | "
                  f"Demand: {demand:3d} | Volume: {volume:.1f}m³ | Service: {service_time:2d}min")
        
        print(f"\n💔 MISSED TOTALS:")
        print(f"   Lost Demand: {total_missed_demand} units")
        print(f"   Lost Volume: {total_missed_volume:.1f} m³")
        print(f"   Lost Service Time: {total_missed_service_time} minutes")
    
    # Vehicle workload distribution
    print(f"\n🚛 VEHICLE WORKLOAD DISTRIBUTION:")
    total_served_customers = 0
    total_served_demand = 0
    total_served_volume = 0
    
    for vehicle_id, customers in vehicle_customers.items():
        customer_count = len(customers)
        if customer_count > 0:
            total_demand = sum(next((loc.get('demand', 0) for loc in locations if loc['id'] == customer), 0) 
                             for customer in customers)
            total_volume = sum(next((loc.get('volume_demand', 0) for loc in locations if loc['id'] == customer), 0) 
                             for customer in customers)
            
            total_served_customers += customer_count
            total_served_demand += total_demand
            total_served_volume += total_volume
            
            # Show customer list for this vehicle
            customer_list = ", ".join(sorted(customers))
            print(f"   {vehicle_id:<20} | Customers: {customer_count:2d} | "
                  f"Total Demand: {total_demand:3d} | Total Volume: {total_volume:.1f}m³")
            print(f"   {'':>20}   └─ Served: {customer_list}")
        else:
            print(f"   {vehicle_id:<20} | Customers:  0 | "
                  f"Total Demand:   0 | Total Volume:  0.0m³")
    
    print(f"\n📊 FLEET TOTALS:")
    print(f"   Total customers served: {total_served_customers}")
    print(f"   Total demand served: {total_served_demand} units")
    print(f"   Total volume served: {total_served_volume:.1f} m³")

    # Identify and list missing customers based on vehicle workload
    # Exclude depot and overnight pseudo-customers (same filtering as real_customers above)
    real_customers = [loc for loc in locations 
                     if (loc['id'] != 'depot' and 
                         not loc['id'].startswith('road_overnight') and
                         not loc['id'].startswith('overnight_') and
                         not loc.get('is_overnight_stop', False) and
                         not loc.get('type', '').lower() in ['overnight', 'depot'])]
    all_ids = {loc['id'] for loc in real_customers}
    served_ids = set().union(*vehicle_customers.values()) if vehicle_customers else set()
    missing = sorted(all_ids - served_ids)

    if missing:
        print(f"\n🚧 MISSING CUSTOMERS ({len(missing)}):")
        for cust in missing:
            info = next((loc for loc in locations if loc['id'] == cust), {})
            print(f"   • {cust:<15} | {info.get('address','Unknown'):<25} | Demand: {info.get('demand',0):3d}")


def compare_customer_coverage(original_solution, new_solution, locations):
    """Compare customer coverage between both implementations."""
    print(f"\n🔍 DETAILED COVERAGE COMPARISON")
    print("=" * 60)
    
    # Analyze both solutions
    orig_served, orig_unserved, orig_vehicles = analyze_customer_coverage(original_solution, locations)
    new_served, new_unserved, new_vehicles = analyze_customer_coverage(new_solution, locations)
    
    if orig_served is None or new_served is None:
        print("❌ Cannot compare - one or both solutions invalid")
        return
    
    # Find differences
    only_original = orig_served - new_served if orig_served and new_served else set()
    only_new = new_served - orig_served if orig_served and new_served else set()
    both_served = orig_served & new_served if orig_served and new_served else set()
    neither_served = orig_unserved & new_unserved if orig_unserved and new_unserved else set()
    
    print(f"📈 Coverage Comparison Summary:")
    print(f"   Served by BOTH: {len(both_served)}")
    print(f"   Served by ORIGINAL only: {len(only_original)}")
    print(f"   Served by NEW only: {len(only_new)}")
    print(f"   Served by NEITHER: {len(neither_served)}")
    
    if only_original:
        print(f"\n🔵 CUSTOMERS SERVED ONLY BY ORIGINAL ({len(only_original)}):")
        for customer in sorted(only_original):
            customer_info = next((loc for loc in locations if loc['id'] == customer), {})
            print(f"   • {customer:<15} | {customer_info.get('address', 'Unknown'):<25} | "
                  f"Demand: {customer_info.get('demand', 0):3d}")
    
    if only_new:
        print(f"\n🟢 CUSTOMERS SERVED ONLY BY NEW ({len(only_new)}):")
        for customer in sorted(only_new):
            customer_info = next((loc for loc in locations if loc['id'] == customer), {})
            print(f"   • {customer:<15} | {customer_info.get('address', 'Unknown'):<25} | "
                  f"Demand: {customer_info.get('demand', 0):3d}")
    
    if neither_served:
        print(f"\n⚫ CUSTOMERS MISSED BY BOTH ({len(neither_served)}):")
        total_lost_demand = 0
        for customer in sorted(neither_served):
            customer_info = next((loc for loc in locations if loc['id'] == customer), {})
            demand = customer_info.get('demand', 0)
            total_lost_demand += demand
            print(f"   • {customer:<15} | {customer_info.get('address', 'Unknown'):<25} | "
                  f"Demand: {demand:3d}")
        print(f"   💔 Total Lost Demand: {total_lost_demand} units")


if __name__ == "__main__":
    test_both_implementations()
