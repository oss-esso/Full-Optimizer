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
            
    elif original_solution:
        print("⚠️  Only original implementation completed")
    elif new_solution:
        print("⚠️  Only new implementation completed")
    else:
        print("❌ Both implementations failed")
    
    return original_solution, new_solution


if __name__ == "__main__":
    test_both_implementations()
