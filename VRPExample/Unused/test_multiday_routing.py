#!/usr/bin/env python3
"""
Test script for multi-day/overnight routing functionality.

This script tests the new overnight routing capability where vehicles can
sleep along their route when destinations are too far to reach within 
their daily driving time limit.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import importlib.util
spec = importlib.util.spec_from_file_location("vrp_optimizer", 
                                              os.path.join(os.path.dirname(__file__), "vrp_optimizer_clean copy.py"))
vrp_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vrp_module)
CleanVRPOptimizer = vrp_module.CleanVRPOptimizer

def create_multiday_test_scenario():
    """Create a test scenario that requires overnight stays."""
    
    # Create locations spread out geographically 
    # Some locations are far from depot (require overnight stays)
    locations = [
        # Depot
        {'id': 'depot', 'x': 11.0, 'y': 46.0, 'demand': 0, 'service_time': 0, 'address': 'Main Depot'},
        
        # Nearby locations (reachable same day)
        {'id': 'location_1', 'x': 11.1, 'y': 46.1, 'demand': 100, 'volume_demand': 2.0, 'service_time': 30, 'address': 'Location 1'},
        {'id': 'location_2', 'x': 11.2, 'y': 46.2, 'demand': 150, 'volume_demand': 3.0, 'service_time': 30, 'address': 'Location 2'},
        
        # Far locations (require overnight stays)
        {'id': 'location_far_1', 'x': 13.0, 'y': 47.5, 'demand': 200, 'volume_demand': 4.0, 'service_time': 30, 'address': 'Far Location 1'},
        {'id': 'location_far_2', 'x': 13.5, 'y': 48.0, 'demand': 250, 'volume_demand': 5.0, 'service_time': 30, 'address': 'Far Location 2'},
        {'id': 'location_far_3', 'x': 9.0, 'y': 44.0, 'demand': 300, 'volume_demand': 6.0, 'service_time': 30, 'address': 'Far Location 3'},
    ]
    
    # Create vehicles with limited daily driving time
    vehicles = [
        {
            'id': 'truck_24t',
            'capacity': 1000,  # kg
            'volume_capacity': 20.0,  # m³
            'cost_per_km': 1.80,
            'start_location': 'depot',
            'end_location': 'depot',
            'max_time': 480  # 8 hours per day (very restrictive for testing)
        },
        {
            'id': 'van_4t',
            'capacity': 500,  # kg
            'volume_capacity': 10.0,  # m³
            'cost_per_km': 0.95,
            'start_location': 'depot',
            'end_location': 'depot',
            'max_time': 480  # 8 hours per day
        }
    ]
    
    return locations, vehicles

def test_single_vs_multiday():
    """Compare single-day vs multi-day routing for the same scenario."""
    
    print("🧪 Testing Multi-Day vs Single-Day Routing")
    print("=" * 60)
    
    locations, vehicles = create_multiday_test_scenario()
    
    print(f"📍 Test scenario:")
    print(f"  - {len(locations)} locations (including depot)")
    print(f"  - {len(vehicles)} vehicles with 8-hour daily limits")
    print(f"  - Far locations may require overnight stays")
    
    # Test 1: Single-day routing (should struggle with far locations)
    print(f"\n--- Test 1: Single-Day Routing ---")
    optimizer_1day = CleanVRPOptimizer(vehicles=vehicles, locations=locations)
    
    try:
        result_1day = optimizer_1day.solve(
            constraint_level="capacity",
            verbose=False,
            max_days=1,
            daily_time_limit_minutes=480
        )
        
        if result_1day and result_1day.get('status') == 'SUCCESS':
            print(f"✅ Single-day solution found:")
            print(f"   Total distance: {result_1day.get('total_distance', 0):.1f} km")
            print(f"   Total cost: €{result_1day.get('total_cost', 0):.2f}")
            print(f"   Routes: {len(result_1day.get('routes', []))}")
            
            # Count served locations
            served_locations = set()
            for route in result_1day.get('routes', []):
                for stop in route.get('stops', []):
                    if stop['location_id'] != 'depot':
                        served_locations.add(stop['location_id'])
            
            print(f"   Served locations: {len(served_locations)}/{len(locations)-1}")
            print(f"   Served: {sorted(served_locations)}")
            
        else:
            print(f"❌ Single-day solution failed or not found")
            
    except Exception as e:
        print(f"❌ Single-day routing failed: {e}")
        result_1day = None
    
    # Test 2: Multi-day routing (should handle far locations better)
    print(f"\n--- Test 2: Multi-Day Routing (3 days) ---")
    optimizer_3day = CleanVRPOptimizer(vehicles=vehicles, locations=locations)
    
    try:
        result_3day = optimizer_3day.solve(
            constraint_level="capacity",
            verbose=False,
            max_days=3,
            daily_time_limit_minutes=480
        )
        
        if result_3day and result_3day.get('status') == 'SUCCESS':
            print(f"✅ Multi-day solution found:")
            print(f"   Total distance: {result_3day.get('total_distance', 0):.1f} km")
            print(f"   Total cost: €{result_3day.get('total_cost', 0):.2f}")
            print(f"   Routes: {len(result_3day.get('routes', []))}")
            print(f"   Max days: {result_3day.get('max_days', 1)}")
            
            # Count served locations and overnight stays
            served_locations = set()
            overnight_stays = 0
            
            # Add detailed route reporting
            print(f"\n📋 Detailed Multi-Day Route Report:")
            print("=" * 80)
            
            for route in result_3day.get('routes', []):
                vehicle_id = route.get('vehicle_id')
                total_distance = route.get('total_distance', 0)
                total_cost = route.get('total_cost', 0)
                
                print(f"\n🚛 Route for vehicle {vehicle_id}:")
                
                stops = route.get('stops', [])
                if not stops:
                    print("   No stops in route")
                    continue
                
                # Build route string with detailed information
                route_parts = []
                current_distance = 0
                current_time = 0
                current_load_weight = 0
                current_load_volume = 0
                vehicle_capacity_weight = route.get('vehicle_capacity_kg', 1000)
                vehicle_capacity_volume = route.get('vehicle_capacity_m3', 20.0)
                
                for i, stop in enumerate(stops):
                    location_id = stop['location_id']
                    stop_type = stop.get('type', 'regular')
                    day = stop.get('day', 1)
                    
                    # Handle different stop types
                    if stop_type == 'overnight_stay':
                        route_parts.append(f"🌙Night_Day{day} {current_distance}m Day{day} Time({current_time},{current_time}) Load({current_load_weight}/{vehicle_capacity_weight}kg, {current_load_volume:.1f}/{vehicle_capacity_volume}m³)")
                        # Reset time for overnight (new day starts)
                        current_time = 0
                    elif stop_type == 'morning_start':
                        route_parts.append(f"🌅Morning_Day{day} {current_distance}m Day{day} Time({current_time},{current_time}) Load({current_load_weight}/{vehicle_capacity_weight}kg, {current_load_volume:.1f}/{vehicle_capacity_volume}m³)")
                    else:
                        # Regular stop
                        weight_usage = stop.get('weight_usage_kg', 0)
                        volume_usage = stop.get('volume_usage_m3', 0.0)
                        
                        # Update cumulative load (simplified - showing current load)
                        current_load_weight = weight_usage
                        current_load_volume = volume_usage
                        
                        # Time window info (simplified)
                        tw_start = 0  # Would need to get from location data
                        tw_end = 1440  # 24 hours in minutes
                        
                        route_parts.append(f"{location_id} {current_distance}m Day{day} TW:[{tw_start},{tw_end}] Time({current_time},{current_time}) Load({current_load_weight}/{vehicle_capacity_weight}kg, {current_load_volume:.1f}/{vehicle_capacity_volume}m³)")
                        
                        # Estimate time increment (simplified)
                        current_time += 30  # Service time
                    
                    # Estimate distance increment for next segment (simplified)
                    if i < len(stops) - 1:
                        current_distance += 50000  # Rough estimate in meters
                
                # Print the route
                route_string = " -> ".join(route_parts)
                print(f"   {route_string}")
                print(f"   Distance of the route: {total_distance*1000:.0f}m")
                print(f"   Cost of the route: €{total_cost:.2f}")
                print(f"   Vehicle capacity: {vehicle_capacity_weight}kg, {vehicle_capacity_volume}m³")
            
            # Original summary reporting
            for route in result_3day.get('routes', []):
                print(f"   Vehicle {route.get('vehicle_id')} route:")
                for stop in route.get('stops', []):
                    if stop.get('type') == 'overnight_stay':
                        overnight_stays += 1
                        print(f"     Day {stop.get('day')}: {stop.get('message')}")
                    elif stop.get('type') == 'morning_start':
                        print(f"     Day {stop.get('day')}: {stop.get('message')}")
                    elif stop['location_id'] != 'depot':
                        served_locations.add(stop['location_id'])
                        day = stop.get('day', 1)
                        print(f"     Day {day}: {stop['location_id']}")
            
            print(f"   Served locations: {len(served_locations)}/{len(locations)-1}")
            print(f"   Overnight stays: {overnight_stays}")
            print(f"   Served: {sorted(served_locations)}")
            
            # Check overnight stays info
            overnight_info = result_3day.get('multi_day_info', {}).get('overnight_stays', [])
            if overnight_info:
                print(f"   Overnight stays details:")
                for stay in overnight_info:
                    print(f"     {stay['vehicle_id']} slept on day {stay['day']}")
        else:
            print(f"❌ Multi-day solution failed or not found")
            result_3day = None
            
    except Exception as e:
        print(f"❌ Multi-day routing failed: {e}")
        result_3day = None
    
    # Compare results
    print(f"\n--- Comparison ---")
    if result_1day and result_3day:
        # Count served locations for both
        served_1day = set()
        for route in result_1day.get('routes', []):
            for stop in route.get('stops', []):
                if stop['location_id'] != 'depot':
                    served_1day.add(stop['location_id'])
        
        served_3day = set()
        for route in result_3day.get('routes', []):
            for stop in route.get('stops', []):
                if stop['location_id'] != 'depot' and not stop.get('type') in ['overnight_stay', 'morning_start']:
                    served_3day.add(stop['location_id'])
        
        print(f"Single-day served: {len(served_1day)} locations")
        print(f"Multi-day served: {len(served_3day)} locations")
        
        if len(served_3day) > len(served_1day):
            print(f"✅ Multi-day routing served {len(served_3day) - len(served_1day)} more locations!")
        elif len(served_3day) == len(served_1day):
            print(f"⚖️  Both approaches served the same number of locations")
        else:
            print(f"🤔 Single-day served more locations than multi-day")
            
        # Cost comparison
        cost_1day = result_1day.get('total_cost', 0)
        cost_3day = result_3day.get('total_cost', 0)
        print(f"Single-day cost: €{cost_1day:.2f}")
        print(f"Multi-day cost: €{cost_3day:.2f}")
        
    else:
        print(f"Cannot compare - one or both solutions failed")
    
    return result_1day, result_3day

def test_with_furgoni_scenario():
    """Test multi-day routing with the existing furgoni scenario."""
    print(f"\n🚛 Testing Multi-Day Routing with Furgoni Scenario")
    print("=" * 60)
    
    try:
        from vrp_scenarios import create_furgoni_scenario
        
        # Create the scenario
        scenario = create_furgoni_scenario()
        print(f"📊 Furgoni scenario details:")
        print(f"  - Locations: {len(scenario.locations)}")
        print(f"  - Vehicles: {len(scenario.vehicles)}")
        
        # Test with multi-day routing
        optimizer = CleanVRPOptimizer(vrp_instance=scenario)
        
        result = optimizer.solve(
            constraint_level="capacity",
            verbose=False,
            max_days=4,  # Allow up to 4 days
            daily_time_limit_minutes=480  # 8 hours per day
        )
        
        if result and result.get('status') == 'SUCCESS':
            print(f"✅ Furgoni multi-day solution found:")
            print(f"   Total distance: {result.get('total_distance', 0):.1f} km")
            print(f"   Total cost: €{result.get('total_cost', 0):.2f}")
            print(f"   Routes: {len(result.get('routes', []))}")
            
            # Analyze routes for overnight stays
            overnight_stays = 0
            for route in result.get('routes', []):
                print(f"   Vehicle {route.get('vehicle_id')}:")
                current_day = 1
                for stop in route.get('stops', []):
                    if stop.get('type') == 'overnight_stay':
                        overnight_stays += 1
                        print(f"     🌙 Day {stop.get('day')}: Overnight stay")
                    elif stop.get('type') == 'morning_start':
                        print(f"     🌅 Day {stop.get('day')}: Morning start")
                    elif stop['location_id'] != 'depot':
                        day = stop.get('day', current_day)
                        print(f"     📍 Day {day}: {stop['location_id']}")
            
            print(f"   Total overnight stays: {overnight_stays}")
            
            return result
        else:
            print(f"❌ Furgoni multi-day solution failed")
            return None
            
    except ImportError:
        print(f"❌ Cannot import furgoni scenario - skipping this test")
        return None
    except Exception as e:
        print(f"❌ Furgoni multi-day test failed: {e}")
        return None

def print_detailed_route_report(result):
    """Print detailed route report in the requested format with day information."""
    if not result or result.get('status') != 'SUCCESS':
        return
    
    print(f"\n📋 DETAILED ROUTE REPORT")
    print("=" * 80)
    
    routes = result.get('routes', [])
    
    for route_idx, route in enumerate(routes):
        vehicle_id = route.get('vehicle_id', f'Vehicle_{route_idx}')
        stops = route.get('stops', [])
        
        print(f"\nRoute for vehicle {vehicle_id}:")
        
        # Calculate actual times and distances
        cumulative_distance_m = 0
        cumulative_time_min = 0
        current_day = 1
        day_start_time = 0  # Time when current day started
        
        # Group stops by day for display
        days_routes = {}
        
        for stop_idx, stop in enumerate(stops):
            location_id = stop.get('location_id', f'Node_{stop_idx}')
            stop_type = stop.get('type', 'regular')
            stop_day = stop.get('day', current_day)
            
            # Handle special stop types
            if stop_type == 'overnight_stay':
                if stop_day not in days_routes:
                    days_routes[stop_day] = []
                
                # Overnight stay ends the current day
                days_routes[stop_day].append({
                    'type': 'overnight',
                    'name': f"🌙Night_Day{stop_day}",
                    'distance_m': cumulative_distance_m,
                    'travel_distance_m': 0,  # No travel for overnight stay
                    'time_window': "TW:[0,86400]",
                    'arrival_time': cumulative_time_min - day_start_time,
                    'departure_time': cumulative_time_min - day_start_time,
                    'load': stop.get('weight_usage_kg', 0),
                    'capacity': route.get('vehicle_capacity_kg', 1000)
                })
                
                # Reset for new day
                current_day = stop_day + 1
                day_start_time = cumulative_time_min
                continue
                
            elif stop_type == 'morning_start':
                if stop_day not in days_routes:
                    days_routes[stop_day] = []
                
                days_routes[stop_day].append({
                    'type': 'morning',
                    'name': f"🌅Morning_Day{stop_day}",
                    'distance_m': cumulative_distance_m,
                    'travel_distance_m': 0,  # No travel for morning start
                    'time_window': "TW:[0,86400]",
                    'arrival_time': cumulative_time_min - day_start_time,
                    'departure_time': cumulative_time_min - day_start_time,
                    'load': stop.get('weight_usage_kg', 0),
                    'capacity': route.get('vehicle_capacity_kg', 1000)
                })
                current_day = stop_day
                continue
            
            # Regular stops - estimate travel time and distance
            travel_distance = 0
            if stop_idx > 0:
                # Estimate travel time based on location (simplified)
                # In real implementation, this would use the distance matrix
                if 'far' in location_id.lower():
                    travel_time = 120  # 2 hours to far locations
                    travel_distance = 150000  # 150km in meters
                else:
                    travel_time = 30   # 30 minutes to nearby locations
                    travel_distance = 20000   # 20km in meters
                
                cumulative_time_min += travel_time
                cumulative_distance_m += travel_distance
            
            # Service time at location
            service_time = 30 if location_id != 'depot' else 0
            
            # Add to the appropriate day
            if stop_day not in days_routes:
                days_routes[stop_day] = []
            
            days_routes[stop_day].append({
                'type': 'regular',
                'name': location_id,
                'distance_m': cumulative_distance_m,
                'travel_distance_m': travel_distance,  # Distance from previous stop
                'time_window': "TW:[0,86400]",  # Match original format (seconds)
                'arrival_time': cumulative_time_min - day_start_time,
                'departure_time': cumulative_time_min - day_start_time + service_time,
                'load': stop.get('weight_usage_kg', 0),
                'capacity': route.get('vehicle_capacity_kg', 1000),
                'volume_load': stop.get('volume_usage_m3', 0),
                'volume_capacity': route.get('vehicle_capacity_m3', 20)
            })
            
            # Add service time
            cumulative_time_min += service_time
            current_day = stop_day
        
        # Print each day's route
        for day in sorted(days_routes.keys()):
            day_stops = days_routes[day]
            
            # Skip days that only have morning starts with no regular stops
            has_regular_stops = any(stop['type'] == 'regular' for stop in day_stops)
            has_overnight_stops = any(stop['type'] == 'overnight' for stop in day_stops)
            
            if not has_regular_stops and not has_overnight_stops:
                # Skip days with only morning starts and no activity
                continue
            
            # Also skip days where the only regular stop is the depot at the end
            regular_stops = [stop for stop in day_stops if stop['type'] == 'regular']
            if len(regular_stops) == 1 and regular_stops[0]['name'] == 'depot':
                continue
                
            print(f"  Day {day}:")
            
            route_parts = []
            
            for stop in day_stops:
                if stop['type'] in ['overnight', 'morning']:
                    route_parts.append(stop['name'])
                else:
                    # Format: location_id distance TW:[start,end] Time(arrival,departure) Load(current/capacity)
                    arr_min = int(stop['arrival_time'])
                    dep_min = int(stop['departure_time'])
                    dist_km = stop['distance_m'] / 1000
                    # Show both weight and volume if available
                    if 'volume_load' in stop and 'volume_capacity' in stop:
                        load_str = f"Load({stop['load']}/{stop['capacity']},{stop['volume_load']:.1f}/{stop['volume_capacity']:.1f})"
                    else:
                        load_str = f"Load({stop['load']}/{stop['capacity']})"
                    stop_str = f"{stop['name']} {dist_km:.1f}km {stop['time_window']} Time({arr_min},{dep_min}) {load_str}"
                    route_parts.append(stop_str)
            
            # Print the day's route
            route_line = " -> ".join(route_parts)
            print(f"    {route_line}")
            
            # Day summary - calculate actual distance and time for this day only
            if day_stops:
                day_travel_distance = 0
                max_time = 0
                
                for stop in day_stops:
                    if stop['type'] == 'regular':
                        # Add travel distance for this stop
                        if 'travel_distance_m' in stop:
                            day_travel_distance += stop['travel_distance_m']
                        # Track maximum departure time for the day
                        max_time = max(max_time, stop.get('departure_time', 0))
                
                # Only print distance summary if there was actual travel
                if day_travel_distance > 0:
                    print(f"    Distance of day {day}: {day_travel_distance/1000:.1f}km")
                    print(f"    Time of day {day}: {int(max_time)}min")
        
        # Print total route summary
        total_distance = route.get('total_distance', 0)
        total_time = cumulative_time_min
        total_cost = route.get('total_cost', 0)
        
        print(f"  Total route distance: {total_distance:.1f}km")
        print(f"  Total route time: {total_time}min")
        print(f"  Total route cost: €{total_cost:.2f}")
        
        # Check for overnight stays
        overnight_count = sum(1 for day_stops in days_routes.values() 
                            for stop in day_stops 
                            if stop['type'] == 'overnight')
        if overnight_count > 0:
            print(f"  Overnight stays: {overnight_count}")

def debug_route_data(result):
    """Debug function to inspect what data is available in route stops."""
    if not result or result.get('status') != 'SUCCESS':
        return
        
    print(f"\n🔍 DEBUG: Inspecting route data structure")
    print("=" * 60)
    
    routes = result.get('routes', [])
    
    for route_idx, route in enumerate(routes):
        vehicle_id = route.get('vehicle_id', f'Vehicle_{route_idx}')
        stops = route.get('stops', [])
        
        print(f"\nVehicle {vehicle_id} - Route data:")
        print(f"  Route keys: {list(route.keys())}")
        
        print(f"  Stops ({len(stops)}):")
        for i, stop in enumerate(stops[:3]):  # Show first 3 stops
            print(f"    Stop {i}: {stop.get('location_id', 'unknown')}")
            print(f"      Keys: {list(stop.keys())}")
            print(f"      Type: {stop.get('type', 'regular')}")
            print(f"      Day: {stop.get('day', 'N/A')}")
            
            # Check for time-related fields
            time_fields = [k for k in stop.keys() if 'time' in k.lower()]
            distance_fields = [k for k in stop.keys() if 'distance' in k.lower()]
            load_fields = [k for k in stop.keys() if any(word in k.lower() for word in ['load', 'capacity', 'weight', 'volume'])]
            
            if time_fields:
                print(f"      Time fields: {time_fields}")
                for tf in time_fields:
                    print(f"        {tf}: {stop.get(tf)}")
            else:
                print(f"      Time fields: None")
                
            if distance_fields:
                print(f"      Distance fields: {distance_fields}")
                for df in distance_fields:
                    print(f"        {df}: {stop.get(df)}")
            else:
                print(f"      Distance fields: None")
                
            if load_fields:
                print(f"      Load fields: {load_fields}")
                for lf in load_fields:
                    print(f"        {lf}: {stop.get(lf)}")
            else:
                print(f"      Load fields: None")
            print()

if __name__ == "__main__":
    print("🚚 MULTI-DAY ROUTING TEST SUITE")
    print("=" * 60)
    
    # Test 1: Compare single-day vs multi-day on custom scenario
    result_1day, result_3day = test_single_vs_multiday()
    
    # Print detailed route reports
    if result_1day:
        print("\n📋 DETAILED SINGLE-DAY ROUTE REPORT:")
        debug_route_data(result_1day)
        print_detailed_route_report(result_1day)
        
    if result_3day:
        print("\n📋 DETAILED MULTI-DAY ROUTE REPORT:")
        debug_route_data(result_3day)
        print_detailed_route_report(result_3day)
    
    # Test 2: Test with furgoni scenario
    #furgoni_result = test_with_furgoni_scenario()
    
    print(f"\n🏁 Multi-day routing tests completed!")
    print(f"💡 Key observations:")
    print(f"   - Multi-day routing allows vehicles to sleep along their route")
    print(f"   - Overnight stays enable reaching distant locations")
    print(f"   - Virtual night/morning nodes handle day transitions")
    print(f"   - Time dimension enforces daily driving limits")
