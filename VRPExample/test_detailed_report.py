#!/usr/bin/env python3
"""
Quick test to show the detailed route report format.
"""

from vrp_scenarios import create_MODA_small_scenario, create_custom_multiday_scenario
# Import from the copy file since that's the one with all latest features
exec(open('vrp_optimizer_clean copy.py').read())
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
        
        # Group stops by day
        days_data = {}
        current_day = 1
        cumulative_distance = 0
        cumulative_time = 0
        current_load = 0  # Track current load
        
        for stop_idx, stop in enumerate(stops):
            stop_day = stop.get('day', current_day)
            
            # Handle special stop types
            if stop.get('type') == 'overnight_stay':
                if stop_day not in days_data:
                    days_data[stop_day] = []
                days_data[stop_day].append({
                    'type': 'overnight_stay',
                    'day': stop_day,
                    'message': stop.get('message', 'Overnight stay'),
                    'distance': cumulative_distance,
                    'time': cumulative_time,
                    'load': current_load
                })
                current_day = stop_day + 1
                continue
            elif stop.get('type') == 'morning_start':
                if stop_day not in days_data:
                    days_data[stop_day] = []
                days_data[stop_day].append({
                    'type': 'morning_start',
                    'day': stop_day,
                    'message': stop.get('message', 'Morning start'),
                    'distance': cumulative_distance,
                    'time': cumulative_time,
                    'load': current_load
                })
                current_day = stop_day
                continue
            
            # Regular stops
            location_id = stop.get('location_id', f'Node_{stop_idx}')
            
            # Update cumulative distance
            if 'distance_from_previous' in stop:
                cumulative_distance += stop['distance_from_previous']
            elif 'cumulative_distance' in stop:
                cumulative_distance = stop['cumulative_distance']
            
            # Update cumulative time
            if 'time_from_previous' in stop:
                cumulative_time += stop['time_from_previous']
            elif 'cumulative_time' in stop:
                cumulative_time = stop['cumulative_time']
            
            # Update load based on demand
            demand = stop.get('demand', 0)
            if demand > 0:
                current_load += demand
            elif demand < 0:
                current_load += demand  # demand is negative for delivery
            
            # Time window info
            time_window_start = stop.get('time_window_start', 0)
            time_window_end = stop.get('time_window_end', 24)  # 24 hours default
            
            # Arrival/departure time in hours:minutes format
            arrival_time = stop.get('arrival_time', cumulative_time / 60)  # Convert to hours
            departure_time = stop.get('departure_time', arrival_time)
            
            # Convert to hours:minutes format
            arr_hours = int(arrival_time)
            arr_minutes = int((arrival_time % 1) * 60)
            dep_hours = int(departure_time)
            dep_minutes = int((departure_time % 1) * 60)
            
            # Service time
            service_time = stop.get('service_time', 0)
            if service_time > 0:
                departure_time = arrival_time + service_time / 60
                dep_hours = int(departure_time)
                dep_minutes = int((departure_time % 1) * 60)
            
            # Capacity info
            capacity_info = stop.get('capacity_analysis', {})
            weight_capacity = capacity_info.get('vehicle_weight_capacity', 1000)
            volume_capacity = capacity_info.get('vehicle_volume_capacity', 20)
            current_weight = capacity_info.get('current_weight', current_load)
            current_volume = capacity_info.get('current_volume', current_load * 0.02)  # Estimate
            
            if stop_day not in days_data:
                days_data[stop_day] = []
            
            days_data[stop_day].append({
                'type': 'regular',
                'location_id': location_id,
                'day': stop_day,
                'distance': cumulative_distance,
                'time_window': (time_window_start, time_window_end),
                'arrival_time': (arr_hours, arr_minutes),
                'departure_time': (dep_hours, dep_minutes),
                'load': current_weight,
                'capacity': weight_capacity,
                'volume_load': current_volume,
                'volume_capacity': volume_capacity,
                'demand': demand
            })
            
            current_day = stop_day
        
        # Print each day's route
        for day in sorted(days_data.keys()):
            day_stops = days_data[day]
            print(f"  Day {day}:")
            
            route_parts = []
            day_distance = 0
            day_time = 0
            
            for stop in day_stops:
                if stop['type'] == 'overnight_stay':
                    route_parts.append(f"🌙 {stop['message']}")
                elif stop['type'] == 'morning_start':
                    route_parts.append(f"🌅 {stop['message']}")
                else:
                    # Regular stop in requested format
                    loc_id = stop['location_id']
                    dist_m = int(stop['distance'] * 1000)  # Convert km to meters
                    tw_start_hours = int(stop['time_window'][0])
                    tw_start_minutes = int((stop['time_window'][0] % 1) * 60)
                    tw_end_hours = int(stop['time_window'][1])
                    tw_end_minutes = int((stop['time_window'][1] % 1) * 60)
                    arr_time = f"{stop['arrival_time'][0]:02d}:{stop['arrival_time'][1]:02d}"
                    dep_time = f"{stop['departure_time'][0]:02d}:{stop['departure_time'][1]:02d}"
                    load = int(stop['load'])
                    capacity = int(stop['capacity'])
                    
                    stop_str = f"{loc_id} {dist_m}m TW:[{tw_start_hours:02d}:{tw_start_minutes:02d},{tw_end_hours:02d}:{tw_end_minutes:02d}] Time({arr_time},{dep_time}) Load({load}/{capacity})"
                    route_parts.append(stop_str)
                    
                    # Track day totals
                    if stop == day_stops[-1]:  # Last stop of the day
                        day_distance = stop['distance']
                        day_time = stop['departure_time'][0] * 60 + stop['departure_time'][1]
            
            # Print the day's route
            route_line = " -> ".join(route_parts)
            print(f"    {route_line}")
            
            if day_distance > 0:
                print(f"    Distance of day {day}: {int(day_distance * 1000)}m")
                print(f"    Time of day {day}: {int(day_time)}min")
        
        # Print total route summary
        total_distance = route.get('total_distance', 0)
        total_time = route.get('total_time', 0)
        total_cost = route.get('total_cost', 0)
        
        print(f"  Total route distance: {int(total_distance * 1000)}m")
        print(f"  Total route time: {int(total_time)}min")
        print(f"  Total route cost: €{total_cost:.2f}")
        
        # Check for overnight stays
        overnight_count = sum(1 for day_stops in days_data.values() 
                            for stop in day_stops 
                            if stop['type'] == 'overnight_stay')
        if overnight_count > 0:
            print(f"  Overnight stays: {overnight_count}")

def test_detailed_report():
    """Quick test of the detailed route report format."""
    print("🧪 Testing Detailed Route Report Format")
    print("=" * 60)
    
    # Create a simple scenario
    vehicles, locations, ride_requests = create_custom_multiday_scenario()
    
    print(f"📍 Test scenario:")
    print(f"  - {len(locations)} locations (including depot)")
    print(f"  - {len(vehicles)} vehicles")
    print(f"  - Testing multi-day routing with detailed reporting")
    
    # Test with multi-day settings
    optimizer = VRPOptimizer()
    
    result = optimizer.solve(
        vehicles=vehicles,
        locations=locations,
        ride_requests=ride_requests,
        constraint_level='capacity',
        multiday_max_days=3,
        multiday_minutes_per_day=480,
        verbose=False
    )
    
    if result and result.get('status') == 'SUCCESS':
        print("\n✅ Solution found!")
        print_detailed_route_report(result)
        return result
    else:
        print("\n❌ No solution found")
        return None

if __name__ == "__main__":
    test_detailed_report()
