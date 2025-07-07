#!/usr/bin/env python3
"""
Simple test for the detailed route report function.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mock a result to test the reporting function
mock_result = {
    'status': 'SUCCESS',
    'routes': [
        {
            'vehicle_id': 'truck_24t',
            'vehicle_capacity_kg': 1000,
            'total_distance': 47.97,  # km
            'total_cost': 86.35,
            'stops': [
                {'location_id': 'depot', 'day': 1, 'weight_usage_kg': 0},
                {'location_id': 'location_1', 'day': 1, 'weight_usage_kg': 0},
                {'location_id': 'location_2', 'day': 1, 'weight_usage_kg': 100},
                {'location_id': 'morning_day_2', 'type': 'morning_start', 'day': 2, 'message': 'Start of day 2'},
                {'location_id': 'location_far_3', 'day': 2, 'weight_usage_kg': 250},
                {'location_id': 'morning_day_3', 'type': 'morning_start', 'day': 3, 'message': 'Start of day 3'},
            ]
        },
        {
            'vehicle_id': 'van_4t',
            'vehicle_capacity_kg': 500,
            'total_distance': 147.59,  # km
            'total_cost': 140.21,
            'stops': [
                {'location_id': 'depot', 'day': 1, 'weight_usage_kg': 0},
                {'location_id': 'night_day_1', 'type': 'overnight_stay', 'day': 1, 'message': 'End of day 1 - overnight stay'},
                {'location_id': 'location_far_2', 'day': 1, 'weight_usage_kg': 0},
                {'location_id': 'location_far_1', 'day': 1, 'weight_usage_kg': 250},
                {'location_id': 'night_day_2', 'type': 'overnight_stay', 'day': 1, 'message': 'End of day 1 - overnight stay'},
            ]
        }
    ]
}

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
                    'time_window': "TW:[0,86400]",
                    'arrival_time': cumulative_time_min - day_start_time,
                    'departure_time': cumulative_time_min - day_start_time,
                    'load': stop.get('weight_usage_kg', 0),
                    'capacity': route.get('vehicle_capacity_kg', 1000)
                })
                current_day = stop_day
                continue
            
            # Regular stops - estimate travel time and distance
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
                'time_window': "TW:[0,86400]",
                'arrival_time': cumulative_time_min - day_start_time,
                'departure_time': cumulative_time_min - day_start_time + service_time,
                'load': stop.get('weight_usage_kg', 0),
                'capacity': route.get('vehicle_capacity_kg', 1000)
            })
            
            # Add service time
            cumulative_time_min += service_time
            current_day = stop_day
        
        # Print each day's route
        for day in sorted(days_routes.keys()):
            day_stops = days_routes[day]
            print(f"  Day {day}:")
            
            route_parts = []
            
            for stop in day_stops:
                if stop['type'] in ['overnight', 'morning']:
                    route_parts.append(stop['name'])
                else:
                    # Format: location_id distance TW:[start,end] Time(arrival,departure) Load(current/capacity)
                    arr_minutes = int(stop['arrival_time'])
                    dep_minutes = int(stop['departure_time'])
                    
                    stop_str = f"{stop['name']} {stop['distance_m']}m {stop['time_window']} Time({arr_minutes},{dep_minutes}) Load({stop['load']}/{stop['capacity']})"
                    route_parts.append(stop_str)
            
            # Print the day's route
            route_line = " -> ".join(route_parts)
            print(f"    {route_line}")
            
            # Day summary
            if day_stops:
                last_stop = day_stops[-1]
                print(f"    Distance of day {day}: {last_stop['distance_m']}m")
                print(f"    Time of day {day}: {int(last_stop['departure_time'])}min")
        
        # Print total route summary
        total_distance = route.get('total_distance', 0)
        total_time = cumulative_time_min
        total_cost = route.get('total_cost', 0)
        
        print(f"  Total route distance: {int(total_distance * 1000)}m")
        print(f"  Total route time: {total_time}min")
        print(f"  Total route cost: €{total_cost:.2f}")
        
        # Check for overnight stays
        overnight_count = sum(1 for day_stops in days_routes.values() 
                            for stop in day_stops 
                            if stop['type'] == 'overnight')
        if overnight_count > 0:
            print(f"  Overnight stays: {overnight_count}")

if __name__ == "__main__":
    print("🧪 Testing Detailed Route Report Format")
    print_detailed_route_report(mock_result)
