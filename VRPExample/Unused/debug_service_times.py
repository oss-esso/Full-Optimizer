#!/usr/bin/env python3
"""
Debug service time calculation to understand the variation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from VRPExample.vrp_scenarios import get_all_scenarios

def debug_service_times():
    """Debug the actual service times in the data."""
    print("🔍 DEBUGGING SERVICE TIMES IN SCENARIO DATA")
    print("=" * 60)
    
    # Get scenario
    scenarios = get_all_scenarios()
    scenario = scenarios['MODA_first']
    
    # Analyze service times
    service_time_stats = {}
    location_types = {'depot': [], 'pickup': [], 'dropoff': [], 'other': []}
    
    for loc_id, location in scenario.locations.items():
        service_time = getattr(location, 'service_time', None)
        
        # Categorize location
        if 'depot' in loc_id.lower():
            category = 'depot'
        elif 'pickup' in loc_id.lower():
            category = 'pickup'
        elif 'dropoff' in loc_id.lower():
            category = 'dropoff'
        else:
            category = 'other'
        
        location_types[category].append({
            'id': loc_id,
            'service_time': service_time
        })
        
        if service_time is not None:
            if service_time not in service_time_stats:
                service_time_stats[service_time] = 0
            service_time_stats[service_time] += 1
    
    print(f"📊 SERVICE TIME DISTRIBUTION:")
    for time, count in sorted(service_time_stats.items()):
        print(f"   {time} minutes: {count} locations")
    
    print(f"\n📊 BY LOCATION TYPE:")
    for category, locations in location_types.items():
        if locations:
            service_times = [loc['service_time'] for loc in locations if loc['service_time'] is not None]
            if service_times:
                avg_service_time = sum(service_times) / len(service_times)
                print(f"   {category}: {len(locations)} locations, avg service time: {avg_service_time:.1f}min")
                print(f"      Range: {min(service_times)}-{max(service_times)}min")
                
                # Show a few examples
                for i, loc in enumerate(locations[:3]):
                    print(f"      Example: {loc['id']} = {loc['service_time']}min")
            else:
                print(f"   {category}: {len(locations)} locations, no service times set")
    
    # Calculate expected service time for a 6-stop route
    print(f"\n📊 EXPECTED SERVICE TIME FOR 6-STOP ROUTE:")
    
    # Simulate a typical route with 3 pickups + 3 dropoffs
    example_route_service_time = 0
    pickup_count = 0
    dropoff_count = 0
    
    for category, locations in location_types.items():
        if category == 'pickup' and pickup_count < 3:
            for loc in locations[:3]:
                if loc['service_time'] is not None:
                    example_route_service_time += loc['service_time']
                    pickup_count += 1
                    print(f"   Pickup {loc['id']}: +{loc['service_time']}min")
        elif category == 'dropoff' and dropoff_count < 3:
            for loc in locations[:3]:
                if loc['service_time'] is not None:
                    example_route_service_time += loc['service_time']
                    dropoff_count += 1
                    print(f"   Dropoff {loc['id']}: +{loc['service_time']}min")
    
    print(f"   Total expected service time: {example_route_service_time}min")
    print(f"   Vs standard 6×15min = 90min")

if __name__ == "__main__":
    debug_service_times()
