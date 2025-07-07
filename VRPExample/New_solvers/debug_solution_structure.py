#!/usr/bin/env python3
import json

# Load the most recent solution file
with open('complete_furgoni_results_20250629_160504.json', 'r') as f:
    solution = json.load(f)

print('=== SOLUTION STRUCTURE ===')
print('Top-level keys:', list(solution.keys()))
print()

if 'vehicle_routes' in solution:
    print('=== VEHICLE_ROUTES STRUCTURE ===')
    for vid, route_data in solution['vehicle_routes'].items():
        print(f'Vehicle {vid} keys:', list(route_data.keys()))
        if 'full_route' in route_data:
            print(f'  full_route length: {len(route_data["full_route"])}')
            print(f'  first 3 items: {route_data["full_route"][:3]}')
        break
print()

if 'daily_solutions' in solution:
    print('=== DAILY_SOLUTIONS STRUCTURE ===')
    days = list(solution['daily_solutions'].keys())
    print(f'Days: {days}')
    if days:
        day_sol = solution['daily_solutions'][days[0]]
        print(f'Day {days[0]} keys:', list(day_sol.keys()))
        if 'routes' in day_sol:
            vehicles = list(day_sol['routes'].keys())
            if vehicles:
                print(f'  Vehicle {vehicles[0]} route keys:', list(day_sol['routes'][vehicles[0]].keys()))
                vehicle_route = day_sol['routes'][vehicles[0]]
                if 'route' in vehicle_route:
                    print(f'    route length: {len(vehicle_route["route"])}')
                    print(f'    first 3 route items: {vehicle_route["route"][:3]}')
