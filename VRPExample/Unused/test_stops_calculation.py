#!/usr/bin/env python3

from vrp_scenarios import create_moda_first_scenario

def main():
    print("Creating MODA_first scenario...")
    scenario = create_moda_first_scenario()
    
    total_stops = len(scenario.ride_requests) * 2  # pickup + dropoff
    num_vehicles = len(scenario.vehicles)
    
    print(f'Total ride requests: {len(scenario.ride_requests)}')
    print(f'Total stops (pickup + dropoff): {total_stops}')
    print(f'Number of vehicles: {num_vehicles}')
    print(f'Average stops per vehicle: {total_stops / num_vehicles:.2f}')
    print(f'Minimum stops per vehicle if evenly distributed: {total_stops // num_vehicles}')
    print(f'Maximum stops per vehicle if evenly distributed: {(total_stops + num_vehicles - 1) // num_vehicles}')
    
    # Current constraint is 8 stops per vehicle
    print(f'\nCurrent constraint: 8 stops per vehicle')
    print(f'Total capacity with current constraint: {8 * num_vehicles} stops')
    print(f'Excess demand: {total_stops - (8 * num_vehicles)} stops')
    
    # Suggested new constraint
    suggested_max = (total_stops + num_vehicles - 1) // num_vehicles + 2  # Add some buffer
    print(f'\nSuggested new constraint: {suggested_max} stops per vehicle')
    print(f'Total capacity with suggested constraint: {suggested_max * num_vehicles} stops')
    print(f'Buffer capacity: {(suggested_max * num_vehicles) - total_stops} stops')

if __name__ == "__main__":
    main()
