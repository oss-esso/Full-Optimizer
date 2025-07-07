#!/usr/bin/env python3

from vrp_scenarios import create_moda_small_scenario
from vrp_optimizer_rolling_window import VRPOptimizerRollingWindow
from vrp_data_models import VRPObjective

def debug_ortools_time_reporting():
    """Debug what OR-Tools is actually calculating for time."""
    
    print("Debugging OR-Tools time reporting...")
    scenario = create_moda_small_scenario()
    
    # Get the solution
    optimizer = VRPOptimizerRollingWindow(scenario, VRPObjective.MINIMIZE_DISTANCE)
    result = optimizer.optimize_with_rolling_window()
    
    if result.status == "optimal":
        print(f"\nSolution found:")
        print(f"Routes: {result.routes}")
        
        # Calculate the time manually for the solution route
        for vehicle_id, route in result.routes.items():
            if len(route) > 2:  # Active route
                print(f"\nAnalyzing {vehicle_id} route: {route}")
                
                total_service_time = 0
                total_travel_time = 0
                
                for i, location_id in enumerate(route):
                    # Service time at this location
                    if location_id in scenario.locations:
                        location = scenario.locations[location_id]
                        service_time = getattr(location, 'service_time', 0)
                        total_service_time += service_time
                        print(f"  {i}: {location_id} - service time: {service_time} min")
                    
                    # Travel time to next location
                    if i < len(route) - 1:
                        from_loc = location_id
                        to_loc = route[i + 1]
                        distance = scenario.get_distance(from_loc, to_loc)
                        
                        # Apply same calculation as optimizer
                        distance_km = distance * 111
                        travel_time_minutes = (distance_km / 50) * 60
                        total_travel_time += travel_time_minutes
                        print(f"    -> {to_loc}: {distance:.6f} units ({distance_km:.2f} km) = {travel_time_minutes:.1f} min")
                
                print(f"\nManual calculation for {vehicle_id}:")
                print(f"  - Total service time: {total_service_time} minutes ({total_service_time/60:.1f} hours)")
                print(f"  - Total travel time: {total_travel_time:.1f} minutes ({total_travel_time/60:.1f} hours)")
                print(f"  - TOTAL TIME: {total_service_time + total_travel_time:.1f} minutes ({(total_service_time + total_travel_time)/60:.1f} hours)")
                print(f"  - OR-Tools reported: 456.9 minutes (7.6 hours)")
                print(f"  - Difference: {456.9 - (total_service_time + total_travel_time):.1f} minutes")
                
                # Check vehicle time limit
                vehicle = scenario.vehicles[vehicle_id]
                max_time = getattr(vehicle, 'max_total_work_time', 600)
                print(f"  - Vehicle time limit: {max_time} minutes ({max_time/60:.1f} hours)")
                
                if (total_service_time + total_travel_time) > max_time:
                    print(f"  ❌ VIOLATION: Route exceeds time limit by {(total_service_time + total_travel_time) - max_time:.1f} minutes!")
                else:
                    print(f"  ✅ Route is within time limit")
    else:
        print(f"No solution found: {result.status}")

if __name__ == "__main__":
    debug_ortools_time_reporting()
