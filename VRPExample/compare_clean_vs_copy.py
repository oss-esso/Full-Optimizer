"""
Compare CleanVRPOptimizer (original vs copy) on the MODA_small scenario.
Demonstrates the difference in constraints and reporting.
"""
import logging
import sys

# Import both optimizers with aliasing
from vrp_optimizer_clean import CleanVRPOptimizer as CleanVRPOptimizerOriginal
from vrp_optimizer_clean_copy import CleanVRPOptimizer as CleanVRPOptimizerCopy

# Import the scenario creator
try:
    from vrp_scenarios import create_furgoni_scenario
except ImportError:
    print("❌ Could not import vrp_scenarios. Make sure the file is available.")
    sys.exit(1)

def run_test():
    scenario = create_furgoni_scenario()
    print("\n=== Scenario details ===")
    print(f"Locations: {len(scenario.locations)}")
    print(f"Vehicles: {len(scenario.vehicles)}")
    print(f"Ride requests: {len(scenario.ride_requests)}")

    # Prepare vehicle dicts
    vehicle_ids = list(scenario.vehicles.keys())
    vehicles_from_scenario = [scenario.vehicles[vid] for vid in vehicle_ids]
    vehicles_dicts = [{
        'id': v.id,
        'capacity': v.capacity,
        'start_location': v.depot_id,
        'end_location': v.depot_id,
        'max_time': getattr(v, 'max_time', 24 * 60)
    } for v in vehicles_from_scenario]

    print("\n================ ORIGINAL CLEAN OPTIMIZER ================")
    optimizer_orig = CleanVRPOptimizerOriginal(vehicles=vehicles_dicts, locations=None, vrp_instance=scenario)
    optimizer_orig.ride_requests = scenario.ride_requests
    result_orig = optimizer_orig.solve(constraint_level="full", verbose=False)
    print("\n--- Original CleanVRPOptimizer Output ---")
    if result_orig:
        result, status, applied_constraints = result_orig
        print(f"Total distance: {result.get('total_distance', 'N/A')}")
        print(f"Objective value: {result.get('objective_value', 'N/A')}")
        print(f"Validation: {result.get('validation_results', {})}")
        print(f"Status: {status}")
        print(f"Applied constraints: {applied_constraints}")
    else:
        print("No solution found.")

    print("\n================ COPY (DUAL CAPACITY, OSM) OPTIMIZER ================")
    optimizer_copy = CleanVRPOptimizerCopy(vehicles=vehicles_dicts, locations=None, vrp_instance=scenario)
    optimizer_copy.ride_requests = scenario.ride_requests
    result_copy = optimizer_copy.solve(constraint_level="full", verbose=False)
    print("\n--- Copy CleanVRPOptimizer Output ---")
    if result_copy:
        print(f"Total distance: {result_copy.get('total_distance', 'N/A')}")
        print(f"Total cost: {result_copy.get('total_cost', 'N/A')}")
        print(f"Truck speed enabled: {result_copy.get('truck_speed_enabled', False)}")
        print(f"Applied constraints: {result_copy.get('applied_constraints', [])}")
        print("(See above for detailed dual capacity and per-stop analysis)")
    else:
        print("No solution found.")

    # After both optimizers have run:
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    if result_orig:
        result, status, applied_constraints = result_orig
        print(f"Original: {applied_constraints} constraints")
        print_solution_summary("Original", result, vehicles_dicts, is_copy=False)
    
    if result_copy:
        applied_constraints_copy = result_copy.get('applied_constraints', [])
        print(f"Copy: {applied_constraints_copy} constraints")
        print_solution_summary("Copy", result_copy, vehicles_dicts, is_copy=True)
    
    print("\nKey Differences:")
    print("- Original: Basic capacity, pickup-delivery, time window constraints + validation")
    print("- Copy: OSM routing, dual capacity (weight+volume), vehicle-specific costs + all constraints")
    print("- Copy now implements ALL constraint types for 'full' level")

def print_solution_summary(label, result, vehicles_dicts, is_copy=False):
    print(f"\n=== {label} Solution Summary ===")
    routes = result.get('routes', {})
    
    if is_copy:
        # Copy optimizer: routes is a list of route objects with 'stops' and other fields
        for i, route_data in enumerate(routes):
            stops = route_data.get('stops', [])
            if not stops:
                continue
            # Count non-depot stops
            customer_stops = [s for s in stops if 'depot' not in str(s.get('location_id', '')).lower()]
            num_customers = len(customer_stops)
            # Get distance and convert to hours (assume 50 km/h average speed)
            distance_km = route_data.get('total_distance', 0)
            hours = distance_km / 50 if distance_km else 0
            vehicle_id = route_data.get('vehicle_id', f'Vehicle_{i}')
            
            # Check if this vehicle violates its time constraint
            vehicle_data = next((v for v in vehicles_dicts if v['id'] == vehicle_id), None)
            max_hours = vehicle_data.get('max_time', 900) / 60 if vehicle_data else 15.0  # Convert minutes to hours
            
            violation_flag = "⚠️ TIME VIOLATION" if hours > max_hours else ""
            print(f"Vehicle {vehicle_id}: {num_customers} customers, {hours:.2f}h driven (limit: {max_hours:.1f}h) {violation_flag}")
    else:
        # Original optimizer: routes is a dict with vehicle_id as key
        for vehicle_id, route_data in routes.items():
            stops = route_data.get('route', [])
            if not stops:
                continue
            # Count non-depot stops
            customer_stops = [s for s in stops if 'depot' not in str(s.get('location_id', s.get('id', ''))).lower()]
            num_customers = len(customer_stops)
            
            # The original optimizer's 'time' field is not properly populated (always 0)
            # So we calculate driving time from distance using a realistic speed
            distance_km = route_data.get('distance', 0)
            driving_time_min = distance_km * 60 / 60 if distance_km else 0  # 60 km/h average
            hours = driving_time_min / 60 if driving_time_min else 0
            
            # Check if this vehicle violates its time constraint
            vehicle_data = next((v for v in vehicles_dicts if v['id'] == vehicle_id), None)
            max_hours = vehicle_data.get('max_time', 900) / 60 if vehicle_data else 15.0  # Convert minutes to hours
            
            violation_flag = "⚠️ TIME VIOLATION" if hours > max_hours else ""
            print(f"Vehicle {vehicle_id}: {num_customers} customers, {hours:.2f}h driven (limit: {max_hours:.1f}h) {violation_flag}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_test()
