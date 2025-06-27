"""
Compare CleanVRPOptimizer (original vs copy) on the MODA_small scenario.
Demonstrates the difference in constraints and reporting.
"""
import logging
import sys

# Import both optimizers with aliasing
from VRPExample.vrp_optimizer_clean import CleanVRPOptimizer as CleanVRPOptimizerOriginal
from VRPExample.vrp_optimizer_clean copy import CleanVRPOptimizer as CleanVRPOptimizerCopy

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
        print(f"Total distance: {result_orig.get('total_distance', 'N/A')}")
        print(f"Objective value: {result_orig.get('objective_value', 'N/A')}")
        print(f"Validation: {result_orig.get('validation_results', {})}")
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

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_test()
