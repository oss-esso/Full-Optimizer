import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from VRPExample.vrp_optimizer_clean import CleanVRPOptimizer

def main():
    """Test the CleanVRPOptimizer with different constraint levels."""

    # 1. Define the Vehicle List
    vehicle_list = [
        {"id": 0, "capacity": 15, "start_location": "A", "end_location": "A"},
        {"id": 1, "capacity": 15, "start_location": "A", "end_location": "A"},
        {"id": 2, "capacity": 15, "start_location": "A", "end_location": "A"},
        {"id": 3, "capacity": 15, "start_location": "A", "end_location": "A"},
    ]

    # 2. Define the Location List
    location_list = [
        {"id": "A", "demand": 0, "time_window": (0, 0)},  # Depot
        {"id": "B", "demand": -1, "time_window": (7, 12), "pickup": "C"},
        {"id": "C", "demand": 1, "time_window": (7, 12), "delivery": "B"},
        {"id": "D", "demand": -1, "time_window": (5, 10), "pickup": "E"},
        {"id": "E", "demand": 1, "time_window": (5, 10), "delivery": "D"},
        {"id": "F", "demand": 2, "time_window": (8, 15)},
        {"id": "G", "demand": 1, "time_window": (9, 14)},
    ]

    # 3. Test each constraint level
    constraint_levels = ["none", "capacity", "pickup_delivery", "time_windows", "full"]

    for level in constraint_levels:
        print(f"\n----- TESTING CONSTRAINT LEVEL: {level} -----")
        optimizer = CleanVRPOptimizer(
            vehicles=vehicle_list,
            locations=location_list,
            distance_matrix_provider="google", # or another provider
        )

        solution, status, constraints = optimizer.solve(constraint_level=level)

        print(f"Solution Status: {status}")
        print(f"Constraints Applied: {constraints}")
        if solution:
            print("Solution Found:")
            for vehicle_id, route in solution.items():
                print(f"  Vehicle {vehicle_id}: {route}")
        else:
            print("No solution found.")

if __name__ == "__main__":
    main()
