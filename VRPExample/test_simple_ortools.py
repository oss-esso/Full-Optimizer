import os
import sys

# Add parent directory to path for absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from vrp_scenarios_backup import VRPScenarioGenerator
from vrp_optimizer_rolling_window import VRPOptimizerRollingWindow

try:
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False

def test_simple_ortools():
    """Test if the issue is with the rolling window optimizer itself."""
    
    print("TESTING SIMPLE OR-TOOLS SOLVER")
    print("=" * 40)
    
    if not ORTOOLS_AVAILABLE:
        print("OR-Tools not available!")
        return
    
    gen = VRPScenarioGenerator()
    scenario = gen.create_small_delivery_scenario()
    
    print(f"Scenario: {scenario.name}")
    print(f"Locations: {len(scenario.locations)}")
    print(f"Vehicles: {len(scenario.vehicles)}")
    
    # Create a simple OR-Tools solver without any time constraints
    num_locations = len(scenario.locations)
    num_vehicles = len(scenario.vehicles)
    
    # Create index manager
    manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, 0)  # Single depot at index 0
    
    # Create routing model
    routing = pywrapcp.RoutingModel(manager)
    
    # Create distance callback
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(scenario.distance_matrix[from_node][to_node])
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # Add capacity constraint
    def demand_callback(from_index):
        """Returns the demand of the node."""
        from_node = manager.IndexToNode(from_index)
        location_id = list(scenario.locations.keys())[from_node]
        return scenario.locations[location_id].demand
    
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    
    # Add capacity dimension
    capacities = [vehicle.capacity for vehicle in scenario.vehicles]
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        capacities,  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity'
    )
    
    # NO TIME CONSTRAINTS AT ALL - just basic VRP
    
    # Set search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.time_limit.seconds = 30
    
    print("\nSolving with basic OR-Tools (no time constraints)...")
    
    # Solve
    solution = routing.SolveWithParameters(search_parameters)
    
    if solution:
        print("✅ SOLUTION FOUND!")
        print(f"Total distance: {solution.ObjectiveValue()}")
        
        # Print solution
        for vehicle_id in range(num_vehicles):
            index = routing.Start(vehicle_id)
            route = []
            route_distance = 0
            
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                location_id = list(scenario.locations.keys())[node_index]
                route.append(location_id)
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
            
            # Add final location
            node_index = manager.IndexToNode(index)
            location_id = list(scenario.locations.keys())[node_index]
            route.append(location_id)
            
            print(f"Vehicle {vehicle_id}: {' -> '.join(route)} (distance: {route_distance})")
    else:
        print("❌ NO SOLUTION FOUND even with basic OR-Tools!")
        
    print("\n" + "=" * 40)
    print("This tests whether the rolling window optimizer has fundamental issues")
    print("or if it's just the time constraint implementation.")

if __name__ == "__main__":
    test_simple_ortools()
