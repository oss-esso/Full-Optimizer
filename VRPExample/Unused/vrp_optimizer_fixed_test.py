#!/usr/bin/env python3
"""
Fixed VRP Optimizer that properly handles time windows and constraints.
"""

import logging
import time
from typing import Dict, List, Tuple, Optional, Any

try:
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False
    routing_enums_pb2 = None
    pywrapcp = None

from vrp_data_models import VRPInstance, VRPResult

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VRPOptimizerFixed:
    """VRP optimizer with properly configured OR-Tools constraints."""
    
    def __init__(self):
        self.diagnostic_info = {}
    
    def solve(self, instance: VRPInstance, time_limit_seconds: int = 300) -> Dict[str, Any]:
        """Solve VRP with proper OR-Tools configuration."""
        logger.info("=" * 60)
        logger.info("FIXED VRP OPTIMIZER - PROPER CONSTRAINTS")
        logger.info("=" * 60)
        
        if not ORTOOLS_AVAILABLE:
            return {'success': False, 'error': 'OR-Tools not available'}
        
        try:
            # Create model
            num_locations = len(instance.locations)
            num_vehicles = len(instance.vehicles)
            depot_index = 0  # Use first location as depot
            
            logger.info(f"Creating model: {num_locations} locations, {num_vehicles} vehicles")
            
            manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, depot_index)
            routing = pywrapcp.RoutingModel(manager)
            
            # Store location list for consistent indexing
            location_list = list(instance.locations.values())
            
            # 1. Distance callback
            def distance_callback(from_index, to_index):
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                
                if from_node >= len(location_list) or to_node >= len(location_list):
                    return 0
                    
                from_loc = location_list[from_node]
                to_loc = location_list[to_node]
                
                if hasattr(instance, 'distance_matrix') and instance.distance_matrix is not None:
                    return int(instance.distance_matrix[from_node][to_node])
                else:
                    # Manhattan distance
                    dx = abs(from_loc.x - to_loc.x)
                    dy = abs(from_loc.y - to_loc.y)
                    return int((dx + dy) * 10)  # Scale for precision
            
            transit_callback_index = routing.RegisterTransitCallback(distance_callback)
            routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
            logger.info("✅ Added distance constraints")
            
            # 2. Time dimension
            def time_callback(from_index, to_index):
                # Convert distance to time (assume 1 unit distance = 1 minute)
                return distance_callback(from_index, to_index) // 10
            
            time_callback_index = routing.RegisterTransitCallback(time_callback)
            
            routing.AddDimension(
                time_callback_index,
                60,   # 1 hour slack
                1440, # 24 hours max route time
                False, # Don't force start cumul to zero
                'Time'
            )
            time_dimension = routing.GetDimensionOrDie('Time')
            logger.info("✅ Added time dimension")
            
            # 3. Time windows
            time_windows_added = 0
            for location_idx, location in enumerate(location_list):
                if hasattr(location, 'time_window_start') and location.time_window_start is not None:
                    index = manager.NodeToIndex(location_idx)
                    time_dimension.CumulVar(index).SetRange(
                        int(location.time_window_start),
                        int(location.time_window_end)
                    )
                    time_windows_added += 1
            
            logger.info(f"✅ Added {time_windows_added} time windows")
            
            # 4. Capacity constraints
            if instance.ride_requests:
                def demand_callback(from_index):
                    from_node = manager.IndexToNode(from_index)
                    if from_node >= len(location_list):
                        return 0
                        
                    location = location_list[from_node]
                    demand = 0
                    
                    # Sum up cargo for this location
                    for req in instance.ride_requests:
                        if req.pickup_location == location.id:
                            demand += int(req.passengers)
                        elif req.dropoff_location == location.id:
                            demand -= int(req.passengers)
                    
                    return demand
                
                demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
                
                vehicle_capacities = [int(v.capacity) for v in instance.vehicles.values()]
                
                routing.AddDimensionWithVehicleCapacity(
                    demand_callback_index,
                    0,  # No slack
                    vehicle_capacities,
                    True,  # Start cumul at zero
                    'Capacity'
                )
                logger.info("✅ Added capacity constraints")
            
            # 5. Pickup and delivery constraints
            pickup_delivery_pairs = 0
            if instance.ride_requests:
                location_ids = [loc.id for loc in location_list]
                
                for req in instance.ride_requests:
                    try:
                        pickup_idx = location_ids.index(req.pickup_location)
                        dropoff_idx = location_ids.index(req.dropoff_location)
                        
                        pickup_index = manager.NodeToIndex(pickup_idx)
                        dropoff_index = manager.NodeToIndex(dropoff_idx)
                        
                        # Add pickup-delivery pair
                        routing.AddPickupAndDelivery(pickup_index, dropoff_index)
                        
                        # Same vehicle constraint
                        routing.solver().Add(
                            routing.VehicleVar(pickup_index) == routing.VehicleVar(dropoff_index)
                        )
                        
                        # Pickup before dropoff
                        routing.solver().Add(
                            time_dimension.CumulVar(pickup_index) <= time_dimension.CumulVar(dropoff_index)
                        )
                        
                        pickup_delivery_pairs += 1
                        
                    except ValueError:
                        logger.warning(f"Skipping request {req.id} - location not found")
                
                logger.info(f"✅ Added {pickup_delivery_pairs} pickup-delivery pairs")
            
            # 6. Solve
            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
            search_parameters.time_limit.FromSeconds(time_limit_seconds)
            search_parameters.log_search = True
            
            logger.info("🔍 Solving...")
            start_time = time.time()
            assignment = routing.SolveWithParameters(search_parameters)
            solve_time = time.time() - start_time
            
            if assignment:
                total_distance = routing.GetObjectiveValue(assignment)
                logger.info(f"✅ SOLUTION FOUND!")
                logger.info(f"   Objective value: {total_distance}")
                logger.info(f"   Solve time: {solve_time:.2f}s")
                
                # Extract routes
                routes = {}
                for vehicle_id in range(routing.vehicles()):
                    route = []
                    index = routing.Start(vehicle_id)
                    
                    while not routing.IsEnd(index):
                        node_index = manager.IndexToNode(index)
                        route.append(location_list[node_index].id)
                        index = assignment.Value(routing.NextVar(index))
                    
                    if len(route) > 1:  # Vehicle was used
                        routes[f"vehicle_{vehicle_id}"] = route
                
                return {
                    'success': True,
                    'objective_value': total_distance,
                    'routes': routes,
                    'solve_time': solve_time,
                    'constraints_added': {
                        'time_windows': time_windows_added,
                        'pickup_delivery_pairs': pickup_delivery_pairs,
                        'vehicles': num_vehicles
                    }
                }
            else:
                logger.error("❌ NO SOLUTION FOUND")
                return {
                    'success': False,
                    'error': 'No feasible solution found',
                    'solve_time': solve_time
                }
                
        except Exception as e:
            logger.error(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

def test_fixed_optimizer():
    """Test the fixed optimizer on MODA scenario."""
    from vrp_scenarios import create_moda_small_scenario
    
    print("Testing Fixed VRP Optimizer")
    print("=" * 40)
    
    try:
        # Create scenario
        scenario = create_moda_small_scenario()
        
        # Test with fixed optimizer
        optimizer = VRPOptimizerFixed()
        result = optimizer.solve(scenario, time_limit_seconds=60)
        
        print(f"\nRESULT:")
        print(f"Success: {result['success']}")
        
        if result['success']:
            print(f"Objective: {result['objective_value']}")
            print(f"Routes found: {len(result['routes'])}")
            print(f"Solve time: {result['solve_time']:.2f}s")
            
            for vehicle, route in result['routes'].items():
                print(f"  {vehicle}: {len(route)} stops")
        else:
            print(f"Error: {result.get('error', 'Unknown')}")
            
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fixed_optimizer()
