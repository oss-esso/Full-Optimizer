#!/usr/bin/env python3
"""
VRP Optimizer with proper numeric handling and simplified constraints.
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

from vrp_data_models import VRPInstance

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VRPOptimizerNumericFixed:
    """VRP optimizer with proper numeric handling."""
    
    def solve(self, instance: VRPInstance, time_limit_seconds: int = 60) -> Dict[str, Any]:
        """Solve VRP with proper numeric constraints."""
        logger.info("🚀 STARTING NUMERIC-FIXED VRP OPTIMIZER")
        
        if not ORTOOLS_AVAILABLE:
            return {'success': False, 'error': 'OR-Tools not available'}
        
        try:
            # Setup
            num_locations = len(instance.locations)
            num_vehicles = len(instance.vehicles)
            depot_index = 0
            
            logger.info(f"Model: {num_locations} locations, {num_vehicles} vehicles")
            
            manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, depot_index)
            routing = pywrapcp.RoutingModel(manager)
            
            location_list = list(instance.locations.values())
            
            # 1. Simple distance callback with bounds checking
            def distance_callback(from_index, to_index):
                try:
                    # Validate indices first
                    if from_index < 0 or to_index < 0:
                        return 1000
                    if from_index >= num_locations * num_vehicles or to_index >= num_locations * num_vehicles:
                        return 1000
                        
                    from_node = manager.IndexToNode(from_index)
                    to_node = manager.IndexToNode(to_index)
                    
                    # Bounds check
                    if from_node < 0 or from_node >= num_locations or to_node < 0 or to_node >= num_locations:
                        return 1000
                    
                    from_loc = location_list[from_node]
                    to_loc = location_list[to_node]
                    
                    # Simple Manhattan distance
                    dx = abs(from_loc.x - to_loc.x)
                    dy = abs(from_loc.y - to_loc.y)
                    distance = int(dx + dy + 1)  # +1 to avoid zero distance
                    
                    # Clamp to reasonable range
                    return max(1, min(distance, 10000))
                    
                except Exception as e:
                    logger.warning(f"Distance callback error: {e}")
                    return 1000
            
            transit_callback_index = routing.RegisterTransitCallback(distance_callback)
            routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
            logger.info("✅ Added distance constraints")
            
            # 2. Add time windows without time dimension first (simpler)
            time_windows_added = 0
            has_time_windows = any(hasattr(loc, 'time_window_start') and loc.time_window_start is not None 
                                 for loc in location_list)
            
            if has_time_windows:
                # Add time dimension
                routing.AddDimension(
                    transit_callback_index,  # Use distance as proxy for time
                    30,   # 30 units slack (waiting time)
                    1000, # 1000 units max route duration
                    False, # Don't force start cumul to zero
                    'Time'
                )
                time_dimension = routing.GetDimensionOrDie('Time')
                
                # Add time window constraints with scaling
                for location_idx, location in enumerate(location_list):
                    if hasattr(location, 'time_window_start') and location.time_window_start is not None:
                        try:
                            index = manager.NodeToIndex(location_idx)
                            
                            # Scale time windows to reasonable range (0-1000)
                            start_scaled = max(0, min(int(location.time_window_start // 2), 1000))
                            end_scaled = max(start_scaled + 10, min(int(location.time_window_end // 2), 1000))
                            
                            time_dimension.CumulVar(index).SetRange(start_scaled, end_scaled)
                            time_windows_added += 1
                            
                        except Exception as e:
                            logger.warning(f"Time window error for location {location_idx}: {e}")
                
                logger.info(f"✅ Added {time_windows_added} time windows (scaled)")
            
            # 3. Simple pickup-delivery constraints (no capacity for now)
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
                        
                        pickup_delivery_pairs += 1
                        
                    except (ValueError, Exception) as e:
                        logger.warning(f"Skipping request {req.id}: {e}")
                
                logger.info(f"✅ Added {pickup_delivery_pairs} pickup-delivery pairs")
            
            # 4. Solve with multiple strategies
            strategies = [
                routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
                routing_enums_pb2.FirstSolutionStrategy.SAVINGS,
                routing_enums_pb2.FirstSolutionStrategy.PATH_MOST_CONSTRAINED_ARC,
            ]
            
            for i, strategy in enumerate(strategies):
                logger.info(f"🔍 Trying strategy {i+1}/{len(strategies)}: {strategy}")
                
                search_parameters = pywrapcp.DefaultRoutingSearchParameters()
                search_parameters.first_solution_strategy = strategy
                search_parameters.time_limit.FromSeconds(time_limit_seconds // len(strategies))
                
                start_time = time.time()
                assignment = routing.SolveWithParameters(search_parameters)
                solve_time = time.time() - start_time
                
                if assignment:
                    objective_value = routing.GetObjectiveValue(assignment)
                    logger.info(f"✅ SOLUTION FOUND with strategy {strategy}!")
                    logger.info(f"   Objective: {objective_value}")
                    logger.info(f"   Time: {solve_time:.2f}s")
                    
                    # Extract routes
                    routes = {}
                    vehicles_used = 0
                    
                    for vehicle_id in range(routing.vehicles()):
                        route = []
                        index = routing.Start(vehicle_id)
                        
                        while not routing.IsEnd(index):
                            node_index = manager.IndexToNode(index)
                            route.append(location_list[node_index].id)
                            index = assignment.Value(routing.NextVar(index))
                        
                        if len(route) > 1:  # Vehicle was used
                            routes[f"vehicle_{vehicle_id}"] = route
                            vehicles_used += 1
                    
                    return {
                        'success': True,
                        'objective_value': objective_value,
                        'routes': routes,
                        'vehicles_used': vehicles_used,
                        'solve_time': solve_time,
                        'strategy_used': str(strategy),
                        'constraints': {
                            'time_windows': time_windows_added,
                            'pickup_delivery_pairs': pickup_delivery_pairs
                        }
                    }
                else:
                    logger.info(f"❌ No solution with strategy {strategy}")
            
            return {
                'success': False,
                'error': 'No feasible solution found with any strategy',
                'constraints': {
                    'time_windows': time_windows_added,
                    'pickup_delivery_pairs': pickup_delivery_pairs
                }
            }
                
        except Exception as e:
            logger.error(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

def test_numeric_fixed():
    """Test the numeric-fixed optimizer."""
    from vrp_scenarios import create_moda_small_scenario
    
    print("Testing Numeric-Fixed VRP Optimizer")
    print("=" * 50)
    
    try:
        # Create smaller test first
        scenario = create_moda_small_scenario()
        
        # Create simplified version for testing
        simplified = create_simple_test_scenario()
        
        # Test simplified first
        print("\n1. Testing simplified scenario...")
        optimizer = VRPOptimizerNumericFixed()
        result = optimizer.solve(simplified, time_limit_seconds=30)
        
        print(f"   Success: {result['success']}")
        if result['success']:
            print(f"   Vehicles used: {result['vehicles_used']}")
            print(f"   Strategy: {result['strategy_used']}")
        
        # Test full scenario
        print("\n2. Testing MODA small scenario...")
        result = optimizer.solve(scenario, time_limit_seconds=60)
        
        print(f"   Success: {result['success']}")
        if result['success']:
            print(f"   Objective: {result['objective_value']}")
            print(f"   Vehicles used: {result['vehicles_used']}")
            print(f"   Routes: {len(result['routes'])}")
            print(f"   Strategy: {result['strategy_used']}")
            print(f"   Time: {result['solve_time']:.2f}s")
            
            for vehicle, route in result['routes'].items():
                print(f"     {vehicle}: {len(route)} stops")
        else:
            print(f"   Error: {result.get('error', 'Unknown')}")
            if 'constraints' in result:
                print(f"   Constraints added: {result['constraints']}")
            
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

def create_simple_test_scenario():
    """Create a very simple test scenario."""
    from vrp_data_models import VRPInstance, Location, Vehicle, RideRequest
    
    instance = VRPInstance("Simple Test")
    
    # Add locations
    instance.add_location(Location("depot", 0, 0))
    instance.add_location(Location("pickup_1", 1, 1))
    instance.add_location(Location("dropoff_1", 2, 2))
    instance.add_location(Location("pickup_2", 3, 1))
    instance.add_location(Location("dropoff_2", 4, 2))
    
    # Add vehicles
    instance.add_vehicle(Vehicle("vehicle_1", capacity=100, depot_id="depot"))
    instance.add_vehicle(Vehicle("vehicle_2", capacity=100, depot_id="depot"))
    
    # Add requests
    instance.add_ride_request(RideRequest("req_1", "pickup_1", "dropoff_1", passengers=10))
    instance.add_ride_request(RideRequest("req_2", "pickup_2", "dropoff_2", passengers=15))
    
    return instance

if __name__ == "__main__":
    test_numeric_fixed()
