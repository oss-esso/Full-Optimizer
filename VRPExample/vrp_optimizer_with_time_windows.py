#!/usr/bin/env python3
"""
Enhanced VRP optimizer that properly handles time windows.
This addresses the missing time window constraints in the current implementation.
"""

import os
import sys
import time
import logging
from typing import Dict, List, Tuple, Optional, Any

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

try:
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False

from vrp_data_models import VRPInstance, VRPResult, VRPObjective

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VRPOptimizerWithTimeWindows:
    """VRP optimizer that properly handles time window constraints."""
    
    def __init__(self, instance: VRPInstance, objective: VRPObjective = VRPObjective.MINIMIZE_DISTANCE):
        self.instance = instance
        self.objective = objective
        
    def solve(self) -> VRPResult:
        """Solve the VRP with proper time window handling."""
        if not ORTOOLS_AVAILABLE:
            return VRPResult(
                status="error",
                objective_value=0.0,
                routes={},
                metrics={"error": "OR-Tools not available"},
                runtime=0.0
            )
        
        logger.info(f"Solving VRP: {self.instance.name}")
        start_time = time.time()
        
        try:
            # Check if this is a pickup-delivery scenario
            is_pickup_delivery = bool(self.instance.ride_requests)
            
            # Create routing model
            model_data = self._create_routing_model()
            if not model_data:
                return VRPResult(status="error", objective_value=0.0, routes={}, runtime=0.0)
                
            manager = model_data['manager']
            routing = model_data['routing']
            location_list = model_data['location_list']
            location_to_index = model_data['location_to_index']
            
            # Add distance objective
            self._add_distance_constraints(manager, routing, location_list)
            
            # Add capacity constraints if needed
            if any(getattr(v, 'capacity', 0) > 0 for v in self.instance.vehicles.values()):
                self._add_capacity_constraints(manager, routing, location_list, is_pickup_delivery)
            
            # Add pickup-delivery constraints
            if is_pickup_delivery:
                self._add_pickup_delivery_constraints(manager, routing, location_to_index)
            
            # Add time window constraints (THIS IS THE KEY MISSING PIECE!)
            time_windows_added = self._add_time_window_constraints(manager, routing, location_list)
            if time_windows_added:
                logger.info("✅ Time window constraints added successfully")
            else:
                logger.info("ℹ️ No time window constraints needed")
            
            # Solve
            solution = self._solve_routing_model(routing)
            
            runtime = (time.time() - start_time) * 1000
            
            if solution:
                # Extract solution
                result = self._extract_solution(manager, routing, solution, location_list, runtime)
                logger.info(f"✅ Solution found: {result.objective_value:.1f} distance, {result.runtime:.1f}ms")
                return result
            else:
                logger.warning("❌ No solution found")
                return VRPResult(
                    status="infeasible", 
                    objective_value=0.0, 
                    routes={}, 
                    runtime=runtime
                )
                
        except Exception as e:
            logger.error(f"❌ Error during optimization: {e}")
            import traceback
            traceback.print_exc()
            return VRPResult(
                status="error",
                objective_value=0.0,
                routes={},
                metrics={"error": str(e)},
                runtime=(time.time() - start_time) * 1000
            )
    
    def _create_routing_model(self) -> Optional[Dict[str, Any]]:
        """Create the OR-Tools routing model with proper multi-depot handling."""
        is_pickup_delivery = bool(self.instance.ride_requests)
        
        if is_pickup_delivery:
            # Handle pickup-delivery scenarios
            depot_locations = [loc_id for loc_id in self.instance.location_ids if loc_id.startswith("depot")]
            
            if len(depot_locations) > 1:
                # Multi-depot scenario
                location_list = depot_locations.copy()
                location_to_index = {loc: idx for idx, loc in enumerate(depot_locations)}
                
                # Add pickup and dropoff locations
                for request in self.instance.ride_requests:
                    for loc_id in [request.pickup_location, request.dropoff_location]:
                        if loc_id not in location_to_index:
                            location_to_index[loc_id] = len(location_list)
                            location_list.append(loc_id)
                
                # Create vehicle start/end mappings
                vehicle_starts = []
                vehicle_ends = []
                depot_indices = {depot: location_to_index[depot] for depot in depot_locations}
                
                for vehicle in self.instance.vehicles.values():
                    depot_idx = depot_indices.get(vehicle.depot_id, 0)
                    vehicle_starts.append(depot_idx)
                    vehicle_ends.append(depot_idx)
                
                manager = pywrapcp.RoutingIndexManager(
                    len(location_list), len(self.instance.vehicles), 
                    vehicle_starts, vehicle_ends
                )
            else:
                # Single depot scenario
                depot_id = depot_locations[0] if depot_locations else self.instance.location_ids[0]
                location_list = [depot_id]
                location_to_index = {depot_id: 0}
                
                # Add pickup and dropoff locations
                for request in self.instance.ride_requests:
                    for loc_id in [request.pickup_location, request.dropoff_location]:
                        if loc_id not in location_to_index:
                            location_to_index[loc_id] = len(location_list)
                            location_list.append(loc_id)
                
                manager = pywrapcp.RoutingIndexManager(
                    len(location_list), len(self.instance.vehicles), 0
                )
        else:
            # Standard VRP
            depot_locations = [loc_id for loc_id in self.instance.location_ids if loc_id.startswith("depot")]
            depot_id = depot_locations[0] if depot_locations else self.instance.location_ids[0]
            
            customer_locations = [loc_id for loc_id in self.instance.location_ids if not loc_id.startswith("depot")]
            location_list = [depot_id] + customer_locations
            location_to_index = {loc: idx for idx, loc in enumerate(location_list)}
            
            manager = pywrapcp.RoutingIndexManager(
                len(location_list), len(self.instance.vehicles), 0
            )
        
        routing = pywrapcp.RoutingModel(manager)
        
        return {
            'manager': manager,
            'routing': routing,
            'location_list': location_list,
            'location_to_index': location_to_index
        }
    
    def _add_distance_constraints(self, manager, routing, location_list):
        """Add distance callback and objective."""
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            from_location_id = location_list[from_node]
            to_location_id = location_list[to_node]
            return int(self.instance.get_distance(from_location_id, to_location_id))
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    def _add_capacity_constraints(self, manager, routing, location_list, is_pickup_delivery):
        """Add capacity constraints."""
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            location_id = location_list[from_node]
            
            demand = 0
            if location_id.startswith("depot"):
                demand = 0
            elif is_pickup_delivery:
                if location_id.startswith("pickup"):
                    for request in self.instance.ride_requests:
                        if request.pickup_location == location_id:
                            demand = request.passengers
                            break
                elif location_id.startswith("dropoff"):
                    for request in self.instance.ride_requests:
                        if request.dropoff_location == location_id:
                            demand = -request.passengers
                            break
            else:
                location = self.instance.locations[location_id]
                demand = getattr(location, 'demand', 1)
            
            return demand
        
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        
        vehicle_capacities = [getattr(v, 'capacity', 100) for v in self.instance.vehicles.values()]
        
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            vehicle_capacities,
            True,  # start cumul to zero
            'Capacity'
        )
    
    def _add_pickup_delivery_constraints(self, manager, routing, location_to_index):
        """Add pickup and delivery constraints."""
        for request in self.instance.ride_requests:
            pickup_idx = location_to_index[request.pickup_location]
            delivery_idx = location_to_index[request.dropoff_location]
            
            pickup_node = manager.NodeToIndex(pickup_idx)
            delivery_node = manager.NodeToIndex(delivery_idx)
            
            routing.AddPickupAndDelivery(pickup_node, delivery_node)
            routing.solver().Add(
                routing.VehicleVar(pickup_node) == routing.VehicleVar(delivery_node)
            )
    
    def _add_time_window_constraints(self, manager, routing, location_list) -> bool:
        """Add time window constraints - THIS IS THE KEY MISSING FUNCTIONALITY!"""
        # Check if any locations have time windows
        locations_with_time_windows = []
        for location_id in location_list:
            location = self.instance.locations[location_id]
            if (hasattr(location, 'time_window_start') and 
                location.time_window_start is not None and
                hasattr(location, 'time_window_end') and
                location.time_window_end is not None):
                locations_with_time_windows.append(location_id)
        
        if not locations_with_time_windows:
            return False
        
        logger.info(f"Adding time windows for {len(locations_with_time_windows)} locations")
        
        # Create time callback (using distance as time proxy for simplicity)
        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            from_location_id = location_list[from_node]
            to_location_id = location_list[to_node]
            
            # Get travel distance
            distance = self.instance.get_distance(from_location_id, to_location_id)
            
            # Convert to travel time (assuming 1 distance unit = 1 time unit for simplicity)
            travel_time = int(distance)
            
            # Add service time at 'from' location
            from_location = self.instance.locations[from_location_id]
            service_time = getattr(from_location, 'service_time', 0)
            
            return travel_time + service_time
        
        time_callback_index = routing.RegisterTransitCallback(time_callback)
        
        # Add time dimension
        max_time = 1440  # 24 hours in minutes
        routing.AddDimension(
            time_callback_index,
            120,  # Allow 2 hours waiting time
            max_time,  # Maximum route duration
            False,  # Don't force start cumul to zero
            'Time'
        )
        
        time_dimension = routing.GetDimensionOrDie('Time')
        
        # Add time window constraints for each location
        for location_idx, location_id in enumerate(location_list):
            location = self.instance.locations[location_id]
            if (hasattr(location, 'time_window_start') and 
                location.time_window_start is not None):
                
                index = manager.NodeToIndex(location_idx)
                time_dimension.CumulVar(index).SetRange(
                    int(location.time_window_start),
                    int(location.time_window_end)
                )
                
                logger.debug(f"Set time window for {location_id}: "
                           f"[{location.time_window_start}, {location.time_window_end}]")
        
        return True
    
    def _solve_routing_model(self, routing):
        """Solve the routing model with appropriate search parameters."""
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.FromSeconds(30)  # Reasonable time limit
        
        return routing.SolveWithParameters(search_parameters)
    
    def _extract_solution(self, manager, routing, assignment, location_list, runtime) -> VRPResult:
        """Extract the solution from OR-Tools assignment."""
        routes = {}
        total_distance = 0
        vehicles_used = 0
        
        for vehicle_id in range(routing.vehicles()):
            index = routing.Start(vehicle_id)
            route = []
            route_distance = 0
            
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                location_id = location_list[node_index]
                route.append(location_id)
                
                if not routing.IsEnd(assignment.Value(routing.NextVar(index))):
                    route_distance += routing.GetArcCostForVehicle(
                        index, assignment.Value(routing.NextVar(index)), vehicle_id
                    )
                
                index = assignment.Value(routing.NextVar(index))
            
            if len(route) > 1:  # Vehicle was used
                vehicles_used += 1
                total_distance += route_distance
                routes[f"vehicle_{vehicle_id}"] = route
        
        return VRPResult(
            status="optimal",
            objective_value=total_distance,
            routes=routes,
            solution_time=runtime,
            total_distance=total_distance,
            metrics={
                "total_distance": total_distance,
                "vehicles_used": vehicles_used,
                "solution_method": "OR-Tools with time windows"
            },
            runtime=runtime
        )

def test_time_window_fix():
    """Test the enhanced optimizer with time windows."""
    from vrp_scenarios import create_moda_small_scenario
    
    print("Testing Enhanced VRP Optimizer with Time Windows")
    print("=" * 60)
    
    # Create MODA scenario
    scenario = create_moda_small_scenario()
    
    # Test with time windows
    print(f"\\n🔧 Testing: {scenario.name}")
    print(f"   Locations: {len(scenario.locations)}")
    print(f"   Vehicles: {len(scenario.vehicles)}")
    print(f"   Requests: {len(scenario.ride_requests)}")
    
    # Count locations with time windows
    tw_count = sum(1 for loc in scenario.locations.values() 
                   if hasattr(loc, 'time_window_start') and loc.time_window_start is not None)
    print(f"   Time windows: {tw_count}/{len(scenario.locations)} locations")
    
    # Solve with enhanced optimizer
    optimizer = VRPOptimizerWithTimeWindows(scenario)
    result = optimizer.solve()
    
    print(f"\\n📊 Results:")
    print(f"   Status: {result.status}")
    print(f"   Objective: {result.objective_value}")
    print(f"   Runtime: {result.runtime:.1f}ms")
    print(f"   Vehicles used: {len(result.routes)}")
    
    if result.routes:
        print(f"\\n🚛 Routes:")
        for vehicle, route in result.routes.items():
            print(f"   {vehicle}: {len(route)} stops")
        
        return True
    else:
        print("   ❌ No routes found")
        return False

if __name__ == "__main__":
    test_time_window_fix()
