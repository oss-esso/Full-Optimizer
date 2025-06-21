#!/usr/bin/env python3
"""
Robust VRP Optimizer with constraint validation and fallback strategies.
"""

import logging
from typing import Dict, Any

try:
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False

from vrp_data_models import VRPInstance

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VRPOptimizerRobust:
    """Robust VRP optimizer with constraint validation and progressive relaxation."""
    
    def solve(self, instance: VRPInstance, time_limit_seconds: int = 120) -> Dict[str, Any]:
        """Solve VRP with progressive constraint relaxation if needed."""
        logger.info("🛡️ ROBUST VRP OPTIMIZER WITH CONSTRAINT VALIDATION")
        logger.info("=" * 60)
        
        # Print sanity check
        self._print_sanity_check(instance)
        
        if not ORTOOLS_AVAILABLE:
            return {'success': False, 'error': 'OR-Tools not available'}
        
        # Try different constraint configurations
        constraint_configs = [
            {'name': 'Full Constraints', 'time_windows': True, 'capacity': True, 'pickup_delivery': True},
            {'name': 'No Time Windows', 'time_windows': False, 'capacity': True, 'pickup_delivery': True},
            {'name': 'Basic P&D Only', 'time_windows': False, 'capacity': False, 'pickup_delivery': True},
            {'name': 'Distance Only', 'time_windows': False, 'capacity': False, 'pickup_delivery': False},
        ]
        
        for config in constraint_configs:
            logger.info(f"\n🔧 Trying: {config['name']}")
            result = self._solve_with_config(instance, config, time_limit_seconds // len(constraint_configs))
            
            if result['success']:
                result['constraint_config'] = config
                return result
            else:
                logger.info(f"   ❌ Failed with {config['name']}: {result.get('error', 'Unknown')}")
        
        return {'success': False, 'error': 'Failed with all constraint configurations'}
    
    def _solve_with_config(self, instance: VRPInstance, config: Dict, time_limit: int) -> Dict[str, Any]:
        """Solve with specific constraint configuration."""
        try:
            num_locations = len(instance.locations)
            num_vehicles = len(instance.vehicles)
            
            manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, 0)
            routing = pywrapcp.RoutingModel(manager)
            location_list = list(instance.locations.values())
            
            # Distance callback
            def distance_callback(from_index, to_index):
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                from_loc = location_list[from_node]
                to_loc = location_list[to_node]
                dx = abs(from_loc.x - to_loc.x)
                dy = abs(from_loc.y - to_loc.y)
                return int((dx + dy) * 100)
            
            transit_callback_index = routing.RegisterTransitCallback(distance_callback)
            routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
            
            constraints_added = ['distance']
            
            # Add time dimension if needed
            if config['time_windows'] or config['capacity']:
                # Time callback with service times
                def time_callback(from_index, to_index):
                    from_node = manager.IndexToNode(from_index)
                    travel_time = distance_callback(from_index, to_index) // 100
                    service_time = getattr(location_list[from_node], 'service_time', 0)
                    return int(travel_time + service_time)
                
                time_callback_index = routing.RegisterTransitCallback(time_callback)
                
                # Get max vehicle time
                max_time = 600  # Default
                for vehicle in instance.vehicles.values():
                    vehicle_max = (getattr(vehicle, 'max_total_work_time', None) or 
                                 getattr(vehicle, 'max_time', None) or 600)
                    max_time = max(max_time, int(vehicle_max))
                
                routing.AddDimension(
                    time_callback_index,
                    60,  # Slack
                    max_time + 100,  # Add buffer to max time
                    False,
                    'Time'
                )
                time_dimension = routing.GetDimensionOrDie('Time')
                constraints_added.append('time_dimension')
                
                # Add individual vehicle time limits
                for vehicle_id, vehicle in enumerate(instance.vehicles.values()):
                    vehicle_max_time = (getattr(vehicle, 'max_total_work_time', None) or 
                                      getattr(vehicle, 'max_time', None) or 600)
                    end_index = routing.End(vehicle_id)
                    time_dimension.CumulVar(end_index).SetMax(int(vehicle_max_time))
                
                constraints_added.append('vehicle_time_limits')
            
            # Add time windows with validation
            if config['time_windows']:
                time_windows_added = 0
                time_windows_failed = 0
                
                for location_idx, location in enumerate(location_list):
                    if hasattr(location, 'time_window_start') and location.time_window_start is not None:
                        try:
                            index = manager.NodeToIndex(location_idx)
                            start = int(location.time_window_start)
                            end = int(location.time_window_end)
                            
                            # Validate time window
                            if start >= 0 and end > start and end <= 1440:
                                time_dimension.CumulVar(index).SetRange(start, end)
                                time_windows_added += 1
                            else:
                                time_windows_failed += 1
                                logger.warning(f"Invalid time window for {location.id}: [{start}, {end}]")
                                
                        except Exception as e:
                            time_windows_failed += 1
                            logger.warning(f"Failed to add time window for {location.id}: {e}")
                
                logger.info(f"   Time windows: {time_windows_added} added, {time_windows_failed} failed")
                constraints_added.append(f'time_windows_{time_windows_added}')
            
            # Add capacity constraints
            if config['capacity'] and instance.ride_requests:
                def demand_callback(from_index):
                    from_node = manager.IndexToNode(from_index)
                    location = location_list[from_node]
                    demand = 0
                    for req in instance.ride_requests:
                        if req.pickup_location == location.id:
                            demand += int(req.passengers)
                        elif req.dropoff_location == location.id:
                            demand -= int(req.passengers)
                    return demand
                
                demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
                vehicle_capacities = [int(v.capacity) for v in instance.vehicles.values()]
                
                routing.AddDimensionWithVehicleCapacity(
                    demand_callback_index, 0, vehicle_capacities, True, 'Capacity'
                )
                constraints_added.append('capacity')
            
            # Add pickup-delivery constraints
            pickup_delivery_pairs = 0
            if config['pickup_delivery'] and instance.ride_requests:
                location_ids = [loc.id for loc in location_list]
                
                for req in instance.ride_requests:
                    try:
                        pickup_idx = location_ids.index(req.pickup_location)
                        dropoff_idx = location_ids.index(req.dropoff_location)
                        
                        pickup_index = manager.NodeToIndex(pickup_idx)
                        dropoff_index = manager.NodeToIndex(dropoff_idx)
                        
                        routing.AddPickupAndDelivery(pickup_index, dropoff_index)
                        routing.solver().Add(
                            routing.VehicleVar(pickup_index) == routing.VehicleVar(dropoff_index)
                        )
                        
                        # Add time ordering if time dimension exists
                        if config['time_windows'] and 'Time' in [routing.GetDimensionName(i) for i in range(routing.GetNumberOfDimensions())]:
                            routing.solver().Add(
                                time_dimension.CumulVar(pickup_index) <= time_dimension.CumulVar(dropoff_index)
                            )
                        
                        pickup_delivery_pairs += 1
                        
                    except (ValueError, Exception) as e:
                        logger.warning(f"Skipping request {req.id}: {e}")
                
                constraints_added.append(f'pickup_delivery_{pickup_delivery_pairs}')
            
            logger.info(f"   Constraints added: {constraints_added}")
            
            # Solve
            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
            search_parameters.time_limit.FromSeconds(time_limit)
            search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC
            
            assignment = routing.SolveWithParameters(search_parameters)
            
            if assignment:
                try:
                    objective_value = assignment.ObjectiveValue()
                except AttributeError:
                    objective_value = routing.GetCost(assignment)
                
                # Extract routes
                routes = {}
                vehicles_used = 0
                total_time_used = 0
                
                for vehicle_id in range(routing.vehicles()):
                    route = []
                    index = routing.Start(vehicle_id)
                    
                    while not routing.IsEnd(index):
                        node_index = manager.IndexToNode(index)
                        route.append(location_list[node_index].id)
                        index = assignment.Value(routing.NextVar(index))
                    
                    if len(route) > 1:
                        routes[f"vehicle_{vehicle_id}"] = route
                        vehicles_used += 1
                        
                        # Get route time if time dimension exists
                        if 'Time' in [routing.GetDimensionName(i) for i in range(routing.GetNumberOfDimensions())]:
                            end_time_var = time_dimension.CumulVar(routing.End(vehicle_id))
                            route_time = assignment.Value(end_time_var)
                            total_time_used += route_time
                            
                            vehicle = list(instance.vehicles.values())[vehicle_id]
                            max_time = getattr(vehicle, 'max_total_work_time', 600)
                            logger.info(f"   Vehicle {vehicle_id}: {len(route)} stops, {route_time}min ({route_time/max_time*100:.1f}% of {max_time}min limit)")
                
                logger.info(f"✅ SUCCESS! Objective: {objective_value}, Vehicles: {vehicles_used}/{num_vehicles}")
                
                return {
                    'success': True,
                    'objective_value': objective_value,
                    'routes': routes,
                    'vehicles_used': vehicles_used,
                    'total_vehicles': num_vehicles,
                    'constraints_added': constraints_added,
                    'pickup_delivery_pairs': pickup_delivery_pairs
                }
            else:
                return {'success': False, 'error': 'No solution found'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _print_sanity_check(self, instance: VRPInstance):
        """Print key statistics."""
        logger.info("📊 INSTANCE SUMMARY:")
        
        # Basic stats
        num_locations = len(instance.locations)
        num_vehicles = len(instance.vehicles)
        num_requests = len(instance.ride_requests) if instance.ride_requests else 0
        
        logger.info(f"   📍 {num_locations} locations, 🚛 {num_vehicles} vehicles, 📦 {num_requests} requests")
        
        # Vehicle capacities and time limits
        capacities = [v.capacity for v in instance.vehicles.values()]
        time_limits = []
        for v in instance.vehicles.values():
            time_limit = getattr(v, 'max_total_work_time', None) or getattr(v, 'max_time', None) or 600
            time_limits.append(time_limit)
        
        logger.info(f"   🚛 Capacities: {capacities} kg")
        logger.info(f"   ⏱️ Time limits: {time_limits} min")
        
        # Total demand
        if instance.ride_requests:
            total_demand = sum(req.passengers for req in instance.ride_requests)
            total_capacity = sum(capacities)
            logger.info(f"   📊 Total demand: {total_demand}kg vs capacity: {total_capacity}kg ({total_demand/total_capacity*100:.1f}%)")
        
        # Time windows
        time_windowed = sum(1 for loc in instance.locations.values() 
                           if hasattr(loc, 'time_window_start') and loc.time_window_start is not None)
        logger.info(f"   ⏰ {time_windowed}/{num_locations} locations have time windows")

def test_robust_optimizer():
    """Test the robust optimizer."""
    from vrp_scenarios import create_moda_small_scenario
    
    print("Testing Robust VRP Optimizer")
    print("=" * 50)
    
    try:
        scenario = create_moda_small_scenario()
        optimizer = VRPOptimizerRobust()
        
        result = optimizer.solve(scenario, time_limit_seconds=180)
        
        print(f"\n🎯 FINAL RESULTS:")
        print(f"Success: {result['success']}")
        
        if result['success']:
            print(f"✅ SOLVED!")
            print(f"   Configuration: {result.get('constraint_config', {}).get('name', 'Unknown')}")
            print(f"   Objective: {result['objective_value']}")
            print(f"   Vehicles used: {result['vehicles_used']}/{result['total_vehicles']}")
            print(f"   Constraints: {result['constraints_added']}")
            print(f"   Pickup-delivery pairs: {result.get('pickup_delivery_pairs', 0)}")
            
            print(f"\n📍 ROUTES:")
            for vehicle, route in result['routes'].items():
                print(f"   {vehicle}: {len(route)} stops -> {' -> '.join(route[:3])}{'...' if len(route) > 3 else ''}")
                
            # Check if we're using multiple vehicles appropriately
            if result['vehicles_used'] == 1 and result['total_vehicles'] > 1:
                print(f"\n⚠️ ANALYSIS: Only 1 vehicle used out of {result['total_vehicles']}.")
                print(f"   This suggests missing constraints (time limits, capacity, etc.)")
                print(f"   Configuration used: {result.get('constraint_config', {})}")
        else:
            print(f"❌ Failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_robust_optimizer()
