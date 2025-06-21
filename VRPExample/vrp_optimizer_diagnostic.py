#!/usr/bin/env python3
"""
Enhanced VRP Optimizer with Detailed Diagnostics

This version provides comprehensive diagnostic information to help identify 
the specific bottlenecks causing infeasibility in complex VRP scenarios.
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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VRPOptimizerDiagnostic:
    """Enhanced VRP optimizer with detailed diagnostic capabilities."""
    
    def __init__(self):
        """Initialize the diagnostic VRP optimizer."""
        self.diagnostic_info = {}
        
    def solve(self, instance: VRPInstance, time_limit_seconds: int = 300) -> Dict[str, Any]:
        """
        Solve VRP with comprehensive diagnostics.
        
        Returns detailed information about solver performance and constraints.
        """
        logger.info("=" * 80)
        logger.info("STARTING DIAGNOSTIC VRP OPTIMIZATION")
        logger.info("=" * 80)
        start_time = time.time()

        self._validate_instance(instance)
        
        # Print comprehensive sanity check
        self._print_comprehensive_sanity_check(instance)
        
        # Create routing model
        model_result = self._create_routing_model_with_diagnostics(instance)
        
        if not model_result['success']:
            return model_result
        
        manager = model_result['manager']
        routing = model_result['routing']
        
        # Add constraints
        constraints_result = self._add_constraints_with_diagnostics(instance, manager, routing)
        
        if not constraints_result['success']:
            return constraints_result
            
        # Solve with detailed monitoring
        solution_result = self._solve_with_diagnostics(instance, manager, routing, time_limit_seconds)
        
        total_time = time.time() - start_time
        solution_result['total_solve_time'] = total_time
        solution_result['diagnostic_info'] = self.diagnostic_info
        
        logger.info(f"Total optimization time: {total_time:.2f} seconds")
        logger.info("=" * 80)
        
        return solution_result

    def _create_routing_model_with_diagnostics(self, instance: VRPInstance) -> Dict[str, Any]:
        """Create OR-Tools routing model with diagnostic information."""
        logger.info("CREATING ROUTING MODEL")
        logger.info("-" * 40)
        
        if not ORTOOLS_AVAILABLE:
            return {'success': False, 'error': 'OR-Tools not available'}
        
        try:
            # Create routing model - use single depot approach for simplicity
            num_locations = len(instance.locations)
            num_vehicles = len(instance.vehicles)
            
            # Use first location as single depot for all vehicles
            depot_index = 0
            
            logger.info(f"Creating model with {num_locations} locations, {num_vehicles} vehicles, depot at index {depot_index}")
            
            manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, depot_index)
            routing = pywrapcp.RoutingModel(manager)
            
            self.diagnostic_info['model_creation'] = {
                'num_locations': num_locations,
                'num_vehicles': num_vehicles,
                'depot_index': depot_index,
                'success': True
            }
            
            return {
                'success': True,
                'manager': manager,
                'routing': routing
            }
            
        except Exception as e:
            error_msg = f"Failed to create routing model: {str(e)}"
            logger.error(error_msg)            
            
            return {'success': False, 'error': error_msg}
    
    def _add_constraints_with_diagnostics(self, instance: VRPInstance, manager, routing) -> Dict[str, Any]:
        """Add constraints with diagnostic monitoring."""
        logger.info("ADDING CONSTRAINTS")
        logger.info("-" * 40)
        
        constraints_added = []
        location_list = list(instance.locations.values())
        
        try:
            # Add distance callback with proper scaling
            def distance_callback(from_index, to_index):
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                
                from_loc = location_list[from_node]
                to_loc = location_list[to_node]
                
                if hasattr(instance, 'distance_matrix') and instance.distance_matrix is not None:
                    return int(instance.distance_matrix[from_node][to_node])
                else:
                    # Calculate Manhattan distance with proper scaling
                    dx = abs(from_loc.x - to_loc.x)
                    dy = abs(from_loc.y - to_loc.y)
                    return int((dx + dy) * 100)  # Scale for integer precision
            
            transit_callback_index = routing.RegisterTransitCallback(distance_callback)
            routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
            constraints_added.append('distance_costs')
              # Add time dimension with proper scaling
            def time_callback(from_index, to_index):
                # Convert distance to time + add service time
                base_time = distance_callback(from_index, to_index) // 100  # Convert back to minutes
                
                # Add service time for the destination location
                to_node = manager.IndexToNode(to_index)
                to_location = location_list[to_node]
                service_time = getattr(to_location, 'service_time', 0)
                
                return base_time + service_time
            
            time_callback_index = routing.RegisterTransitCallback(time_callback)
            
            # Add time dimension with vehicle work hour limits
            max_vehicle_time = 1440  # Default 24 hours
            if instance.vehicles:
                # Use the maximum work time from vehicles
                vehicle_times = []
                for vehicle in instance.vehicles.values():
                    max_time = getattr(vehicle, 'max_total_work_time', 600)  # Default 10 hours
                    vehicle_times.append(max_time)
                max_vehicle_time = max(vehicle_times)
            
            routing.AddDimension(
                time_callback_index,
                60,   # Allow 1 hour waiting time
                max_vehicle_time, # Use actual vehicle time limits
                False, # Don't force start cumul to zero
                'Time'
            )
            time_dimension = routing.GetDimensionOrDie('Time')
            constraints_added.append('time_dimension_with_service_time')
            
            # Add individual vehicle time limits
            if instance.vehicles:
                vehicles_with_limits = 0
                for vehicle_idx, vehicle in enumerate(instance.vehicles.values()):
                    max_time = getattr(vehicle, 'max_total_work_time', 600)
                    vehicle_start = routing.Start(vehicle_idx)
                    vehicle_end = routing.End(vehicle_idx)
                    time_dimension.CumulVar(vehicle_end).SetMax(max_time)
                    vehicles_with_limits += 1
                
                if vehicles_with_limits > 0:
                    constraints_added.append(f'vehicle_time_limits_{vehicles_with_limits}')
            
            # Add time window constraints for locations
            time_windows_count = 0
            for location_idx, location in enumerate(location_list):
                if hasattr(location, 'time_window_start') and location.time_window_start is not None:
                    index = manager.NodeToIndex(location_idx)
                    time_dimension.CumulVar(index).SetRange(
                        int(location.time_window_start),
                        int(location.time_window_end)
                    )
                    time_windows_count += 1
            
            if time_windows_count > 0:
                constraints_added.append(f'time_windows_{time_windows_count}')
            
            # Add capacity constraints with proper demand calculation
            if instance.ride_requests:
                def demand_callback(from_index):
                    from_node = manager.IndexToNode(from_index)
                    location = location_list[from_node]
                    
                    # Calculate demand based on ride requests
                    demand = 0
                    for req in instance.ride_requests:
                        if req.pickup_location == location.id:
                            demand += int(req.passengers)  # Pick up cargo
                        elif req.dropoff_location == location.id:
                            demand -= int(req.passengers)  # Drop off cargo
                    
                    return demand
                
                demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
                
                # Get vehicle capacities
                vehicle_capacities = []
                for vehicle in instance.vehicles.values():
                    vehicle_capacities.append(int(vehicle.capacity))
                
                routing.AddDimensionWithVehicleCapacity(
                    demand_callback_index,
                    0,  # No slack
                    vehicle_capacities,
                    True,  # Start cumul at zero
                    'Capacity'
                )
                
                constraints_added.append('capacity_constraints')
            
            # Add pickup and delivery constraints
            if instance.ride_requests:
                pickup_delivery_pairs = 0
                location_ids = [loc.id for loc in location_list]
                
                for req in instance.ride_requests:
                    try:
                        pickup_idx = location_ids.index(req.pickup_location)
                        dropoff_idx = location_ids.index(req.dropoff_location)
                        
                        pickup_index = manager.NodeToIndex(pickup_idx)
                        dropoff_index = manager.NodeToIndex(dropoff_idx)
                        
                        routing.AddPickupAndDelivery(pickup_index, dropoff_index)
                        
                        # Same vehicle constraint
                        routing.solver().Add(
                            routing.VehicleVar(pickup_index) == routing.VehicleVar(dropoff_index)
                        )
                        
                        # Pickup before dropoff constraint
                        time_dimension.CumulVar(pickup_index).SetRange(0, 1440)
                        time_dimension.CumulVar(dropoff_index).SetRange(0, 1440)
                        routing.solver().Add(
                            time_dimension.CumulVar(pickup_index) <= time_dimension.CumulVar(dropoff_index)
                        )
                        
                        pickup_delivery_pairs += 1
                        
                    except ValueError as e:
                        logger.warning(f"Could not find location for request {req.id}: {e}")
                
                if pickup_delivery_pairs > 0:
                    constraints_added.append(f'pickup_delivery_{pickup_delivery_pairs}')
            
            logger.info(f"Successfully added constraints: {constraints_added}")
            
            self.diagnostic_info['constraints'] = {
                'added': constraints_added,
                'success': True
            }
            
            return {'success': True}
            
        except Exception as e:
            error_msg = f"Failed to add constraints: {str(e)}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
                
            # Solve with detailed monitoring
            solution_result = self._solve_with_diagnostics(instance, manager, routing, time_limit_seconds)
            
            total_time = time.time() - start_time
            solution_result['total_solve_time'] = total_time
            solution_result['diagnostic_info'] = self.diagnostic_info
            
            logger.info(f"Total optimization time: {total_time:.2f} seconds")
            logger.info("=" * 80)
            
            return solution_result
            
        except Exception as e:
            logger.error(f"Optimization failed with exception: {str(e)}")
            return {
                'success': False,
                'error': f'Exception during optimization: {str(e)}',
                'diagnostic_info': self.diagnostic_info
            }
    
    def _validate_instance(self, instance: VRPInstance) -> None:
        """Validate the VRP instance and log diagnostic information."""
        logger.info("INSTANCE VALIDATION")
        logger.info("-" * 40)
        
        num_locations = len(instance.locations)
        num_vehicles = len(instance.vehicles)
        num_requests = len(instance.ride_requests) if hasattr(instance, 'ride_requests') else 0
        
        logger.info(f"Locations: {num_locations}")
        logger.info(f"Vehicles: {num_vehicles}")
        logger.info(f"Ride requests: {num_requests}")
        
        # Analyze vehicle capacities and constraints
        vehicle_analysis = self._analyze_vehicles(instance)
        logger.info(f"Vehicle analysis: {vehicle_analysis}")
        
        # Analyze time windows
        time_window_analysis = self._analyze_time_windows(instance)
        logger.info(f"Time window analysis: {time_window_analysis}")
        
        # Analyze distances
        distance_analysis = self._analyze_distances(instance)
        logger.info(f"Distance analysis: {distance_analysis}")
        
        self.diagnostic_info['instance_validation'] = {
            'num_locations': num_locations,
            'num_vehicles': num_vehicles,
            'num_requests': num_requests,
            'vehicle_analysis': vehicle_analysis,
            'time_window_analysis': time_window_analysis,
            'distance_analysis': distance_analysis
        }
    
    def _analyze_vehicles(self, instance: VRPInstance) -> Dict[str, Any]:
        """Analyze vehicle constraints and capabilities."""
        vehicles = list(instance.vehicles.values())
        
        analysis = {
            'total_capacity': sum(v.capacity for v in vehicles),
            'min_capacity': min(v.capacity for v in vehicles),
            'max_capacity': max(v.capacity for v in vehicles),
            'avg_capacity': sum(v.capacity for v in vehicles) / len(vehicles),
            'capacity_distribution': {},
            'time_constraints': {}
        }
        
        # Capacity distribution
        for v in vehicles:
            cap = v.capacity
            analysis['capacity_distribution'][cap] = analysis['capacity_distribution'].get(cap, 0) + 1
        
        # Time constraint analysis
        max_times = [getattr(v, 'max_total_work_time', 600) for v in vehicles]
        analysis['time_constraints'] = {
            'min_work_time': min(max_times),
            'max_work_time': max(max_times),
            'avg_work_time': sum(max_times) / len(max_times)
        }
        
        return analysis
    
    def _analyze_time_windows(self, instance: VRPInstance) -> Dict[str, Any]:
        """Analyze time window constraints."""
        locations = list(instance.locations.values())
        
        time_constrained = [loc for loc in locations 
                           if hasattr(loc, 'time_window_start') and loc.time_window_start is not None]
        
        if not time_constrained:
            return {'has_time_windows': False}
        
        starts = [loc.time_window_start for loc in time_constrained]
        ends = [loc.time_window_end for loc in time_constrained]
        spans = [end - start for start, end in zip(starts, ends)]
        service_times = [getattr(loc, 'service_time', 0) for loc in time_constrained]
        
        analysis = {
            'has_time_windows': True,
            'num_time_constrained': len(time_constrained),
            'earliest_start': min(starts),
            'latest_end': max(ends),
            'total_time_span': max(ends) - min(starts),
            'window_spans': {
                'min': min(spans),
                'max': max(spans),
                'avg': sum(spans) / len(spans)
            },
            'service_times': {
                'min': min(service_times),
                'max': max(service_times),
                'avg': sum(service_times) / len(service_times),
                'total': sum(service_times)
            }
        }
        
        return analysis
    
    def _analyze_distances(self, instance: VRPInstance) -> Dict[str, Any]:
        """Analyze distance matrix for routing feasibility."""
        if not hasattr(instance, 'distance_matrix') or instance.distance_matrix is None:
            return {'has_distances': False}
        
        matrix = instance.distance_matrix
        n = len(matrix)
        
        # Flatten matrix (excluding diagonal)
        distances = [matrix[i][j] for i in range(n) for j in range(n) if i != j]
        
        analysis = {
            'has_distances': True,
            'matrix_size': n,
            'min_distance': min(distances),
            'max_distance': max(distances),
            'avg_distance': sum(distances) / len(distances),
            'zero_distances': sum(1 for d in distances if d == 0)
        }
        
        return analysis
    
    def _solve_with_diagnostics(self, instance: VRPInstance, manager, routing, time_limit_seconds: int) -> Dict[str, Any]:
        """Solve with detailed monitoring and diagnostics."""
        logger.info("SOLVING WITH OR-TOOLS")
        logger.info("-" * 40)
        
        # Set up search parameters with multiple strategies
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.time_limit.FromSeconds(time_limit_seconds)
        search_parameters.log_search = True  # Enable detailed logging
        
        # Try multiple solution strategies
        strategies = [
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
            routing_enums_pb2.FirstSolutionStrategy.PATH_MOST_CONSTRAINED_ARC,
            routing_enums_pb2.FirstSolutionStrategy.EVALUATOR_STRATEGY,
            routing_enums_pb2.FirstSolutionStrategy.SAVINGS,
            routing_enums_pb2.FirstSolutionStrategy.SWEEP,
        ]
        
        results = []
        
        for i, strategy in enumerate(strategies):
            logger.info(f"Trying strategy {i+1}/{len(strategies)}: {strategy}")
            
            search_parameters.first_solution_strategy = strategy
            
            solve_start = time.time()
            assignment = routing.SolveWithParameters(search_parameters)
            solve_time = time.time() - solve_start
            
            strategy_result = {
                'strategy': strategy,
                'solve_time': solve_time,
                'found_solution': assignment is not None
            }
            
            if assignment:
                # Get solution statistics
                stats = self._get_solution_statistics(instance, manager, routing, assignment)
                strategy_result.update(stats)                
                logger.info(f"✅ Solution found with strategy {strategy}")
                logger.info(f"   Total distance: {stats['total_distance']}")
                logger.info(f"   Total time: {stats['total_time']}")
                logger.info(f"   Vehicles used: {stats['vehicles_used']}")
                
                # Convert to VRP result format
                vrp_result = VRPResult(
                    objective_value=stats['total_distance'],
                    routes={f"vehicle_{i}": route_info['route'] for i, route_info in enumerate(stats['route_statistics'])},
                    solution_time=solve_time,
                    total_distance=stats['total_distance']
                )
                
                return {
                    'success': True,
                    'solution': vrp_result,
                    'strategy_used': strategy,
                    'solve_time': solve_time,
                    'statistics': stats,
                    'all_strategies': results + [strategy_result]
                }
            else:
                logger.info(f"❌ No solution found with strategy {strategy}")
                
                # Try to get more specific failure information
                failure_info = self._analyze_failure(routing)
                strategy_result['failure_info'] = failure_info
                
            results.append(strategy_result)
        
        logger.warning("❌ No solution found with any strategy")
        
        # Perform detailed infeasibility analysis
        infeasibility_analysis = self._analyze_infeasibility(instance, manager, routing)
        
        return {
            'success': False,
            'error': 'No feasible solution found with any strategy',
            'all_strategies': results,
            'infeasibility_analysis': infeasibility_analysis
        }
    
    def _get_solution_statistics(self, instance: VRPInstance, manager, routing, assignment) -> Dict[str, Any]:
        """Extract detailed statistics from a solution."""
        total_distance = 0
        total_time = 0
        vehicles_used = 0
        route_stats = []
        
        for vehicle_id in range(routing.vehicles()):
            index = routing.Start(vehicle_id)
            route_distance = 0
            route_time = 0
            route_load = 0
            stops = 0
            
            route = []
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route.append(node_index)
                stops += 1
                
                if not routing.IsEnd(assignment.Value(routing.NextVar(index))):
                    route_distance += routing.GetArcCostForVehicle(
                        index, assignment.Value(routing.NextVar(index)), vehicle_id)
                
                index = assignment.Value(routing.NextVar(index))
            
            if stops > 1:  # Vehicle was used (more than just depot)
                vehicles_used += 1
                total_distance += route_distance
                
                route_stats.append({
                    'vehicle_id': vehicle_id,
                    'stops': stops,
                    'distance': route_distance,
                    'route': route
                })
        
        return {
            'total_distance': total_distance,
            'total_time': total_time,
            'vehicles_used': vehicles_used,
            'route_statistics': route_stats
        }
    
    def _analyze_failure(self, routing) -> Dict[str, Any]:
        """Analyze why OR-Tools failed to find a solution."""
        # This is limited by OR-Tools API, but we can gather some info
        return {
            'vehicles': routing.vehicles(),
            'nodes': routing.nodes(),
            'message': 'OR-Tools failed to find feasible solution'
        }
    
    def _analyze_infeasibility(self, instance: VRPInstance, manager, routing) -> Dict[str, Any]:
        """Perform detailed analysis of why the problem might be infeasible."""
        logger.info("INFEASIBILITY ANALYSIS")
        logger.info("-" * 40)
        
        analysis = {}
        
        # Check basic feasibility
        num_requests = len(instance.ride_requests) if hasattr(instance, 'ride_requests') else 0
        num_vehicles = len(instance.vehicles)
        
        # Vehicle capacity analysis
        total_demand = sum(getattr(req, 'passengers', 1) for req in instance.ride_requests) if hasattr(instance, 'ride_requests') else 0
        total_capacity = sum(v.capacity for v in instance.vehicles.values())
        
        analysis['capacity_check'] = {
            'total_demand': total_demand,
            'total_capacity': total_capacity,
            'feasible': total_demand <= total_capacity,
            'utilization': total_demand / total_capacity if total_capacity > 0 else float('inf')
        }
        
        # Time window tightness analysis
        if hasattr(instance, 'ride_requests'):
            pickup_dropoff_pairs = []
            for req in instance.ride_requests:
                pickup_loc = instance.locations[req.pickup_location]
                dropoff_loc = instance.locations[req.dropoff_location]
                pickup_dropoff_pairs.append((pickup_loc, dropoff_loc))
            
            tight_windows = 0
            impossible_windows = 0
            
            for pickup, dropoff in pickup_dropoff_pairs:
                if (hasattr(pickup, 'time_window_end') and hasattr(dropoff, 'time_window_start') and
                    pickup.time_window_end is not None and dropoff.time_window_start is not None):
                    
                    # Calculate minimum travel time needed
                    pickup_idx = list(instance.locations.keys()).index(pickup.id)
                    dropoff_idx = list(instance.locations.keys()).index(dropoff.id)
                    
                    if hasattr(instance, 'distance_matrix') and instance.distance_matrix:
                        travel_distance = instance.distance_matrix[pickup_idx][dropoff_idx]
                        min_travel_time = travel_distance  # Assuming 1 km = 1 minute for simplicity
                        min_required_time = min_travel_time + getattr(pickup, 'service_time', 0)
                        
                        available_time = dropoff.time_window_start - pickup.time_window_end
                        
                        if available_time < min_required_time:
                            if available_time < 0:
                                impossible_windows += 1
                            else:
                                tight_windows += 1
            
            analysis['time_window_check'] = {
                'total_pairs': len(pickup_dropoff_pairs),
                'tight_windows': tight_windows,
                'impossible_windows': impossible_windows,
                'feasible_time_windows': impossible_windows == 0
            }
        
        # Route duration analysis
        vehicle_work_times = []
        for vehicle in instance.vehicles.values():
            max_work_time = getattr(vehicle, 'max_total_work_time', 600)  # Default 10 hours
            vehicle_work_times.append(max_work_time)
        
        analysis['time_constraint_check'] = {
            'min_work_time': min(vehicle_work_times),
            'max_work_time': max(vehicle_work_times),
            'avg_work_time': sum(vehicle_work_times) / len(vehicle_work_times)
        }
        
        # Summary
        feasibility_issues = []
        if not analysis['capacity_check']['feasible']:
            feasibility_issues.append('insufficient_capacity')
        if 'time_window_check' in analysis and not analysis['time_window_check']['feasible_time_windows']:
            feasibility_issues.append('impossible_time_windows')
        if 'time_window_check' in analysis and analysis['time_window_check']['tight_windows'] > 0:
            feasibility_issues.append('tight_time_windows')
        
        analysis['summary'] = {
            'likely_feasible': len(feasibility_issues) == 0,
            'identified_issues': feasibility_issues,
            'recommendations': self._generate_recommendations(feasibility_issues, analysis)
        }
        
        # Log summary
        if analysis['summary']['likely_feasible']:
            logger.info("✅ Basic feasibility checks passed - issue may be with OR-Tools search")
        else:
            logger.warning(f"❌ Feasibility issues identified: {feasibility_issues}")
            for rec in analysis['summary']['recommendations']:
                logger.info(f"💡 Recommendation: {rec}")
        
        return analysis
    
    def _generate_recommendations(self, issues: List[str], analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on identified issues."""
        recommendations = []
        
        if 'insufficient_capacity' in issues:
            recommendations.append("Increase vehicle capacity or add more vehicles")
        
        if 'impossible_time_windows' in issues:
            recommendations.append("Relax time window constraints or adjust pickup/dropoff timing")
        
        if 'tight_time_windows' in issues:
            recommendations.append("Consider expanding time windows or reducing service times")
        
        if not issues:
            recommendations.extend([
                "Try increasing solution time limit",
                "Consider using different OR-Tools search strategies",
                "Check for isolated location clusters that may be unreachable",
                "Verify distance matrix accuracy"
            ])
        
        return recommendations
    
    def _print_comprehensive_sanity_check(self, instance: VRPInstance) -> None:
        """Print comprehensive sanity check of all constraints and features."""
        logger.info("🔍 COMPREHENSIVE SANITY CHECK")
        logger.info("=" * 60)
        
        # Vehicle analysis
        logger.info("🚛 VEHICLE ANALYSIS")
        total_capacity = 0
        for vehicle_id, vehicle in instance.vehicles.items():
            capacity = vehicle.capacity
            max_time = getattr(vehicle, 'max_total_work_time', 600)
            vehicle_type = getattr(vehicle, 'vehicle_type', 'unknown')
            depot = vehicle.depot_id
            
            logger.info(f"   {vehicle_id}: {capacity}kg capacity, {max_time}min max_time, type={vehicle_type}, depot={depot}")
            total_capacity += capacity
        
        logger.info(f"   Total fleet capacity: {total_capacity:,}kg")
        
        # Demand analysis
        logger.info("\n📦 DEMAND ANALYSIS")
        total_demand = 0
        max_shipment = 0
        if instance.ride_requests:
            for req in instance.ride_requests:
                weight = req.passengers
                total_demand += weight
                max_shipment = max(max_shipment, weight)
                logger.info(f"   {req.id}: {weight}kg ({req.pickup_location} → {req.dropoff_location})")
        
        utilization = (total_demand / total_capacity * 100) if total_capacity > 0 else 0
        logger.info(f"   Total demand: {total_demand:,}kg")
        logger.info(f"   Largest shipment: {max_shipment:,}kg")
        logger.info(f"   Fleet utilization: {utilization:.1f}%")
        
        # Time window analysis
        logger.info("\n⏰ TIME WINDOW ANALYSIS")
        depot_windows = []
        pickup_windows = []
        dropoff_windows = []
        service_areas = []
        
        for loc_id, location in instance.locations.items():
            if hasattr(location, 'time_window_start') and location.time_window_start is not None:
                start = location.time_window_start
                end = location.time_window_end
                window_size = end - start
                service_time = getattr(location, 'service_time', 0)
                
                info = f"[{start}-{end}] ({window_size}min window) +{service_time}min service"
                
                if 'depot' in loc_id:
                    depot_windows.append((loc_id, info))
                elif 'pickup' in loc_id:
                    pickup_windows.append((loc_id, info))
                elif 'dropoff' in loc_id:
                    dropoff_windows.append((loc_id, info))
                else:
                    service_areas.append((loc_id, info))
        
        logger.info(f"   DEPOTS ({len(depot_windows)}):")
        for loc_id, info in depot_windows:
            logger.info(f"    {loc_id}: {info}")
        
        logger.info(f"   PICKUPS ({len(pickup_windows)}):")
        for loc_id, info in pickup_windows[:3]:  # Show first 3
            logger.info(f"    {loc_id}: {info}")
        if len(pickup_windows) > 3:
            logger.info(f"    ... and {len(pickup_windows) - 3} more pickup locations")
            
        logger.info(f"   DROPOFFS ({len(dropoff_windows)}):")
        for loc_id, info in dropoff_windows[:3]:  # Show first 3
            logger.info(f"    {loc_id}: {info}")
        if len(dropoff_windows) > 3:
            logger.info(f"    ... and {len(dropoff_windows) - 3} more dropoff locations")
            
        if service_areas:
            logger.info(f"   SERVICE AREAS ({len(service_areas)}):")
            for loc_id, info in service_areas:
                logger.info(f"    {loc_id}: {info}")
        
        # Pickup-Dropoff feasibility check
        logger.info("\n🔄 PICKUP-DROPOFF FEASIBILITY")
        if instance.ride_requests:
            feasible_pairs = 0
            infeasible_pairs = 0
            
            for req in instance.ride_requests:
                pickup_loc = instance.locations.get(req.pickup_location)
                dropoff_loc = instance.locations.get(req.dropoff_location)
                
                if pickup_loc and dropoff_loc:
                    pickup_end = getattr(pickup_loc, 'time_window_end', None)
                    dropoff_start = getattr(dropoff_loc, 'time_window_start', None)
                    
                    if pickup_end is not None and dropoff_start is not None:
                        gap = dropoff_start - pickup_end
                        if gap >= 0:
                            feasible_pairs += 1
                        else:
                            infeasible_pairs += 1
                            logger.info(f"   ❌ {req.id}: pickup ends {pickup_end}, dropoff starts {dropoff_start} (gap: {gap}min)")
                    else:
                        feasible_pairs += 1  # No time constraints
            
            logger.info(f"   Feasible pairs: {feasible_pairs}")
            logger.info(f"   Infeasible pairs: {infeasible_pairs}")
        
        # Distance matrix info
        logger.info("\n📏 DISTANCE MATRIX")
        if hasattr(instance, 'distance_matrix') and instance.distance_matrix is not None:
            matrix = instance.distance_matrix
            n = len(matrix)
            distances = [matrix[i][j] for i in range(n) for j in range(n) if i != j]
            logger.info(f"   Matrix size: {n}x{n}")
            logger.info(f"   Distance range: {min(distances)} - {max(distances)}")
            logger.info(f"   Average distance: {sum(distances)/len(distances):.1f}")
        else:
            logger.info("   No distance matrix - will calculate Manhattan distances")
        
        logger.info("=" * 60)
        logger.info("")  # Extra space after sanity check

if __name__ == "__main__":
    # Test the diagnostic optimizer
    from vrp_scenarios import create_moda_first_scenario
    
    print("Testing VRP Diagnostic Optimizer")
    print("=" * 50)
    
    # Create test scenario
    scenario = create_moda_first_scenario()
    
    # Initialize optimizer
    optimizer = VRPOptimizerDiagnostic()
    
    # Solve with diagnostics
    result = optimizer.solve(scenario, time_limit_seconds=60)
    
    # Print summary
    print("\nDIAGNOSTIC SUMMARY")
    print("=" * 50)
    print(f"Success: {result['success']}")
    
    if 'diagnostic_info' in result:
        diag = result['diagnostic_info']
        if 'instance_validation' in diag:
            val = diag['instance_validation']
            print(f"Locations: {val['num_locations']}")
            print(f"Vehicles: {val['num_vehicles']}")
            print(f"Requests: {val['num_requests']}")
    
    if 'infeasibility_analysis' in result:
        inf = result['infeasibility_analysis']
        if 'summary' in inf:
            print(f"Likely feasible: {inf['summary']['likely_feasible']}")
            print(f"Issues: {inf['summary']['identified_issues']}")
            for rec in inf['summary']['recommendations']:
                print(f"- {rec}")
    
    # Perform comprehensive sanity check
    optimizer._print_comprehensive_sanity_check(scenario)
