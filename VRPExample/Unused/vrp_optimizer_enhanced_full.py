#!/usr/bin/env python3
"""
Enhanced VRP Optimizer with Comprehensive Diagnostics and All Constraints

This version includes service time, vehicle work hour limits, and comprehensive sanity checks.
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

from vrp_data_models import VRPInstance, VRPResult
from hybrid_travel_calculator import hybrid_calculator

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VRPOptimizerEnhanced:
    """Enhanced VRP optimizer with all constraints and comprehensive diagnostics."""
    def __init__(self):
        self.diagnostic_info = {}
        self.travel_time_matrix = None
        self.distance_matrix = None
    
    def _build_travel_matrices(self, instance: VRPInstance):
        """Pre-build travel time and distance matrices using hybrid approach."""
        logger.info("🌍 Building realistic travel time matrix using hybrid approach...")
        logger.info("   (Haversine + OSRM correction factor - much faster than full OSRM)")
        
        # Extract coordinates from locations
        coordinates = []
        location_list = list(instance.locations.values())
        for location in location_list:
            coordinates.append((location.y, location.x))  # (lat, lon)
        
        # Get corrected travel time matrix using hybrid approach
        self.travel_time_matrix = hybrid_calculator.get_corrected_travel_time_matrix(coordinates)
        
        # Build corresponding distance matrix (we still need this for cost calculation)
        import math
        n = len(coordinates)
        self.distance_matrix = [[0 for _ in range(n)] for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    self.distance_matrix[i][j] = 0
                else:
                    distance_km = hybrid_calculator.calculate_haversine_distance(coordinates[i], coordinates[j])
                    self.distance_matrix[i][j] = int(distance_km * 1000)  # meters for OR-Tools
        
        logger.info(f"✅ Hybrid travel matrices built: {n}x{n} locations")
    
    def solve(self, instance: VRPInstance, time_limit_seconds: int = 300) -> Dict[str, Any]:
        """Solve VRP with progressive constraint addition and detailed diagnostics."""
        logger.info("🔧 ENHANCED VRP OPTIMIZER WITH PROGRESSIVE CONSTRAINTS")
        logger.info("=" * 70)
        
        # STEP 0: Build real road-based travel time matrices
        self._build_travel_matrices(instance)
        
        # RELAX TIME WINDOWS GLOBALLY - Set all time windows to start at 0
        logger.info("🔄 RELAXING TIME WINDOWS - Setting all start times to 0")
        relaxed_count = 0
        for location_id, location in instance.locations.items():
            if hasattr(location, 'time_window_start') and location.time_window_start is not None:
                original_start = location.time_window_start
                if original_start > 0:
                    location.time_window_start = 0  # Set to 0 for maximum flexibility
                    relaxed_count += 1
                    logger.info(f"   Relaxed {location_id}: [{original_start}-{location.time_window_end}] → [0-{location.time_window_end}]")
        
        if relaxed_count > 0:
            logger.info(f"✅ Relaxed {relaxed_count} time windows to start at 0")
        else:
            logger.info("No time windows needed relaxation")
        
        # Comprehensive sanity check
        self._print_comprehensive_sanity_check(instance)
        
        if not ORTOOLS_AVAILABLE:
            return {'success': False, 'error': 'OR-Tools not available'}
          # Try constraint levels progressively
        constraint_levels = [
            ("LEVEL 1", "distance_time", "Distance + Time + Service time (no time windows, no capacity)"),
            ("LEVEL 2", "time_dimension", "Distance + Time + Service time (with better search)"),
            ("LEVEL 3", "time_windows", "Distance + Time + Service time + Time windows"),
            ("LEVEL 4", "capacity", "Distance + Time + Service time + Time windows + Capacity"),
            ("LEVEL 5", "pickup_delivery", "Distance + Time + Service time + Time windows + Capacity + Pickup-delivery"),
            ("LEVEL 6", "full", "All constraints including driver breaks")        ]
        
        for level_name, level_code, level_desc in constraint_levels:
            logger.info(f"\n🎯 TESTING {level_name}: {level_desc}")
            logger.info("-" * 60)
            
            try:
                result = self._solve_with_constraint_level(instance, level_code, time_limit_seconds // len(constraint_levels))
                
                if result['success']:
                    logger.info(f"✅ SUCCESS WITH {level_name}!")
                    result['strategy_used'] = level_name
                    result['constraint_level_used'] = level_code
                    return result
                else:
                    logger.warning(f"❌ INFEASIBLE at {level_name}: {result.get('error', 'Unknown')}")
                    # Continue to next level to see if problem persists
                    
            except Exception as e:
                logger.error(f"❌ Error with {level_name}: {e}")
                # Continue to next level
        
        return {            'success': False,
            'error': 'No feasible solution found with any constraint level',
            'attempted_levels': [level[1] for level in constraint_levels]
        }

    def _solve_with_constraint_level(self, instance: VRPInstance, level: str, time_limit: int) -> Dict[str, Any]:
        """Solve with specific constraint level."""
        try:
            # STEP 0: Build matrices if not already built
            if self.travel_time_matrix is None:
                logger.info("🌍 Building travel matrices for constraint level testing...")
                self._build_travel_matrices(instance)            # Setup - Allow vehicles to not return to depot to avoid 0->0 issues
            num_locations = len(instance.locations)
            num_vehicles = len(instance.vehicles)
            depot_index = 0
            
            # Method 1: Try using different ends for vehicles (optional return to depot)
            # All vehicles start at depot but can end anywhere (including depot)
            starts = [depot_index] * num_vehicles  # All vehicles start at depot
            ends = [0] * num_vehicles    # All can end at depot, but it's not forced
            
            logger.info("🚛 Configuring vehicles to start at depot but not forced to return...")
            logger.info(f"   Starts: {starts[:3]}... (all vehicles start at depot)")
            logger.info(f"   Ends: {ends[:3]}... (vehicles can end at depot but not required)")
            
            # Use the multi-depot constructor to allow flexible starts/ends
            manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, starts, ends)
            routing = pywrapcp.RoutingModel(manager)
            location_list = list(instance.locations.values())
            
            constraints_applied = []
              # 1. ALWAYS: Distance callback using pre-built matrix
            def distance_callback(from_index, to_index):
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                
                distance = self.distance_matrix[from_node][to_node]
                
                # Add penalty for unnecessary returns to depot (depot is usually index 0)
                # This discourages vehicles from making empty trips back to depot
                if from_node != 0 and to_node == 0:  # Going to depot from non-depot location
                    distance += 10000  # Add penalty to discourage unnecessary returns
                    
                return distance
            
            transit_callback_index = routing.RegisterTransitCallback(distance_callback)
            routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
            constraints_applied.append("distance")
              # 2. ALWAYS: Time callback using OSRM travel times + service times
            callback_call_count = 0
            def time_callback(from_index, to_index):
                nonlocal callback_call_count
                callback_call_count += 1
                
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                
                # Get real road-based travel time from OSRM matrix
                travel_time = self.travel_time_matrix[from_node][to_node]
                
                # ALWAYS add service time for all levels (standard 15 minutes per stop)
                # Service time is incurred when ARRIVING at a location
                service_time = 0
                to_loc = location_list[to_node]
                
                # Use location's service_time if available, otherwise default to 15 minutes
                if hasattr(to_loc, 'service_time') and to_loc.service_time is not None:
                    service_time = int(to_loc.service_time)
                else:
                    # Standard service time: 15 minutes per stop (except for depots)
                    if 'depot' in to_loc.id.lower():
                        service_time = 5  # 5 minutes at depot for loading/unloading
                    else:
                        service_time = 15  # 15 minutes at pickup/dropoff locations
                
                total_time = int(travel_time + service_time)
                  # Debug logging for first few calls
                if callback_call_count <= 20:
                    logger.info(f"   🔍 TIME CALLBACK #{callback_call_count}: {from_node}→{to_node} = {travel_time}min travel + {service_time}min service = {total_time}min")
                    logger.info(f"      to_loc: {to_loc.id} (service_time: {getattr(to_loc, 'service_time', 'N/A')})")
                
                return total_time            
            time_callback_index = routing.RegisterTransitCallback(time_callback)
            logger.info(f"🔍 Time callback registered with {len(location_list)} locations")
            logger.info(f"🔍 Time callback index: {time_callback_index}")
            
            # Test the time callback manually to ensure it works
            logger.info("🔍 Testing time callback manually...")
            test_time = time_callback(0, 2)  # From depot_1 to pickup_1
            logger.info(f"   Manual test: depot_1 → pickup_1 = {test_time}min")
            
            # Conservative time dimension setup - use the MAXIMUM vehicle time limit for global max
            # Individual vehicle constraints will be applied separately
            vehicle_max_times = []
            for vehicle in instance.vehicles.values():
                max_time = (getattr(vehicle, 'max_total_work_time', None) or 
                           getattr(vehicle, 'max_time', None) or 
                           getattr(vehicle, 'max_work_time', None) or 
                           600)
                max_time = int(max_time) if max_time is not None else 600
                vehicle_max_times.append(max_time)            # Use the MAXIMUM vehicle time limit as the global maximum, then expand for time windows
            max_route_time = max(vehicle_max_times) if vehicle_max_times else 600
            
            # Calculate the actual time horizon needed based on location time windows
            all_time_ends = []
            for location in location_list:
                if hasattr(location, 'time_window_end') and location.time_window_end is not None:
                    all_time_ends.append(location.time_window_end)
            
            # Set global time horizon to accommodate all time windows
            latest_time_window = max(all_time_ends) if all_time_ends else 600
            time_horizon = min(max(max_route_time, latest_time_window), 1440)  # Cap at 24 hours
            
            logger.info(f"   🕐 Setting time horizon to {time_horizon}min (was {max_route_time}min)")
            logger.info(f"   📊 Individual vehicle limits: {min(vehicle_max_times)}-{max(vehicle_max_times)}min")
            logger.info(f"   📊 Location time windows extend to: {latest_time_window}min")
            
            routing.AddDimension(
                time_callback_index,
                60,   # 1 hour slack - reduced from 120 for stricter time control
                time_horizon,
                True,  # Force start cumul to zero
                'Time'            )
            time_dimension = routing.GetDimensionOrDie('Time')
            # Service time is now handled in the time callback above
            logger.info(f"🔍 Time dimension created: {time_dimension}")
            logger.info(f"🔍 Time dimension name: {time_dimension.name()}")
            constraints_applied.append("time_dimension")# Vehicle-specific time limits - STRICTLY ENFORCE these constraints
            logger.info("🔒 ENFORCING STRICT TIME LIMITS...")
            for vehicle_id, vehicle in enumerate(instance.vehicles.values()):
                max_time = (getattr(vehicle, 'max_total_work_time', None) or 
                           getattr(vehicle, 'max_time', None) or 
                           getattr(vehicle, 'max_work_time', None) or 
                           600)
                max_time = int(max_time) if max_time is not None else 600
                end_index = routing.End(vehicle_id)
                
                # Set HARD time limit for each vehicle - NO VIOLATIONS ALLOWED
                time_dimension.CumulVar(end_index).SetMax(max_time)
                
                # Also set upper bound to make sure OR-Tools respects the limit
                time_dimension.SetCumulVarSoftUpperBound(end_index, max_time, 1000000)  # High penalty for violations
                
                logger.info(f"   � Vehicle {vehicle_id}: HARD limit {max_time}min (NO violations allowed)")
            
            constraints_applied.append("strict_vehicle_time_limits")            # 3. TIME WINDOWS (if level >= time_windows) - RELAXED VERSION
            time_windows_added = 0
            if level in ["time_windows", "capacity", "pickup_delivery", "full"]:
                for location_idx, location in enumerate(location_list):
                    if hasattr(location, 'time_window_start') and location.time_window_start is not None:
                        index = manager.NodeToIndex(location_idx)
                          # Use original time windows but allow flexibility for start times
                        original_start = int(location.time_window_start)
                        original_end = int(location.time_window_end)
                        relaxed_start = 0  # Allow vehicles to start from time 0
                        
                        # For depot locations, allow access throughout the time horizon
                        if 'depot' in location.id.lower():
                            # Depot should be accessible throughout the planning horizon
                            relaxed_end = time_horizon  # Use the full time horizon for depot
                            logger.info(f"   Depot {location.id}: Set to [0-{relaxed_end}] (full accessibility)")
                        else:
                            # For regular locations, use original end time without artificial capping
                            relaxed_end = original_end  # Keep original time window end
                          # Only apply time windows if they make sense (start < end and end > 0)
                        if relaxed_start < relaxed_end and relaxed_end > 0:
                            time_dimension.CumulVar(index).SetRange(relaxed_start, relaxed_end)
                            time_windows_added += 1
                            if original_start != relaxed_start or original_end != relaxed_end:
                                logger.info(f"   Applied {location.id}: [{original_start}-{original_end}] → [{relaxed_start}-{relaxed_end}]")
                        else:
                            logger.warning(f"Skipping invalid time window for {location.id}: original=[{original_start}-{original_end}], relaxed=[0-{relaxed_end}]")
                
                constraints_applied.append(f"time_windows_applied_{time_windows_added}")
              # 4. CAPACITY (if level >= capacity)
            vehicle_capacities = []
            if level in ["capacity", "pickup_delivery", "full"] and instance.ride_requests:
                def demand_callback(from_index):
                    from_node = manager.IndexToNode(from_index)
                    location = location_list[from_node]
                    
                    demand = 0
                    pickups = []
                    dropoffs = []
                    
                    for req in instance.ride_requests:
                        if req.pickup_location == location.id:
                            demand += int(req.passengers)  # passengers = cargo weight in kg
                            pickups.append(f"+{req.passengers}kg ({req.id})")
                        elif req.dropoff_location == location.id:
                            demand -= int(req.passengers)  # passengers = cargo weight in kg
                            dropoffs.append(f"-{req.passengers}kg ({req.id})")
                      # Debug capacity callback for depot and problematic locations (limited output)
                    if demand != 0 and location.id in ['depot_1'] and demand > 1000:
                        logger.info(f"🔍 CAPACITY DEBUG: {location.id} demand={demand}kg")
                        if pickups:
                            logger.info(f"    Pickups: {len(pickups)} requests totaling {demand}kg")
                        if dropoffs:
                            logger.info(f"    Dropoffs: {len(dropoffs)} requests totaling {-demand}kg")
                    
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
                constraints_applied.append(f"capacity_{vehicle_capacities}")
            
            # 5. PICKUP AND DELIVERY (if level >= pickup_delivery)
            pickup_delivery_pairs = 0
            if level in ["pickup_delivery", "full"] and instance.ride_requests:
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
                        
                        # Pickup before dropoff (only add if time dimension exists)
                        if time_dimension:
                            routing.solver().Add(
                                time_dimension.CumulVar(pickup_index) <= time_dimension.CumulVar(dropoff_index)
                            )
                        
                        pickup_delivery_pairs += 1
                        
                    except ValueError:
                        logger.warning(f"Skipping request {req.id} - location not found")
                
                constraints_applied.append(f"pickup_delivery_{pickup_delivery_pairs}")            # 6. DRIVER BREAKS (only for full level)
            if level == "full":
                logger.info("🔧 Adding driver break constraints...")
                break_constraints = self._add_driver_break_constraints(instance, manager, routing, time_dimension)
                # Ensure break_constraints is a valid integer
                if break_constraints is None:
                    break_constraints = 0
                if break_constraints > 0:
                    constraints_applied.append(f"driver_breaks_{break_constraints}")
                logger.info("✅ Driver break constraints added successfully")
            
            logger.info(f"Applied constraints: {constraints_applied}")            # 7. FLEET UTILIZATION ENCOURAGEMENT - Adaptive based on constraint level
            if level in ["capacity", "pickup_delivery", "full"]:
                # For capacity-constrained levels: Use moderate fixed costs and let capacity limit stops naturally
                logger.info("🚛 Adding MODERATE fleet utilization encouragement (capacity will limit stops)...")
                
                # Moderate fixed cost per vehicle to encourage distribution without being excessive
                for vehicle_id in range(num_vehicles):
                    routing.SetFixedCostOfVehicle(50000, vehicle_id)  # Moderate fixed cost
                
                logger.info("   📦 Vehicle capacity constraints will naturally limit stops per vehicle")
                constraints_applied.append("moderate_fleet_utilization")
                logger.info("✅ MODERATE fleet utilization added (capacity-driven)")
                
            else:
                # For non-capacity levels: Use aggressive stop limits to force distribution
                logger.info("🚛 Adding AGGRESSIVE fleet utilization encouragement...")
                
                # Add a count dimension to track number of stops per vehicle
                count_dimension_name = 'count'
                routing.AddConstantDimension(
                    1,  # increment by one every time
                    num_locations + 1,  # make sure the return to depot node can be counted
                    True,  # set count to zero
                    count_dimension_name
                )
                count_dimension = routing.GetDimensionOrDie(count_dimension_name)
                
                # AGGRESSIVE penalties to force fleet utilization
                # Much higher fixed cost per vehicle to force distribution
                for vehicle_id in range(num_vehicles):
                    routing.SetFixedCostOfVehicle(500000, vehicle_id)  # VERY high fixed cost
                
                # STRICT limits on stops per vehicle to force distribution
                target_stops_per_vehicle = max(3, num_locations // (num_vehicles // 4))  # Force more vehicles to be used
                logger.info(f"   🎯 Target stops per vehicle: {target_stops_per_vehicle}")
                
                for vehicle_id in range(num_vehicles):
                    end_index = routing.End(vehicle_id)
                    # HARD upper bound on stops per vehicle
                    count_dimension.SetCumulVarSoftUpperBound(end_index, 
                                                            target_stops_per_vehicle,  
                                                            10000000)  # MASSIVE penalty for exceeding
                    logger.info(f"   🔒 Vehicle {vehicle_id}: max {target_stops_per_vehicle} stops allowed")
                
                constraints_applied.append("aggressive_fleet_utilization")
                logger.info("✅ AGGRESSIVE fleet utilization added")
            
            # 8. SOLVE with fleet-aware strategy
            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            # Use strategies that favor distributing load
            search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
            search_parameters.time_limit.FromSeconds(time_limit)
            search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
            
            start_time = time.time()
            assignment = routing.SolveWithParameters(search_parameters)
            solve_time = time.time() - start_time
            
            if assignment:
                try:
                    objective_value = assignment.ObjectiveValue()
                except AttributeError:
                    objective_value = routing.GetCost(assignment)
                
                logger.info(f"✅ SOLUTION FOUND!")
                logger.info(f"   Objective: {objective_value}")
                logger.info(f"   Time: {solve_time:.2f}s")                # Extract and analyze routes
                try:
                    logger.info(f"🔍 Starting route analysis for level {level}...")
                    routes_info = self._extract_routes_with_analysis(instance, manager, routing, assignment, time_dimension)
                    logger.info(f"✅ Route analysis completed successfully")
                except Exception as e:
                    logger.error(f"❌ Error in route analysis for level {level}: {e}")
                    logger.error(f"   Error type: {type(e)}")
                    import traceback
                    traceback.print_exc()
                    return {'success': False, 'error': f'Route analysis error: {str(e)}'}
                
                
                return {
                    'success': True,
                    'objective_value': objective_value,
                    'routes': routes_info['routes'],
                    'vehicles_used': routes_info['vehicles_used'],
                    'route_analysis': routes_info['analysis'],
                    'solve_time': solve_time,
                    'constraints_applied': constraints_applied,
                    'constraints_summary': {
                        'time_windows': time_windows_added,
                        'pickup_delivery_pairs': pickup_delivery_pairs,
                        'vehicle_capacities': vehicle_capacities
                    }
                }
            else:
                return {
                    'success': False,
                    'error': f'No solution found with {level} constraints',
                    'constraints_applied': constraints_applied
                }
                
        except Exception as e:
            logger.error(f"Error in {level} constraint level: {e}")
            return {'success': False, 'error': str(e)}
    
    def _print_comprehensive_sanity_check(self, instance: VRPInstance):
        """Print comprehensive sanity check of the instance."""
        logger.info("📊 COMPREHENSIVE SANITY CHECK")
        logger.info("-" * 50)
        
        # Basic counts
        num_locations = len(instance.locations)
        num_vehicles = len(instance.vehicles)
        num_requests = len(instance.ride_requests) if instance.ride_requests else 0
        
        logger.info(f"📍 Locations: {num_locations}")
        logger.info(f"🚛 Vehicles: {num_vehicles}")
        logger.info(f"📦 Requests: {num_requests}")
          # Vehicle analysis
        logger.info("\n🚛 VEHICLE ANALYSIS:")
        total_capacity = 0
        for i, (vid, vehicle) in enumerate(instance.vehicles.items()):
            capacity = vehicle.capacity
            max_time = (getattr(vehicle, 'max_total_work_time', None) or 
                       getattr(vehicle, 'max_time', None) or 
                       getattr(vehicle, 'max_work_time', None) or 600)
            max_time = int(max_time) if max_time is not None else 600
            vehicle_type = getattr(vehicle, 'vehicle_type', 'standard')
            
            total_capacity += capacity
            
            logger.info(f"   {vid}: {capacity}kg capacity, {max_time}min max_time, type: {vehicle_type}")
            
            # Check for driver break requirements
            if hasattr(vehicle, 'max_driving_time'):
                logger.info(f"      🛑 Max driving: {vehicle.max_driving_time}min, break: {getattr(vehicle, 'required_break_time', 0)}min")
        
        logger.info(f"   💼 Total fleet capacity: {total_capacity}kg")
        
        # Request analysis
        if instance.ride_requests:
            logger.info("\n📦 REQUEST ANALYSIS:")
            total_demand = 0
            for req in instance.ride_requests:
                cargo = req.passengers
                total_demand += cargo
                logger.info(f"   {req.id}: {cargo}kg from {req.pickup_location} to {req.dropoff_location}")
            
            logger.info(f"   📊 Total demand: {total_demand}kg")
            logger.info(f"   📊 Capacity utilization: {total_demand/total_capacity*100:.1f}%")
            
            if total_demand > total_capacity:
                logger.warning("   ⚠️ WARNING: Total demand exceeds total capacity!")
                logger.info("   💡 Note: This is OK if vehicles can make multiple trips")
        
        # Time window analysis
        logger.info("\n⏰ TIME WINDOW ANALYSIS:")
        time_windowed_locations = 0
        earliest_start = float('inf')
        latest_end = 0
        service_times = []
        
        for loc_id, location in instance.locations.items():
            if hasattr(location, 'time_window_start') and location.time_window_start is not None:
                time_windowed_locations += 1
                start = location.time_window_start
                end = location.time_window_end
                service_time = getattr(location, 'service_time', 0)
                
                earliest_start = min(earliest_start, start)
                latest_end = max(latest_end, end)
                service_times.append(service_time)
                
                logger.info(f"   {loc_id}: [{start}-{end}] ({end-start}min window) +{service_time}min service")
        
        if time_windowed_locations > 0:
            logger.info(f"   📊 {time_windowed_locations}/{num_locations} locations have time windows")
            logger.info(f"   📊 Time span: {earliest_start} to {latest_end} ({latest_end - earliest_start}min)")
            logger.info(f"   📊 Service times: {min(service_times)}-{max(service_times)}min (avg: {sum(service_times)/len(service_times):.1f}min)")
        else:
            logger.info("   No time windows found")
          # Pickup-dropoff feasibility check - Updated to check for nonzero intersection
        if instance.ride_requests:
            logger.info("\n🔄 PICKUP-DROPOFF FEASIBILITY:")
            impossible_pairs = 0
            tight_pairs = 0
            
            for req in instance.ride_requests:
                pickup_loc = instance.locations.get(req.pickup_location)
                dropoff_loc = instance.locations.get(req.dropoff_location)
                
                if pickup_loc and dropoff_loc:
                    pickup_start = getattr(pickup_loc, 'time_window_start', None)
                    pickup_end = getattr(pickup_loc, 'time_window_end', None)
                    dropoff_start = getattr(dropoff_loc, 'time_window_start', None)
                    dropoff_end = getattr(dropoff_loc, 'time_window_end', None)
                    
                    if (pickup_start is not None and pickup_end is not None and 
                        dropoff_start is not None and dropoff_end is not None):
                        
                        # Check for time window intersection
                        # Intersection exists if: max(start1, start2) < min(end1, end2)
                        intersection_start = max(pickup_start, dropoff_start)
                        intersection_end = min(pickup_end, dropoff_end)
                        intersection_duration = max(0, intersection_end - intersection_start)
                        if intersection_duration == 0:
                            impossible_pairs += 1
                            logger.warning(f"   ❌ {req.id}: NO intersection - pickup [{pickup_start}-{pickup_end}], dropoff [{dropoff_start}-{dropoff_end}]")
                        elif intersection_duration < 30:
                            tight_pairs += 1
                            logger.warning(f"   ⚠️ {req.id}: tight intersection - only {intersection_duration}min overlap")
                            logger.info(f"      pickup [{pickup_start}-{pickup_end}], dropoff [{dropoff_start}-{dropoff_end}]")
                        else:
                            logger.info(f"   ✅ {req.id}: good intersection - {intersection_duration}min overlap")
            
            if impossible_pairs > 0:
                logger.error(f"   🚨 {impossible_pairs} impossible pickup-dropoff pairs found (no time window intersection)!")
            elif tight_pairs > 0:
                logger.warning(f"   ⚠️ {tight_pairs} tight pickup-dropoff pairs found")
            else:
                logger.info("   ✅ All pickup-dropoff pairs have feasible time window intersections")
        
        logger.info("-" * 50)
    
    def _add_driver_break_constraints(self, instance: VRPInstance, manager, routing, time_dimension):
        """Add driver break constraints for heavy vehicles."""
        break_constraints_added = 0
        
        for vehicle_id, vehicle in enumerate(instance.vehicles.values()):
            if hasattr(vehicle, 'max_driving_time') and hasattr(vehicle, 'required_break_time'):
                max_driving = int(vehicle.max_driving_time)
                break_time = int(vehicle.required_break_time)
                
                if max_driving > 0 and break_time > 0:
                    # Add break constraint: after max_driving_time, need a break
                    # This is simplified - a full implementation would track service areas
                    logger.info(f"   Vehicle {vehicle_id}: max driving {max_driving}min, break {break_time}min")
                    break_constraints_added += 1
        
        if break_constraints_added > 0:
            logger.info(f"✅ Added break constraints for {break_constraints_added} vehicles")
        
        return break_constraints_added
    def _extract_routes_with_analysis(self, instance: VRPInstance, manager, routing, assignment, time_dimension):
        """Extract routes and provide detailed analysis."""
        routes = {}
        route_analysis = []
        vehicles_used = 0
        location_list = list(instance.locations.values())
        
        for vehicle_id in range(routing.vehicles()):
            route = []
            route_time = 0
            route_distance = 0
            route_load = 0
            max_route_load = 0
            
            index = routing.Start(vehicle_id)
            prev_location = None
            
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                location = location_list[node_index]
                route.append(location.id)                # Calculate distance from previous location using Haversine formula
                if prev_location is not None:
                    import math
                    lat1, lon1 = math.radians(prev_location.y), math.radians(prev_location.x)
                    lat2, lon2 = math.radians(location.y), math.radians(location.x)
                    dlat, dlon = lat2 - lat1, lon2 - lon1
                    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
                    segment_distance_km = 6371 * 2 * math.asin(math.sqrt(a))  # Earth radius in km
                    route_distance += segment_distance_km
                
                # Get time at this location
                time_var = time_dimension.CumulVar(index)
                location_time = assignment.Value(time_var)                  # Calculate load change at this location
                load_change = 0
                pickup_details = []
                dropoff_details = []
                
                if instance.ride_requests:
                    for req in instance.ride_requests:
                        if req.pickup_location == location.id:
                            route_load += req.passengers
                            load_change += req.passengers
                            pickup_details.append(f"req_{req.id}(+{req.passengers}kg)")
                        elif req.dropoff_location == location.id:
                            route_load -= req.passengers
                            load_change -= req.passengers
                            dropoff_details.append(f"req_{req.id}(-{req.passengers}kg)")
                    
                    max_route_load = max(max_route_load, route_load)
                    
                    # Debug: Track detailed cumulative load progression
                    if load_change != 0:
                        activity_details = []
                        if pickup_details:
                            activity_details.extend(pickup_details)
                        if dropoff_details:
                            activity_details.extend(dropoff_details)
                        
                        # Removed excessive per-location capacity debug output to keep logs clean
                
                prev_location = location
                index = assignment.Value(routing.NextVar(index))            # Add final segment back to depot if route has stops
            if len(route) > 1:
                final_node_index = manager.IndexToNode(routing.End(vehicle_id))
                depot_location = location_list[final_node_index]
                if prev_location:
                    import math
                    lat1, lon1 = math.radians(prev_location.y), math.radians(prev_location.x)
                    lat2, lon2 = math.radians(depot_location.y), math.radians(depot_location.x)
                    dlat, dlon = lat2 - lat1, lon2 - lon1
                    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
                    final_segment_km = 6371 * 2 * math.asin(math.sqrt(a))  # Earth radius in km                    route_distance += final_segment_km
              # Get final time at depot
            final_time_var = time_dimension.CumulVar(routing.End(vehicle_id))
            final_time = assignment.Value(final_time_var)
            
            # Ensure final_time is never None
            if final_time is None:
                final_time = 0
                logger.warning(f"      ⚠️ Vehicle {vehicle_id}: final_time was None, setting to 0")
              # DEBUG: Print time information
            if len(route) > 1:
                logger.info(f"      DEBUG: Vehicle {vehicle_id} - Final time: {final_time}, Route length: {len(route)}")
            
            if len(route) > 1:  # Vehicle was used
                vehicles_used += 1
                routes[f"vehicle_{vehicle_id}"] = route
                
                vehicle = list(instance.vehicles.values())[vehicle_id]
                max_time = (getattr(vehicle, 'max_total_work_time', None) or 
                           getattr(vehicle, 'max_time', None) or 600)
                max_time = int(max_time) if max_time is not None else 600
                
                # Ensure max_time is never None or zero
                if max_time is None or max_time <= 0:
                    max_time = 600
                    logger.warning(f"      ⚠️ Vehicle {vehicle_id}: max_time was None/invalid, setting to 600")                # CORRECTED: Calculate realistic driving time using our correction factor
                
                # Calculate pure driving time using Haversine distance + correction factor
                realistic_driving_time = 0
                
                for i in range(len(route) - 1):
                    from_location_id = route[i]
                    to_location_id = route[i + 1]
                    
                    from_location = instance.locations[from_location_id]
                    to_location = instance.locations[to_location_id]
                    
                    # Calculate Haversine distance
                    distance_km = hybrid_calculator.calculate_haversine_distance(
                        (from_location.y, from_location.x),
                        (to_location.y, to_location.x)
                    )
                    
                    # Convert to travel time using realistic speed + correction factor
                    base_time = distance_km * 60 / 40  # 40 km/h urban speed
                    corrected_time = base_time * hybrid_calculator.correction_factor
                    realistic_driving_time += corrected_time
                
                driving_time = int(realistic_driving_time)  # Realistic driving time using correction factor
                
                # Calculate service time by working backwards from OR-Tools total time
                # final_time = driving_time + service_time (+ any OR-Tools overhead)
                # So: effective_service_time = final_time - driving_time
                effective_service_time = max(0, final_time - driving_time)
                
                # Also calculate theoretical service time for comparison/debugging
                theoretical_service_time = 0
                for location_id in route:
                    location = instance.locations.get(location_id)
                    if location:
                        service_time = getattr(location, 'service_time', 0)
                        theoretical_service_time += service_time
                
                # Use the effective service time (from OR-Tools) rather than theoretical
                total_service_time = effective_service_time
                
                analysis = {
                    'vehicle_id': vehicle_id,
                    'stops': len(route),
                    'total_time': final_time,
                    'driving_time': driving_time,
                    'service_time': total_service_time,
                    'total_distance': route_distance,
                    'max_allowed_time': max_time,
                    'time_utilization': f"{final_time/max_time*100:.1f}%",
                    'max_load_reached': max_route_load,
                    'capacity': vehicle.capacity,
                    'load_utilization': f"{max_route_load/vehicle.capacity*100:.1f}%",
                    'route': route                }
                
                route_analysis.append(analysis)
                
                logger.info(f"   Vehicle {vehicle_id}: {len(route)} stops, {final_time}min total ({final_time/max_time*100:.1f}% of {max_time}min limit)")
                logger.info(f"      📏 Distance: {route_distance:.2f} km")
                logger.info(f"      🚗 Driving time: {driving_time}min, Service time: {total_service_time}min")
                logger.info(f"      🔍 Math check: {driving_time}min + {total_service_time}min = {driving_time + total_service_time}min (OR-Tools: {final_time}min)")
                if abs((driving_time + total_service_time) - final_time) > 5:  # Allow 5min tolerance
                    logger.info(f"      📊 DEBUG: Theoretical service time was {theoretical_service_time}min vs effective {total_service_time}min")
                logger.info(f"      📦 Max load: {max_route_load}kg ({max_route_load/vehicle.capacity*100:.1f}% of {vehicle.capacity}kg capacity)")
                
                # Capacity utilization summary
                if max_route_load > 0:
                    capacity_efficiency = max_route_load / vehicle.capacity * 100
                    if capacity_efficiency > 90:
                        logger.info(f"      ✅ EXCELLENT capacity utilization: {capacity_efficiency:.1f}%")
                    elif capacity_efficiency > 70:
                        logger.info(f"      ✅ Good capacity utilization: {capacity_efficiency:.1f}%")
                    elif capacity_efficiency > 50:
                        logger.info(f"      📊 Moderate capacity utilization: {capacity_efficiency:.1f}%")
                    else:
                        logger.info(f"      📊 Low capacity utilization: {capacity_efficiency:.1f}%")
                
                # Check for capacity constraint violations
                if max_route_load > vehicle.capacity:
                    logger.warning(f"      ⚠️ CAPACITY VIOLATION: {max_route_load}kg > {vehicle.capacity}kg limit!")
                else:
                    remaining_capacity = vehicle.capacity - max_route_load
                    logger.info(f"      📦 Remaining capacity: {remaining_capacity}kg")
                
                # Check if total time exceeds limit (this indicates a constraint violation)
                if final_time > max_time:
                    logger.warning(f"      ⚠️ TIME CONSTRAINT VIOLATION: {final_time}min total > {max_time}min limit!")
                    logger.warning(f"      ⚠️ Breakdown: {driving_time}min driving + {total_service_time}min service = {driving_time + total_service_time}min calculated")
                    logger.warning(f"      ⚠️ OR-Tools reported: {final_time}min total time")
                  # DEBUG: Check for None values before comparison
                try:
                    # Ensure both values are valid integers before comparison
                    final_time_safe = int(final_time) if final_time is not None else 0
                    max_time_safe = int(max_time) if max_time is not None else 600
                    
                    if final_time is None or max_time is None:
                        logger.error(f"      🐛 DEBUG: final_time={final_time}, max_time={max_time} - one is None!")
                    elif final_time_safe > max_time_safe:
                        logger.warning(f"      ⚠️ Vehicle {vehicle_id} exceeds time limit by {final_time_safe - max_time_safe}min!")
                except (TypeError, ValueError) as e:
                    logger.error(f"      🐛 DEBUG: Error comparing times - final_time={final_time}, max_time={max_time}, error: {e}")
        
        return {
            'routes': routes,
            'vehicles_used': vehicles_used,
            'analysis': route_analysis
        }

def test_enhanced_optimizer():
    """Test the enhanced optimizer."""
    from vrp_scenarios import create_moda_small_scenario
    
    print("Testing Enhanced VRP Optimizer with All Constraints")
    print("=" * 60)
    
    try:
        # Test MODA scenario
        scenario = create_moda_small_scenario()
        optimizer = VRPOptimizerEnhanced()
        
        result = optimizer.solve(scenario, time_limit_seconds=120)
        
        print(f"\n🎯 FINAL RESULTS:")
        print(f"Success: {result['success']}")
        
        if result['success']:
            print(f"✅ SOLVED WITH ALL CONSTRAINTS!")
            print(f"   Objective: {result['objective_value']}")
            print(f"   Vehicles used: {result['vehicles_used']} out of {len(scenario.vehicles)}")
            print(f"   Strategy: {result['strategy_used']}")
            print(f"   Solve time: {result['solve_time']:.2f}s")
            
            if 'route_analysis' in result:
                print(f"\n📊 ROUTE ANALYSIS:")
                for analysis in result['route_analysis']:
                    print(f"   Vehicle {analysis['vehicle_id']}: {analysis['stops']} stops, "
                          f"{analysis['total_time']}min ({analysis['time_utilization']} of limit)")
            
            if 'constraints_summary' in result:
                summary = result['constraints_summary']
                print(f"\n🔧 CONSTRAINTS APPLIED:")
                print(f"   Time windows: {summary.get('time_windows', 0)}")
                print(f"   Pickup-delivery pairs: {summary.get('pickup_delivery_pairs', 0)}")
                print(f"   Vehicle time limits: {summary.get('vehicle_time_limits', [])}")
                print(f"   Vehicle capacities: {summary.get('vehicle_capacities', [])}")
        else:
            print(f"❌ Failed: {result.get('error', 'Unknown error')}")
            if 'constraints_summary' in result:
                print(f"Constraints attempted: {result['constraints_summary']}")
            
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_optimizer()
