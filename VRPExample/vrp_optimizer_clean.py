"""
Clean VRP Optimizer - Built step by step to debug constraint issues
"""
import logging
from typing import Dict, List, Tuple, Optional
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import matplotlib.pyplot as plt
import numpy as np

class CleanVRPOptimizer:
    """A clean VRP optimizer built step by step to debug constraint issues."""
    
    def __init__(self, vehicles=None, locations=None, vrp_instance=None, distance_matrix_provider: str = "google"):
        """Initializes the optimizer with vehicle and location data or VRPInstance."""
        if vrp_instance is not None:
            # Convert VRPInstance to the expected format
            self.vehicles = self._convert_vehicles_from_instance(vrp_instance)
            self.locations = self._convert_locations_from_instance(vrp_instance)
            self.ride_requests = vrp_instance.ride_requests
        else:
            self.vehicles = vehicles or []
            self.locations = locations or []
            self.ride_requests = {}
        
        self.distance_matrix_provider = distance_matrix_provider
        self.logger = logging.getLogger(__name__)
        self._active_ride_requests = []  # Track only ride requests actually added as pickup-delivery pairs
    
    def _convert_vehicles_from_instance(self, instance):
        """Convert VRPInstance vehicles to dict format (no capacity logic, just copy attributes)."""
        vehicles = []
        for vehicle in instance.vehicles.values():
            vehicles.append({
                'id': vehicle.id,
                'capacity': getattr(vehicle, 'capacity', 0),
                'start_location': getattr(vehicle, 'depot_id', None),
                'end_location': getattr(vehicle, 'depot_id', None),
                'max_time': getattr(vehicle, 'max_time', 24 * 60)
            })
        return vehicles
    
    def _convert_locations_from_instance(self, instance):
        """Convert VRPInstance locations to dict format."""
        locations = []
        
        # First pass: find the depot time window
        depot_time_window = None
        print("  Looking for depot time window...")
        
        for location in instance.locations.values():
            print(f"    Checking location: {location.id}")


            if 'depot' in location.id.lower() and 'bay' not in location.id.lower():
                if hasattr(location, 'time_window_start') and hasattr(location, 'time_window_end'):
                    if location.time_window_start is not None and location.time_window_end is not None:
                        depot_time_window = (location.time_window_start, location.time_window_end)
                        print(f"    Found depot time window: {depot_time_window} from {location.id}")
                        break
        
        if depot_time_window is None:
            print("    ⚠️ No depot time window found, depot bays will have no time windows")
          # Second pass: convert all locations
        for location in instance.locations.values():
            loc_dict = {
                'id': location.id,
                'x': location.x,  # longitude
                'y': location.y,  # latitude
                'demand': location.demand,
                'address': getattr(location, 'address', location.id)
            }
            
            # Add service time - 30 minutes for all locations except depot
            if 'depot' in location.id.lower():
                loc_dict['service_time'] = 0  # No service time at depot
            else:
                loc_dict['service_time'] = 30  # 30 minutes service time for all other locations

            if hasattr(location, 'time_window_start') and hasattr(location, 'time_window_end'):
                if location.time_window_start is not None and location.time_window_end is not None:
                    loc_dict['time_window'] = (location.time_window_start, location.time_window_end)
                else:
                    loc_dict['time_window'] = (0, 1440)  # default full day

            
            locations.append(loc_dict)
        return locations

    def solve(self, constraint_level: str = "none", verbose: bool = True, use_hybrid_calculator: bool = False) -> Optional[Dict]:
        """
        Solves the VRP with the specified level of constraints.
        
        Constraint levels:
        - "none": Just distance minimization
        - "capacity": Add capacity constraints  
        - "pickup_delivery": Add pickup-delivery constraints
        - "time_windows": Add time window constraints
        - "full": All constraints
        - verbose: If False, suppresses OR-Tools search logging
        - use_hybrid_calculator: If True, uses hybrid travel calculator for realistic travel times
        """
        print(f"\n🚀 Solving with constraint level: {constraint_level}")

        # Add comprehensive sanity check before solving
        self._print_comprehensive_sanity_check(constraint_level)

        location_list = self.locations
        vehicle_list =  self.vehicles

        # --- Robust vehicle index mapping ---
        # Map OR-Tools vehicle index to vehicle object (by start location and ID)
        
        vehicle_idx_to_vehicle = {}
        for idx, v in enumerate(vehicle_list):
            vehicle_idx_to_vehicle[idx] = v

        print("📊 Problem size:")
        print(f"  - Locations: {len(location_list)}")
        print(f"  - Vehicles: {len(vehicle_list)}")
        print(f"  - Vehicle order: {[v['id'] for v in vehicle_list]}")

        # Create a mapping from location ID to index for easy lookup
        location_to_index = {loc['id']: i for i, loc in enumerate(location_list)}

        # Get start and end indices for each vehicle
        try:
            start_indices = [location_to_index[v['start_location']] for v in vehicle_list]
            end_indices = [location_to_index[v['end_location']] for v in vehicle_list]
        except KeyError as e:
            self.logger.error(f"Location {e} in vehicle list not found in the main location list.")
            return None, "Error", ["Invalid location in vehicle data"]

        # 1. Create routing model
        manager = pywrapcp.RoutingIndexManager(
            len(location_list), len(vehicle_list), start_indices, end_indices
        )
        
        routing = pywrapcp.RoutingModel(manager)
        
        # 3. Distance callback (always needed)
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
         
            # Fallback to simple Euclidean distance calculation
            from_loc = location_list[from_node]
            to_loc = location_list[to_node]
            
            from_x = from_loc.get('x', 0)
            from_y = from_loc.get('y', 0)
            to_x = to_loc.get('x', 0)
            to_y = to_loc.get('y', 0)
            
            # Calculate Euclidean distance in coordinate units
            distance = ((from_x - to_x) ** 2 + (from_y - to_y) ** 2) ** 0.5
            
            # Convert coordinate distance to kilometers 
            # Using a reasonable scaling factor for the coordinate system
            distance_km = distance * 111  # Consistent with time callback
            
            # Return as integer meters for OR-Tools, but use smaller scale to make constraints dominate
            # Reduced scale so constraint penalties (1,000,000) are much larger than distance costs
            return int(distance_km * 100)  # Reduced from 1000 to 100 to make distance less dominant
        
        distance_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(distance_callback_index)
        
        applied_constraints = ["distance"]
          # 4. Add constraints based on level
        if constraint_level == "capacity":

            pickup_delivery_count = self._add_pickup_delivery_constraints(routing, manager, location_list, vehicle_list)
            applied_constraints.append(f"pickup_delivery_{pickup_delivery_count}")

            self._add_capacity_constraints(routing, manager, location_list, vehicle_list)
            applied_constraints.append("capacity")

        elif constraint_level == "pickup_delivery":
            # Only add pickup-delivery, no capacity
            pickup_delivery_count = self._add_pickup_delivery_constraints(routing, manager, location_list, vehicle_list)
            applied_constraints.append(f"pickup_delivery_{pickup_delivery_count}")
            
        elif constraint_level == "time_windows":
            # Only add time windows, no capacity or pickup-delivery
            self._add_time_window_constraints(routing, manager, location_list, vehicle_list)
            applied_constraints.append("time_windows")
            
        elif constraint_level == "full":

            pickup_delivery_count = self._add_pickup_delivery_constraints(routing, manager, location_list, vehicle_list)
            applied_constraints.append(f"pickup_delivery_{pickup_delivery_count}")

            self._add_capacity_constraints(routing, manager, location_list, vehicle_list)
            applied_constraints.append("capacity")

            
            self._add_time_window_constraints(routing, manager, location_list, vehicle_list)
            applied_constraints.append("time_windows")
        
        print(f"✅ Constraints applied: {applied_constraints}")
        
        # Add constraint penalties to prioritize constraint satisfaction over distance
        if constraint_level in ["capacity", "pickup_delivery", "time_windows", "full"]:
            print("\n⚖️ Adding constraint penalties to prioritize constraint satisfaction...")
            
            # Add high penalty for time window violations
            if "time_windows" in applied_constraints:
                time_dimension = routing.GetDimensionOrDie('Time')
                # Add penalty for being late (soft time windows with high penalty)
                penalty = 1000000  # Very high penalty for time window violations
                for idx, loc in enumerate(location_list):
                    time_window = loc.get('time_window', None)
                    if time_window is not None and len(time_window) == 2:
                        tw_start, tw_end = time_window
                        index = manager.NodeToIndex(idx)
                        # Add penalty for arriving after the time window
                        time_dimension.SetCumulVarSoftUpperBound(index, int(tw_end), penalty)
                        print(f"    Added penalty for late arrival at {loc['id']}: {penalty}")
            
            # Add penalty for vehicle max time violations using span cost coefficient
            if "time_windows" in applied_constraints:
                time_dimension = routing.GetDimensionOrDie('Time')
                # Use span cost coefficient to penalize long routes
                span_penalty = 1000  # Cost per minute of route duration
                for vehicle_idx in range(len(vehicle_list)):
                    vehicle = vehicle_list[vehicle_idx]
                    vehicle_max_time = vehicle.get('max_time', 9 * 60)
                    
                    # Set cost coefficient for vehicle span (route duration)
                    # This adds cost proportional to the route duration
                    time_dimension.SetSpanCostCoefficientForVehicle(span_penalty, vehicle_idx)
                    print(f"    Added span cost coefficient for vehicle {vehicle['id']}: {span_penalty} per minute")
            
            # Add penalty for capacity violations
            if "capacity" in applied_constraints:
                # The capacity constraints are hard constraints, but we can add penalties for load imbalance
                print("    Capacity constraints are hard constraints (no penalties needed)")
            
            print("✅ Constraint penalties added")
        
          # 5. Set search parameters with constraint-focused strategy
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        
        # Use PATH_CHEAPEST_ARC for better constraint satisfaction
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        
        # Use GUIDED_LOCAL_SEARCH for better optimization with constraints
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        
        search_parameters.time_limit.seconds = 120  # Increased time for better constraint satisfaction
        search_parameters.log_search = verbose  # Enable/disable detailed logging
        
        # Additional parameters to improve constraint satisfaction
        search_parameters.solution_limit = 100  # Try more solutions
        
        # Set high cost for unassigned nodes to force assignment
        penalty_cost = 10000000  # Very high penalty for unassigned nodes
        for node in range(routing.Size()):
            if not routing.IsStart(node) and not routing.IsEnd(node):
                routing.AddDisjunction([node], penalty_cost)
        
        print(f"🔧 Search parameters:")
        print(f"  - First solution strategy: PATH_CHEAPEST_ARC")
        print(f"  - Local search: GUIDED_LOCAL_SEARCH")
        print(f"  - Time limit: {search_parameters.time_limit.seconds} seconds")
        print(f"  - Solution limit: {search_parameters.solution_limit}")
        print(f"  - Unassigned node penalty: {penalty_cost}")
        print(f"  - Logging enabled: {search_parameters.log_search}")
          # 6. Solve
        print("🔍 Solving...")
        print("📊 Problem statistics before solving:")
        print(f"  - Total nodes: {routing.Size()}")
        print(f"  - Total vehicles: {routing.vehicles()}")
        print(f"  - Total constraints: {routing.solver().Constraints()}")
        
        solution = routing.SolveWithParameters(search_parameters)
        
        # Detailed status reporting
        status = routing.status()
        print(f"\n📋 Solver status: {status}")
        
        if status == routing_enums_pb2.FirstSolutionStrategy.UNSET:
            print("   Status detail: UNSET - No solution strategy set")
        elif status == routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC:
            print("   Status detail: AUTOMATIC - Automatic strategy")
        elif status == routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC:
            print("   Status detail: PATH_CHEAPEST_ARC - Using cheapest arc strategy")
        elif hasattr(routing_enums_pb2, 'ROUTING_NOT_SOLVED'):
            print("   Status detail: ROUTING_NOT_SOLVED - Problem not solved")
        elif hasattr(routing_enums_pb2, 'ROUTING_SUCCESS'):
            print("   Status detail: ROUTING_SUCCESS - Solution found")
        elif hasattr(routing_enums_pb2, 'ROUTING_FAIL'):
            print("   Status detail: ROUTING_FAIL - No solution exists")
        elif hasattr(routing_enums_pb2, 'ROUTING_FAIL_TIMEOUT'):
            print("   Status detail: ROUTING_FAIL_TIMEOUT - Time limit exceeded")
        elif hasattr(routing_enums_pb2, 'ROUTING_INVALID'):
            print("   Status detail: ROUTING_INVALID - Invalid problem")
        
        if solution:
            print("✅ Solution found!")
            result = self._extract_solution(routing, manager, solution, location_list, vehicle_list, constraint_level, vehicle_idx_to_vehicle)
            return result, "Success", applied_constraints
        else:
            print("❌ No solution found!")
            
            # Additional diagnostics
            print("\n🔍 Diagnostic information:")
            print(f"  - Routing model size: {routing.Size()}")
            print(f"  - Number of vehicles: {routing.vehicles()}")
            print(f"  - Number of constraints: {routing.solver().Constraints()}")
            
            # Check if any vehicles are used
            print("\n🚗 Vehicle diagnostics:")
            for vehicle_idx in range(len(vehicle_list)):
                start_index = routing.Start(vehicle_idx)
                end_index = routing.End(vehicle_idx)
                print(f"  Vehicle {vehicle_idx}: start_node={manager.IndexToNode(start_index)}, end_node={manager.IndexToNode(end_index)}")
            
            # Check pickup-delivery pairs
            print(f"\n📦 Pickup-delivery pairs: {pickup_delivery_count if 'pickup_delivery_count' in locals() else 'N/A'}")
            
            return None, "Failed", applied_constraints
            
    def _add_capacity_constraints(self, routing, manager, location_list, vehicle_list):
        """Add capacity constraints step by step."""
        print("\n📦 Adding capacity constraints...")
        
        # Create demand callback based on ride requests (like enhanced version)
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            location = location_list[from_node]
            
            demand = 0
            pickups = []
            dropoffs = []
            
            # Use only active ride requests
            requests_to_process = self._active_ride_requests if hasattr(self, '_active_ride_requests') and self._active_ride_requests else []
            for req in requests_to_process:
                if hasattr(req, 'pickup_location') and hasattr(req, 'dropoff_location') and hasattr(req, 'passengers'):
                    if req.pickup_location == location['id']:
                        demand += int(req.passengers)  # passengers = cargo weight in kg
                        pickups.append(f"+{req.passengers}kg ({getattr(req, 'id', 'unknown')})")
                    elif req.dropoff_location == location['id']:
                        demand -= int(req.passengers)  # passengers = cargo weight in kg
                        dropoffs.append(f"-{req.passengers}kg ({getattr(req, 'id', 'unknown')})")
            
            # Debug capacity callback for depot and problematic locations (limited output)
            if demand != 0 and location['id'] in ['depot_1'] and demand > 1000:
                print(f"  🔍 CAPACITY DEBUG: {location['id']} demand={demand}kg")
                if pickups:
                    print(f"    Pickups: {len(pickups)} requests totaling {demand}kg")
                if dropoffs:
                    print(f"    Dropoffs: {len(dropoffs)} requests totaling {-demand}kg")
            
            return demand
        
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        
        # Get vehicle capacities
        # Debug: Print vehicle order and capacities
        print("🔍 Vehicle index mapping:")
        for i, vehicle in enumerate(vehicle_list):
            print(f"  Index {i}: {vehicle['id']} -> {vehicle['capacity']}kg")

        vehicle_capacities = [int(vehicle.get('capacity', 0)) for vehicle in vehicle_list]
        print(f"  Final capacity array: {vehicle_capacities}")
        
        # Add dimension
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,  # Demand callback
            0,  # No slack
            vehicle_capacities,  # Vehicle capacity array
            True,  # Capacity cumulates from start of route
            'Capacity'  # Dimension name (required)
        )
        print("✅ Capacity constraints added")
        
    def _add_pickup_delivery_constraints(self, routing, manager, location_list, vehicle_list):
        """Add pickup and delivery constraints using ride requests."""
        print("\n🔄 Adding pickup-delivery constraints...")
        
        pickup_delivery_count = 0
        processed_pairs = set()  # Track processed pairs to avoid duplicates
        processed_dropoffs = set()  # Track dropoffs to avoid conflicts
        
        # Process ride requests to create pickup-delivery pairs
        print(f"  Ride requests type: {type(self.ride_requests)}")
        print(f"  Number of ride requests: {len(self.ride_requests)}")
        
        # Handle both dict and list formats for ride_requests
        if isinstance(self.ride_requests, dict):
            requests_to_process = self.ride_requests.items()
        elif isinstance(self.ride_requests, list):
            # If it's a list, create enumerate to get index and request
            requests_to_process = enumerate(self.ride_requests)
        else:
            print(f"  ⚠️ Unexpected ride_requests type: {type(self.ride_requests)}")
            return 0
        
        # First, identify and resolve conflicts by prioritizing certain pickup locations
        conflict_resolution = {}
        skipped_requests_by_dropoff = {}  # Track skipped requests for diagnostics
        for request_id, request in requests_to_process:
            dropoff_location = request.dropoff_location
            pickup_location = request.pickup_location
            
            if dropoff_location not in conflict_resolution:
                conflict_resolution[dropoff_location] = []
            conflict_resolution[dropoff_location].append((request_id, request))
        
        print(f"  📊 Conflict analysis:")
        for dropoff, requests in conflict_resolution.items():
            if len(requests) > 1:
                print(f"    - {dropoff}: {len(requests)} competing pickups")
                # Prioritize depot_bay pickups over regular pickups
                depot_requests = [r for r in requests if 'depot_bay' in r[1].pickup_location]
                regular_requests = [r for r in requests if 'depot_bay' not in r[1].pickup_location]
                skipped = []
                if depot_requests:
                    chosen = depot_requests[0]  # Choose first depot bay request
                    print(f"      → Prioritizing depot pickup: {chosen[1].pickup_location} → {dropoff}")
                    conflict_resolution[dropoff] = [chosen]
                    # All other requests are skipped
                    skipped = [r for r in requests if r != chosen]
                elif regular_requests:
                    chosen = regular_requests[0]  # Choose first regular pickup
                    print(f"      → Using first pickup: {chosen[1].pickup_location} → {dropoff}")
                    conflict_resolution[dropoff] = [chosen]
                    skipped = [r for r in requests if r != chosen]
                # Print skipped requests for this dropoff
                if skipped:
                    print(f"      Skipped requests for {dropoff}:")
                    for skip_id, skip_req in skipped:
                        print(f"        - {skip_req.pickup_location} → {dropoff} (weight: {getattr(skip_req, 'passengers', '?')}kg)")
                skipped_requests_by_dropoff[dropoff] = skipped
        
        # Now add the resolved pickup-delivery pairs
        for dropoff_location, requests in conflict_resolution.items():
            if len(requests) != 1:
                continue
            
            request_id, request = requests[0]
            pickup_location = request.pickup_location
            
            # Create unique pair identifier
            pair_id = f"{pickup_location}→{dropoff_location}"
            if pair_id in processed_pairs:
                print(f"    ⚠️ Skipping duplicate pair: {pair_id}")
                continue
            
            # Find indices of pickup and dropoff locations
            pickup_idx = None
            dropoff_idx = None
            
            for i, location in enumerate(location_list):
                if location['id'] == pickup_location:
                    pickup_idx = i
                elif location['id'] == dropoff_location:
                    dropoff_idx = i
            
            if pickup_idx is None:
                print(f"    ⚠️ Skipping request {request_id}: pickup location {pickup_location} not found")
                continue
                
            if dropoff_idx is None:
                print(f"    ⚠️ Skipping request {request_id}: dropoff location {dropoff_location} not found")
                continue
            
            # Add pickup-delivery pair
            pickup_index = manager.NodeToIndex(pickup_idx)
            dropoff_index = manager.NodeToIndex(dropoff_idx)
            
            print(f"    Adding pickup-delivery pair: {pickup_location} → {dropoff_location} (weight: {request.passengers}kg)")

            try:
                routing.AddPickupAndDelivery(pickup_index, dropoff_index)
                
                # Ensure the same vehicle handles both pickup and delivery
                routing.solver().Add(routing.VehicleVar(pickup_index) == routing.VehicleVar(dropoff_index))
                
                processed_pairs.add(pair_id)
                processed_dropoffs.add(dropoff_location)
                pickup_delivery_count += 1
                self._active_ride_requests.append(request)  # Track only successfully added requests
            except Exception as e:
                print(f"    ❌ Failed to add pickup-delivery pair {pair_id}: {str(e)}")
                continue
        
        print(f"✅ Added {pickup_delivery_count} pickup-delivery pairs (conflicts resolved)")
        
        # Additional diagnostics
        print("\n📊 Final pickup-delivery diagnostics:")
        if isinstance(self.ride_requests, list):
            print(f"  - Total ride requests: {len(self.ride_requests)}")
            print(f"  - Pickup-delivery pairs created: {pickup_delivery_count}")
            print(f"  - Unique dropoffs used: {len(processed_dropoffs)}")
            
            # Show which requests were skipped
            all_dropoffs = [request.dropoff_location for request in self.ride_requests]
            skipped_requests = len(self.ride_requests) - pickup_delivery_count
            if skipped_requests > 0:
                print(f"  - Requests skipped due to conflicts: {skipped_requests}")
        
        return pickup_delivery_count
        
    def _add_time_window_constraints(self, routing, manager, location_list, vehicle_list):
        """Add time window constraints."""
        print("\n⏰ Adding time window constraints...")
          # Create time callback
        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            from_loc = location_list[from_node]
            
            # Fallback to simple calculation
            to_loc = location_list[to_node]
            from_x = from_loc.get('x', 0)
            from_y = from_loc.get('y', 0)
            to_x = to_loc.get('x', 0)
            to_y = to_loc.get('y', 0)
            
            # Calculate Euclidean distance
            distance = ((to_x - from_x) ** 2 + (to_y - from_y) ** 2) ** 0.5
            
            # Convert coordinate distance to kilometers 
            # Using a smaller scaling factor to avoid overly long travel times
            distance_km = distance * 111  # Reduced from 111 to make travel times more reasonable
            
            # Calculate travel time based on 80 km/h average speed
            travel_time_hours = distance_km / 70.0
            travel_time_minutes = travel_time_hours * 60
            
            # Add service time at the "from" location
            service_time = from_loc.get('service_time', 0)
            
            total_time = travel_time_minutes + service_time
            
            return int(total_time)
        
        time_callback_index = routing.RegisterTransitCallback(time_callback)        # Add time dimension
        max_time = 1440  # 24 hours in minutes for absolute time
        
        print(f"  Setting maximum absolute time to: {max_time} minutes ({max_time/60:.1f} hours)")
        
        routing.AddDimension(
            time_callback_index,            120,            max_time,            True,            'Time'        )
        
        time_dimension = routing.GetDimensionOrDie('Time')
        
        # Set individual vehicle time limits based on their max_time attribute
        print("  Setting individual vehicle time limits:")
        
        for vehicle_idx in range(len(vehicle_list)):
            vehicle = vehicle_list[vehicle_idx]
            vehicle_max_time = vehicle.get('max_time', 9 * 60)  # Default to 9 hours if not specified
            
            print(f"    Vehicle {vehicle['id']}: max time = {vehicle_max_time} minutes ({vehicle_max_time/60:.1f} hours)")
            
            # Set upper bound for vehicle span (total route duration)
            time_dimension.SetSpanUpperBoundForVehicle(vehicle_max_time, vehicle_idx)
            
            # Also set time window constraints for vehicle start nodes if needed
            start_index = routing.Start(vehicle_idx)
            # Allow vehicles to start anytime during the day
            time_dimension.CumulVar(start_index).SetRange(0, 1440)
        
        # Add variables to be minimized by finalizer (like in the reference code)
        for vehicle_idx in range(len(vehicle_list)):
            routing.AddVariableMinimizedByFinalizer(
                time_dimension.CumulVar(routing.Start(vehicle_idx)))
            routing.AddVariableMinimizedByFinalizer(
                time_dimension.CumulVar(routing.End(vehicle_idx)))
        
        print("  ✅ Vehicle time span constraints added (individual limits per vehicle)")
        
        # Apply time window constraints to all locations with time windows
        locations_with_time_windows = 0
        for idx, loc in enumerate(location_list):
            # Check for time_window tuple first (new format)
            time_window = loc.get('time_window', None)
            if time_window is not None and len(time_window) == 2:
                tw_start, tw_end = time_window
            else:
                # Fallback to individual fields (old format)
                tw_start = loc.get('time_window_start', None)
                tw_end = loc.get('time_window_end', None)
            
            if tw_start is not None and tw_end is not None:
                index = manager.NodeToIndex(idx)
                time_dimension.CumulVar(index).SetRange(int(tw_start), int(tw_end))
                locations_with_time_windows += 1
                print(f"    Location {loc['id']}: time window [{tw_start}-{tw_end}]")
        print(f"✅ Time window constraints added for {locations_with_time_windows} locations")
        
    def _validate_capacity_constraints(self, routes, vehicle_list, vehicle_idx_to_vehicle=None):
        """Validate that no vehicle exceeds its capacity at any point. Uses robust mapping."""
        print("\n🔍 CAPACITY CONSTRAINT VALIDATION:")
        print("-" * 50)
        capacity_violations = []
        total_violations = 0
        # Use OR-Tools vehicle index directly for reporting
        for vehicle_idx, vehicle in (vehicle_idx_to_vehicle.items() if vehicle_idx_to_vehicle else enumerate(vehicle_list)):
            vehicle_id = vehicle['id']
            vehicle_capacity = vehicle.get('capacity', 0)
            route_data = routes.get(vehicle_id, None)
            print(f"  Vehicle {vehicle_id} (capacity: {vehicle_capacity}kg):")
            if not route_data:
                print(f"    No route assigned.")
                continue
            route = route_data['route']
            max_load_reached = 0
            load_violations = []
            for i, stop in enumerate(route):
                load = stop.get('load', 0)
                max_load_reached = max(max_load_reached, load)
                if load > vehicle_capacity:
                    load_violations.append({
                        'stop_index': i,
                        'location': stop['location_id'],
                        'load': load,
                        'capacity': vehicle_capacity,
                        'excess': load - vehicle_capacity
                    })
                    total_violations += 1
            if load_violations:
                print(f"    ❌ CAPACITY VIOLATIONS FOUND:")
                for violation in load_violations:
                    print(f"      Stop {violation['stop_index']+1} ({violation['location']}): "
                          f"{violation['load']}kg > {violation['capacity']}kg "
                          f"(excess: {violation['excess']}kg)")
                capacity_violations.append({
                    'vehicle_id': vehicle_id,
                    'vehicle_capacity': vehicle_capacity,
                    'max_load': max_load_reached,
                    'violations': load_violations
                })
            else:
                print(f"    ✅ No capacity violations (max load: {max_load_reached}kg)")
        if capacity_violations:
            print(f"\n🚨 CAPACITY CONSTRAINT VIOLATIONS SUMMARY:")
            print(f"  Total violations: {total_violations}")
            print(f"  Vehicles with violations: {len(capacity_violations)}")
            for violation in capacity_violations:
                print(f"    {violation['vehicle_id']}: max load {violation['max_load']}kg "
                      f"(capacity: {violation['vehicle_capacity']}kg)")
            return False
        else:
            print(f"\n✅ ALL CAPACITY CONSTRAINTS SATISFIED!")
            return True

    def _validate_pickup_delivery_constraints(self, routes):
        """Validate that pickup-delivery pairs are handled by the same vehicle and in correct order."""
        print("\n🔍 PICKUP-DELIVERY CONSTRAINT VALIDATION:")
        print("-" * 50)
        
        if not hasattr(self, 'ride_requests') or not self.ride_requests:
            print("   ℹ️ No ride requests to validate")
            return True
        
        requests_to_validate = []
        if isinstance(self.ride_requests, dict):
            requests_to_validate = list(self.ride_requests.values())
        elif isinstance(self.ride_requests, list):
            requests_to_validate = self.ride_requests
        
        violations = []
        
        for req in requests_to_validate:
            if hasattr(req, 'pickup_location') and hasattr(req, 'dropoff_location'):
                req_id = getattr(req, 'id', 'unknown')
                pickup_loc = req.pickup_location
                dropoff_loc = req.dropoff_location
                
                # Find which vehicles handle pickup and dropoff
                pickup_vehicle = None
                dropoff_vehicle = None
                pickup_position = None
                dropoff_position = None
                
                for vehicle_id, route_data in routes.items():
                    route = route_data['route']
                    for i, stop in enumerate(route):
                        if stop['location_id'] == pickup_loc:
                            pickup_vehicle = vehicle_id
                            pickup_position = i
                        elif stop['location_id'] == dropoff_loc:
                            dropoff_vehicle = vehicle_id
                            dropoff_position = i
                
                # Validate constraints
                if pickup_vehicle is None:
                    violations.append(f"{req_id}: Pickup location {pickup_loc} not found in any route")
                elif dropoff_vehicle is None:
                    violations.append(f"{req_id}: Dropoff location {dropoff_loc} not found in any route")
                elif pickup_vehicle != dropoff_vehicle:
                    violations.append(f"{req_id}: Pickup ({pickup_vehicle}) and dropoff ({dropoff_vehicle}) handled by different vehicles")
                elif pickup_position >= dropoff_position:
                    violations.append(f"{req_id}: Pickup (pos {pickup_position}) occurs after dropoff (pos {dropoff_position})")
                else:
                    print(f"   ✅ {req_id}: Valid - {pickup_vehicle} handles pickup→dropoff (positions {pickup_position}→{dropoff_position})")
        
        if violations:
            print(f"\n🚨 PICKUP-DELIVERY VIOLATIONS:")
            for violation in violations:
                print(f"   ❌ {violation}")
            return False
        else:
            print(f"\n✅ ALL PICKUP-DELIVERY CONSTRAINTS SATISFIED!")
            return True

    def _validate_time_window_constraints(self, routes):
        """Validate that all locations are visited within their time windows."""
        print("\n🔍 TIME WINDOW CONSTRAINT VALIDATION:")
        print("-" * 50)
        
        violations = []
        total_stops = 0
        
        for vehicle_id, route_data in routes.items():
            route = route_data['route']
            print(f"   Vehicle {vehicle_id}:")
            
            for stop in route:
                total_stops += 1
                location_id = stop['location_id']
                arrival_time = stop.get('arrival_time', 0)
                
                # Find the location's time window
                location_tw = None
                for loc in self.locations:
                    if loc['id'] == location_id:
                        location_tw = loc.get('time_window', (0, 1440))
                        break
                
                if location_tw:
                    tw_start, tw_end = location_tw
                    
                    if arrival_time < tw_start:
                        violations.append(f"{vehicle_id}: {location_id} arrived at {arrival_time}min (before window {tw_start}-{tw_end})")
                        print(f"     ❌ {location_id}: arrived {arrival_time}min < {tw_start}min (too early)")
                    elif arrival_time > tw_end:
                        violations.append(f"{vehicle_id}: {location_id} arrived at {arrival_time}min (after window {tw_start}-{tw_end})")
                        print(f"     ❌ {location_id}: arrived {arrival_time}min > {tw_end}min (too late)")
                    else:
                        print(f"     ✅ {location_id}: arrived {arrival_time}min within [{tw_start}-{tw_end}]")
        
        print(f"\n   📊 Validated {total_stops} stops across {len(routes)} vehicles")
        
        if violations:
            print(f"\n🚨 TIME WINDOW VIOLATIONS:")
            for violation in violations:
                print(f"   ❌ {violation}")
            return False
        else:
            print(f"\n✅ ALL TIME WINDOW CONSTRAINTS SATISFIED!")
            return True

    def _extract_solution(self, routing, manager, solution, location_list, vehicle_list, constraint_level: str = "none", vehicle_idx_to_vehicle=None) -> Dict:
        """Extract and format the solution. Uses robust vehicle mapping."""
        print("\n📋 Extracting solution...")
        routes = {}
        total_distance = 0
        total_time = 0
        has_capacity = constraint_level in ["capacity", "full"]
        has_time = constraint_level in ["time_windows", "full"]
        time_dimension = None
        if has_time:
            try:
                time_dimension = routing.GetDimensionOrDie('Time')
                print("  ✅ Time dimension found and will be used for arrival times")
            except:
                has_time = False
                print("  ⚠️ Time dimension not found, arrival times will be 0")
        for vehicle_idx in range(len(vehicle_list)):
            vehicle = vehicle_idx_to_vehicle[vehicle_idx] if vehicle_idx_to_vehicle else vehicle_list[vehicle_idx]
            route = []
            index = routing.Start(vehicle_idx)
            route_distance = 0
            route_time = 0
            manual_load = 0
            max_manual_load = 0
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                location = location_list[node_index]
                arrival_time = 0
                load = 0
                if has_time and time_dimension:
                    try:
                        arrival_time = solution.Value(time_dimension.CumulVar(index))
                        if len(route) <= 3:
                            print(f"    Vehicle {vehicle_idx}, stop {location['id']}: arrival_time = {arrival_time} minutes")
                    except Exception as e:
                        print(f"    ⚠️ Failed to get arrival time for {location['id']}: {str(e)}")
                        arrival_time = 0
                if has_capacity:
                    try:
                        capacity_dimension = routing.GetDimensionOrDie('Capacity')
                        load = solution.Value(capacity_dimension.CumulVar(index))
                        print(f"      [DEBUG] Vehicle {vehicle_idx}, stop {location['id']}: load (OR-Tools) = {load}")
                    except Exception as e:
                        print(f"      [DEBUG] Failed to get load for {location['id']}: {str(e)}")
                        load = 0
                if has_capacity and hasattr(self, '_active_ride_requests') and self._active_ride_requests:
                    for req in self._active_ride_requests:
                        if hasattr(req, 'pickup_location') and hasattr(req, 'dropoff_location') and hasattr(req, 'passengers'):
                            if req.pickup_location == location['id']:
                                manual_load += int(req.passengers)
                            elif req.dropoff_location == location['id']:
                                manual_load -= int(req.passengers)
                    max_manual_load = max(max_manual_load, manual_load)
                route.append({
                    'location_id': location['id'],
                    'location_name': location.get('address', location['id']),
                    'coordinates': (location.get('x', 0), location.get('y', 0)),
                    'arrival_time': arrival_time,
                    'load': manual_load if has_capacity else load
                })
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                if not routing.IsEnd(index):
                    arc_cost_meters = routing.GetArcCostForVehicle(previous_index, index, vehicle_idx)
                    arc_cost_km = arc_cost_meters / 1000.0
                    route_distance += arc_cost_km
            # Add final location (end depot)
            if not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                location = location_list[node_index]
                arrival_time = 0
                load = 0
                if has_time and time_dimension:
                    try:
                        arrival_time = solution.Value(time_dimension.CumulVar(index))
                        print(f"    Vehicle {vehicle_idx}, final stop {location['id']}: arrival_time = {arrival_time} minutes")
                    except Exception as e:
                        print(f"    ⚠️ Failed to get final arrival time for {location['id']}: {str(e)}")
                        arrival_time = 0
                if has_capacity:
                    try:
                        capacity_dimension = routing.GetDimensionOrDie('Capacity')
                        load = solution.Value(capacity_dimension.CumulVar(index))
                    except:
                        load = 0
                if has_capacity and hasattr(self, '_active_ride_requests') and self._active_ride_requests:
                    for req in self._active_ride_requests:
                        if hasattr(req, 'pickup_location') and hasattr(req, 'dropoff_location') and hasattr(req, 'passengers'):
                            if req.pickup_location == location['id']:
                                manual_load += int(req.passengers)
                            elif req.dropoff_location == location['id']:
                                manual_load -= int(req.passengers)
                    max_manual_load = max(max_manual_load, manual_load)
                route.append({
                    'location_id': location['id'],
                    'location_name': location.get('address', location['id']),
                    'coordinates': (location.get('x', 0), location.get('y', 0)),
                    'arrival_time': arrival_time,
                    'load': manual_load if has_capacity else load
                })
            else:
                final_node_index = manager.IndexToNode(routing.End(vehicle_idx))
                final_location = location_list[final_node_index]
                if has_capacity and hasattr(self, '_active_ride_requests') and self._active_ride_requests:
                    final_manual_load = 0
                    for req in self._active_ride_requests:
                        if hasattr(req, 'pickup_location') and hasattr(req, 'dropoff_location') and hasattr(req, 'passengers'):
                            if req.pickup_location == final_location['id']:
                                final_manual_load += int(req.passengers)
                            elif req.dropoff_location == final_location['id']:
                                final_manual_load -= int(req.passengers)
                    if len(route) == 0 or route[-1]['location_id'] != final_location['id']:
                        route.append({
                            'location_id': final_location['id'],
                            'location_name': final_location.get('address', final_location['id']),
                            'coordinates': (final_location.get('x', 0), final_location.get('y', 0)),
                            'arrival_time': 0,
                            'load': final_manual_load if has_capacity else 0
                        })
            if len(route) >= 2 and has_time:
                route_time = route[-1]['arrival_time'] - route[0]['arrival_time']
            routes[vehicle['id']] = {
                'route': route,
                'distance': route_distance,
                'time': route_time
            }
            # Calculate total service time for this vehicle
            total_service_time = 0
            for stop in route:
                location_id = stop['location_id']
                # Find the location in location_list to get service time
                for loc in location_list:
                    if loc['id'] == location_id:
                        total_service_time += loc.get('service_time', 0)
                        break
            
            # Calculate driving time (route_time - total_service_time)
            driving_time = max(0, route_time - total_service_time) if route_time > 0 else 0
            
            total_distance += route_distance
            total_time += route_time
            
            print(f"  Vehicle {vehicle['id']}: {len(route)} stops, distance: {route_distance:.1f} km")
            print(f"    ⏱️ Time breakdown: {route_time}min total = {driving_time}min driving + {total_service_time}min service")
            
            if has_capacity and len(route) > 1:
                print(f"    📦 Load tracking: max load reached = {max_manual_load}kg (capacity: {vehicle.get('capacity', 'N/A')}kg)")
                if max_manual_load > vehicle.get('capacity', 0):
                    print(f"    ⚠️ WARNING: Max load {max_manual_load}kg exceeds capacity {vehicle.get('capacity', 'N/A')}kg!")
                else:
                    print(f"    ✅ Load within capacity limits")
        for vehicle_idx, vehicle in vehicle_idx_to_vehicle.items():
            print(f"Vehicle {vehicle['id']} capacity: {vehicle.get('capacity', 'N/A')}")
        if self.ride_requests:
            print(f"\n📊 Ride requests summary:")
            total_pickups = 0
            total_dropoffs = 0
            ride_requests = self.ride_requests.values() if isinstance(self.ride_requests, dict) else self.ride_requests
            for req in ride_requests:
                if hasattr(req, 'pickup_location') and hasattr(req, 'dropoff_location') and hasattr(req, 'passengers'):
                    total_pickups += int(req.passengers)
                    total_dropoffs += int(req.passengers)
            print(f"  Total cargo to be picked up: {total_pickups}kg")
            print(f"  Total cargo to be delivered: {total_dropoffs}kg")
            print(f"  Net cargo change: {total_pickups - total_dropoffs}kg (should be 0)")
        # Comprehensive solution validation
        print(f"\n🔍 SOLUTION VALIDATION REPORT")
        print("=" * 50)
        
        validation_results = {}
        
        # 1. Capacity validation
        if constraint_level in ["capacity", "pickup_delivery", "full"]:
            capacity_valid = self._validate_capacity_constraints(routes, vehicle_list, vehicle_idx_to_vehicle)
            validation_results['capacity_valid'] = capacity_valid
            if not capacity_valid:
                print("⚠️ WARNING: Capacity constraint violations detected in solution!")
        else:
            print("ℹ️ Capacity constraints not active - skipping capacity validation")
            validation_results['capacity_valid'] = True
        
        # 2. Pickup-delivery validation
        if constraint_level in ["pickup_delivery", "full"]:
            pd_valid = self._validate_pickup_delivery_constraints(routes)
            validation_results['pickup_delivery_valid'] = pd_valid
        else:
            print("ℹ️ Pickup-delivery constraints not active - skipping P-D validation")
            validation_results['pickup_delivery_valid'] = True
        
        # 3. Time window validation
        if constraint_level in ["time_windows", "full"]:
            tw_valid = self._validate_time_window_constraints(routes)
            validation_results['time_windows_valid'] = tw_valid
        else:
            print("ℹ️ Time window constraints not active - skipping time validation")
            validation_results['time_windows_valid'] = True
        
        # 4. Overall validation summary
        all_valid = all(validation_results.values())
        print(f"\n📊 VALIDATION SUMMARY:")
        for constraint_type, is_valid in validation_results.items():
            status = "✅ VALID" if is_valid else "❌ INVALID"
            print(f"   {constraint_type.replace('_', ' ').title()}: {status}")
        
        if all_valid:
            print(f"\n🎉 ALL CONSTRAINTS SATISFIED! Solution is valid.")
        else:
            print(f"\n⚠️ CONSTRAINT VIOLATIONS DETECTED! Review solution carefully.")
        
        print("=" * 50)
        
        return {
            'status': 'success',
            'routes': routes,
            'total_distance': total_distance,
            'total_time': total_time,
            'objective_value': solution.ObjectiveValue(),
            'validation_results': validation_results
        }

    def plot_solution(self, result, title="VRP Solution"):
        """Plot the solution with routes and pickup-delivery pairs."""
        if not result or 'routes' not in result:
            print("No solution to plot")
            return
            
        plt.figure(figsize=(12, 8))
        
        # Colors for different vehicles
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        # Plot all locations first
        for loc in self.locations:
            x, y = loc['x'], loc['y']
            
            # Different markers for different location types
            if 'depot' in loc['id'].lower():
                plt.plot(x, y, 's', color='black', markersize=12, label='Depot' if 'Depot' not in plt.gca().get_legend_handles_labels()[1] else "")
            elif 'pickup' in loc['id'].lower():
                plt.plot(x, y, '^', color='green', markersize=8, alpha=0.7, label='Pickup' if 'Pickup' not in plt.gca().get_legend_handles_labels()[1] else "")
            elif 'dropoff' in loc['id'].lower():
                plt.plot(x, y, 'v', color='red', markersize=8, alpha=0.7, label='Dropoff' if 'Dropoff' not in plt.gca().get_legend_handles_labels()[1] else "")
            else:
                plt.plot(x, y, 'o', color='gray', markersize=6, alpha=0.5)
              # Add location ID as text
            plt.annotate(loc['id'], (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
        
        # Plot pickup-delivery connections (colored arrows)
        if isinstance(self.ride_requests, list):
            for i, request in enumerate(self.ride_requests):
                pickup_loc = None
                dropoff_loc = None
                
                for loc in self.locations:
                    if loc['id'] == request.pickup_location:
                        pickup_loc = loc
                    elif loc['id'] == request.dropoff_location:
                        dropoff_loc = loc
                
                if pickup_loc and dropoff_loc:
                    # Use different colors for different requests
                    arrow_color = colors[i % len(colors)]
                    plt.annotate('', xy=(dropoff_loc['x'], dropoff_loc['y']), 
                               xytext=(pickup_loc['x'], pickup_loc['y']),
                               arrowprops=dict(arrowstyle='->', color=arrow_color, 
                                             alpha=0.7, linewidth=2,
                                             connectionstyle="arc3,rad=0.1"),                               zorder=1)
        
        # Plot vehicle routes
        vehicle_idx = 0
        for vehicle_id, route_data in result['routes'].items():
            route = route_data['route']
            if len(route) > 1:  # Only plot if there are stops
                color = colors[vehicle_idx % len(colors)]
                
                # Extract coordinates
                route_x = [stop['coordinates'][0] for stop in route]
                route_y = [stop['coordinates'][1] for stop in route]
                
                # Plot route
                plt.plot(route_x, route_y, '-o', color=color, linewidth=2, 
                        markersize=4, label=f'Vehicle {vehicle_id}', alpha=0.8)
                
                # Add arrows to show direction
                for i in range(len(route_x)-1):
                    dx = route_x[i+1] - route_x[i]
                    dy = route_y[i+1] - route_y[i]
                    if abs(dx) > 0.001 or abs(dy) > 0.001:  # Only add arrow if there's movement
                        plt.annotate('', xy=(route_x[i+1], route_y[i+1]), 
                                   xytext=(route_x[i], route_y[i]),
                                   arrowprops=dict(arrowstyle='->', color=color, alpha=0.6))
                
                vehicle_idx += 1
        
        plt.title(f"{title}\nTotal Distance: {result['total_distance']:.1f} km, Objective: {result['objective_value']}")
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

    def _print_comprehensive_sanity_check(self, constraint_level: str):
        """Print comprehensive sanity check of the problem instance."""
        print("\n📊 COMPREHENSIVE CONSTRAINTS CHECK")
        print("=" * 60)
        
        # Basic counts
        num_locations = len(self.locations)
        num_vehicles = len(self.vehicles)
        num_requests = len(self.ride_requests) if hasattr(self, 'ride_requests') and self.ride_requests else 0
        
        print(f"📍 Problem Size:")
        print(f"   - Locations: {num_locations}")
        print(f"   - Vehicles: {num_vehicles}")
        print(f"   - Ride requests: {num_requests}")
        print(f"   - Constraint level: {constraint_level}")
        
        # Vehicle analysis
        print(f"\n🚛 VEHICLE ANALYSIS:")
        total_capacity = 0
        for i, vehicle in enumerate(self.vehicles):
            capacity = vehicle.get('capacity', 0)
            max_time = vehicle.get('max_time', 24 * 60)
            start_loc = vehicle.get('start_location', 'N/A')
            end_loc = vehicle.get('end_location', 'N/A')
            
            total_capacity += capacity
            
            print(f"   {vehicle['id']}: {capacity}kg capacity, {max_time}min max_time")
            print(f"      Start: {start_loc}, End: {end_loc}")
        
        print(f"   💼 Total fleet capacity: {total_capacity}kg")
        
        # Request analysis
        if hasattr(self, 'ride_requests') and self.ride_requests:
            print(f"\n📦 REQUEST ANALYSIS:")
            total_demand = 0
            requests_to_analyze = []
            
            # Handle both dict and list formats
            if isinstance(self.ride_requests, dict):
                requests_to_analyze = list(self.ride_requests.values())
            elif isinstance(self.ride_requests, list):
                requests_to_analyze = self.ride_requests
            
            for req in requests_to_analyze:
                if hasattr(req, 'passengers') and hasattr(req, 'pickup_location') and hasattr(req, 'dropoff_location'):
                    cargo = int(req.passengers)
                    total_demand += cargo
                    req_id = getattr(req, 'id', 'unknown')
                    print(f"   {req_id}: {cargo}kg from {req.pickup_location} to {req.dropoff_location}")
            
            print(f"   📊 Total demand: {total_demand}kg")
            if total_capacity > 0:
                print(f"   📊 Capacity utilization: {total_demand/total_capacity*100:.1f}%")
                
                if total_demand > total_capacity:
                    print("   ⚠️ WARNING: Total demand exceeds total capacity!")
                    print("   💡 Note: This is OK if vehicles can make multiple trips")
        
        # Time window analysis
        print(f"\n⏰ TIME WINDOW ANALYSIS:")
        time_windowed_locations = 0
        earliest_start = float('inf')
        latest_end = 0
        service_times = []
        
        for location in self.locations:
            if 'time_window' in location and location['time_window']:
                time_windowed_locations += 1
                start, end = location['time_window']
                service_time = location.get('service_time', 0)
                
                earliest_start = min(earliest_start, start)
                latest_end = max(latest_end, end)
                service_times.append(service_time)
                
                print(f"   {location['id']}: [{start}-{end}] ({end-start}min window) +{service_time}min service")
        
        if time_windowed_locations > 0:
            print(f"   📊 {time_windowed_locations}/{num_locations} locations have time windows")
            print(f"   📊 Time span: {earliest_start} to {latest_end} ({latest_end - earliest_start}min)")
            if service_times:
                print(f"   📊 Service times: {min(service_times)}-{max(service_times)}min (avg: {sum(service_times)/len(service_times):.1f}min)")
        else:
            print("   ℹ️ No time windows found (all locations have 0-1440 range)")
        
        # Pickup-dropoff feasibility check
        if hasattr(self, 'ride_requests') and self.ride_requests:
            print(f"\n🔄 PICKUP-DROPOFF FEASIBILITY:")
            impossible_pairs = 0
            tight_pairs = 0
            
            requests_to_analyze = []
            if isinstance(self.ride_requests, dict):
                requests_to_analyze = list(self.ride_requests.values())
            elif isinstance(self.ride_requests, list):
                requests_to_analyze = self.ride_requests
            
            # Create location lookup
            location_lookup = {loc['id']: loc for loc in self.locations}
            
            for req in requests_to_analyze:
                if hasattr(req, 'pickup_location') and hasattr(req, 'dropoff_location'):
                    pickup_loc = location_lookup.get(req.pickup_location)
                    dropoff_loc = location_lookup.get(req.dropoff_location)
                    
                    if pickup_loc and dropoff_loc:
                        pickup_tw = pickup_loc.get('time_window', (0, 1440))
                        dropoff_tw = dropoff_loc.get('time_window', (0, 1440))
                        
                        pickup_start, pickup_end = pickup_tw
                        dropoff_start, dropoff_end = dropoff_tw
                        
                        # Check for time window intersection
                        intersection_start = max(pickup_start, dropoff_start)
                        intersection_end = min(pickup_end, dropoff_end)
                        intersection_duration = max(0, intersection_end - intersection_start)
                        
                        req_id = getattr(req, 'id', 'unknown')
                        
                        if intersection_duration == 0:
                            impossible_pairs += 1
                            print(f"   ❌ {req_id}: NO intersection - pickup [{pickup_start}-{pickup_end}], dropoff [{dropoff_start}-{dropoff_end}]")
                        elif intersection_duration < 30:
                            tight_pairs += 1
                            print(f"   ⚠️ {req_id}: tight intersection - only {intersection_duration}min overlap")
                        else:
                            print(f"   ✅ {req_id}: good intersection - {intersection_duration}min overlap")
            
            if impossible_pairs > 0:
                print(f"   🚨 {impossible_pairs} impossible pickup-dropoff pairs found!")
            elif tight_pairs > 0:
                print(f"   ⚠️ {tight_pairs} tight pickup-dropoff pairs found")
            else:
                print("   ✅ All pickup-dropoff pairs have feasible time window intersections")
        
        # Constraint-specific warnings
        print(f"\n🔧 CONSTRAINT-SPECIFIC ANALYSIS:")
        if constraint_level == "none":
            print("   ℹ️ Only distance minimization - no capacity, time, or pickup-delivery constraints")
        elif constraint_level == "capacity":
            print("   📦 Capacity constraints active - checking vehicle load limits")
            if total_capacity == 0:
                print("   ⚠️ WARNING: All vehicles have 0 capacity!")
        elif constraint_level == "pickup_delivery":
            print("   🔄 Pickup-delivery constraints active - ensuring same vehicle handles pairs")
            if num_requests == 0:
                print("   ⚠️ WARNING: No ride requests found for pickup-delivery constraints!")
        elif constraint_level == "time_windows":
            print("   ⏰ Time window constraints active - vehicles must respect arrival times")
            if time_windowed_locations == 0:
                print("   ℹ️ All locations have full-day time windows (0-1440)")
        elif constraint_level == "full":
            print("   🎯 ALL constraints active - capacity + pickup-delivery + time windows")
            
            # Check for potential conflicts
            conflicts = []
            if total_capacity == 0:
                conflicts.append("Zero total capacity")
            if num_requests == 0:
                conflicts.append("No ride requests")
            if impossible_pairs > 0:
                conflicts.append(f"{impossible_pairs} impossible pickup-dropoff pairs")
            
            if conflicts:
                print(f"   ⚠️ POTENTIAL CONFLICTS: {', '.join(conflicts)}")
            else:
                print("   ✅ No obvious constraint conflicts detected")
        
        print("=" * 60)

def test_moda_small_scenario():
    """Test the clean optimizer with MODA_small VRPPD scenario, twice with different vehicle orders."""
    print("🧪 Testing Clean VRP Optimizer with MODA_small scenario (order sensitivity test)")
    print("=" * 60)
    try:
        from vrp_scenarios import create_furgoni_scenario
    except ImportError:
        print("❌ Could not import vrp_scenarios. Make sure the file is available.")
        return
    scenario = create_furgoni_scenario()
    print(f"📊 Scenario details:")
    print(f"  - Locations: {len(scenario.locations)}")
    print(f"  - Vehicles: {len(scenario.vehicles)}")
    print(f"  - Ride requests: {len(scenario.ride_requests)}")

    # --- Get vehicle list from scenario ---
    vehicle_ids = list(scenario.vehicles.keys())
    vehicles_from_scenario = [scenario.vehicles[vid] for vid in vehicle_ids]
    # Convert to dicts for CleanVRPOptimizer
    vehicles_dicts = [{
        'id': v.id,
        'capacity': v.capacity,
        'start_location': v.depot_id,
        'end_location': v.depot_id,
        'max_time': getattr(v, 'max_time', 24 * 60)
    } for v in vehicles_from_scenario]

    # --- First run: original order ---
    print("\n================= RUN 1: Original vehicle order ================")
    print("RUN 1 vehicle order:", [v['id'] for v in vehicles_dicts])
    print("RUN 1 vehicle capacities:", [v['capacity'] for v in vehicles_dicts])
    optimizer1 = CleanVRPOptimizer(vehicles=vehicles_dicts, locations=None, vrp_instance=scenario)
    optimizer1.ride_requests = scenario.ride_requests
    result1, status1, applied_constraints1 = optimizer1.solve(constraint_level="full", verbose=False)
    print(f"\n=== RUN 1 RESULT ===")
    if result1:
        print(f"✅ SUCCESS - Status: {status1}")
        print(f"   Constraints applied: {applied_constraints1}")
        print(f"   Objective value: {result1['objective_value']}")
        print(f"   Total distance: {result1['total_distance']:.1f} km")
        for vehicle_id, route_data in result1['routes'].items():
            route = route_data['route']
            print(f"   Vehicle {vehicle_id}: {len(route)} stops, distance: {route_data['distance']:.1f} km")
        # Plot the solution
        optimizer1.plot_solution(result1, title="Furgoni VRP Solution")
    else:
        print(f"❌ FAILED - Status: {status1}")
        print(f"   Constraints applied: {applied_constraints1}")



def test_moda_inverted_scenario():
    """Test the clean optimizer with MODA_small VRPPD scenario, twice with different vehicle orders."""
    print("🧪 Testing Clean VRP Optimizer with MODA_small scenario (order sensitivity test)")
    print("=" * 60)
    try:
        from vrp_scenarios import create_moda_small_scenario
    except ImportError:
        print("❌ Could not import vrp_scenarios. Make sure the file is available.")
        return
    scenario = create_moda_small_scenario()
    print(f"📊 Scenario details:")
    print(f"  - Locations: {len(scenario.locations)}")
    print(f"  - Vehicles: {len(scenario.vehicles)}")
    print(f"  - Ride requests: {len(scenario.ride_requests)}")

    # --- Get vehicle list from scenario ---
    vehicle_ids = list(scenario.vehicles.keys())
    vehicles_from_scenario = [scenario.vehicles[vid] for vid in vehicle_ids]
    # Convert to dicts for CleanVRPOptimizer
    vehicles_dicts = [{
        'id': v.id,
        'capacity': v.capacity,
        'start_location': v.depot_id,
        'end_location': v.depot_id,
        'max_time': getattr(v, 'max_time', 24 * 60)
    } for v in vehicles_from_scenario]

    # --- First run: original order ---
    print("\n================= RUN 1: Original vehicle order ================")
    print("RUN 1 vehicle order:", [v['id'] for v in vehicles_dicts])
    print("RUN 1 vehicle capacities:", [v['capacity'] for v in vehicles_dicts])
    optimizer1 = CleanVRPOptimizer(vehicles=vehicles_dicts, locations=None, vrp_instance=scenario)
    optimizer1.ride_requests = scenario.ride_requests
    result1, status1, applied_constraints1 = optimizer1.solve(constraint_level="full", verbose=False)
    print(f"\n=== RUN 1 RESULT ===")
    if result1:
        print(f"✅ SUCCESS - Status: {status1}")
        print(f"   Constraints applied: {applied_constraints1}")
        print(f"   Objective value: {result1['objective_value']}")
        print(f"   Total distance: {result1['total_distance']:.1f} km")
        for vehicle_id, route_data in result1['routes'].items():
            route = route_data['route']
            print(f"   Vehicle {vehicle_id}: {len(route)} stops, distance: {route_data['distance']:.1f} km")
    else:
        print(f"❌ FAILED - Status: {status1}")
        print(f"   Constraints applied: {applied_constraints1}")

    # --- Second run: reversed order ---
    vehicles_dicts_reversed = list(reversed(vehicles_dicts))
    print("\n================= RUN 2: Reversed vehicle order ================")
    print("RUN 2 vehicle order:", [v['id'] for v in vehicles_dicts_reversed])
    print("RUN 2 vehicle capacities:", [v['capacity'] for v in vehicles_dicts_reversed])
    optimizer2 = CleanVRPOptimizer(vehicles=vehicles_dicts_reversed, locations=None, vrp_instance=scenario)
    optimizer2.ride_requests = scenario.ride_requests
    result2, status2, applied_constraints2 = optimizer2.solve(constraint_level="full", verbose=False)
    print(f"\n=== RUN 2 RESULT ===")
    if result2:
        print(f"✅ SUCCESS - Status: {status2}")
        print(f"   Constraints applied: {applied_constraints2}")
        print(f"   Objective value: {result2['objective_value']}")
        print(f"   Total distance: {result2['total_distance']:.1f} km")
        for vehicle_id, route_data in result2['routes'].items():
            route = route_data['route']
            print(f"   Vehicle {vehicle_id}: {len(route)} stops, distance: {route_data['distance']:.1f} km")
    else:
        print(f"❌ FAILED - Status: {status2}")
        print(f"   Constraints applied: {applied_constraints2}")



def test_constraint_levels():
    """Test the clean optimizer with MODA_small VRPPD scenario, twice with different vehicle orders."""
    print("🧪 Testing Clean VRP Optimizer with MODA_small scenario (order sensitivity test)")
    print("=" * 60)
    try:
        from vrp_scenarios import create_furgoni_scenario
    except ImportError:
        print("❌ Could not import vrp_scenarios. Make sure the file is available.")
        return
    scenario = create_furgoni_scenario()
    print(f"📊 Scenario details:")
    print(f"  - Locations: {len(scenario.locations)}")
    print(f"  - Vehicles: {len(scenario.vehicles)}")
    print(f"  - Ride requests: {len(scenario.ride_requests)}")

    # --- Get vehicle list from scenario ---
    vehicle_ids = list(scenario.vehicles.keys())
    vehicles_from_scenario = [scenario.vehicles[vid] for vid in vehicle_ids]
    # Convert to dicts for CleanVRPOptimizer
    vehicles_dicts = [{
        'id': v.id,
        'capacity': v.capacity,
        'start_location': v.depot_id,
        'end_location': v.depot_id,
        'max_time': getattr(v, 'max_time', 24 * 60)
    } for v in vehicles_from_scenario]

    levels = ["none",  "pickup_delivery", "time_windows", "capacity", "full"]
    for i, level in enumerate(levels):

        # --- First run: original order ---
        print(f"\n================= RUN {i+1}: Level: {level} ================")
        print("RUN 1 vehicle order:", [v['id'] for v in vehicles_dicts])
        print("RUN 1 vehicle capacities:", [v['capacity'] for v in vehicles_dicts])
        optimizer1 = CleanVRPOptimizer(vehicles=vehicles_dicts, locations=None, vrp_instance=scenario)
        optimizer1.ride_requests = scenario.ride_requests
        result1, status1, applied_constraints1 = optimizer1.solve(constraint_level=level, verbose=False)
        print(f"\n=== RUN {i+1} RESULT ===")
        if result1:
            print(f"✅ SUCCESS - Status: {status1}")
            print(f"   Constraints applied: {applied_constraints1}")
            print(f"   Objective value: {result1['objective_value']}")
            print(f"   Total distance: {result1['total_distance']:.1f} km")
            for vehicle_id, route_data in result1['routes'].items():
                route = route_data['route']
                print(f"   Vehicle {vehicle_id}: {len(route)} stops, distance: {route_data['distance']:.1f} km")
        else:
            print(f"❌ FAILED - Status: {status1}")
            print(f"   Constraints applied: {applied_constraints1}")

def test_moda_first_scenario():
    """Test the clean optimizer with MODA_first VRPPD scenario."""
    print("🧪 Testing Clean VRP Optimizer with MODA_first scenario")
    print("=" * 60)
    try:
        from vrp_scenarios import create_moda_first_scenario
    except ImportError:
        print("❌ Could not import vrp_scenarios. Make sure the file is available.")
        return
    scenario = create_moda_first_scenario()
    print(f"📊 Scenario details:")
    print(f"  - Locations: {len(scenario.locations)}")
    print(f"  - Vehicles: {len(scenario.vehicles)}")
    print(f"  - Ride requests: {len(scenario.ride_requests)}")

    # --- Get vehicle list from scenario ---
    vehicle_ids = list(scenario.vehicles.keys())
    vehicles_from_scenario = [scenario.vehicles[vid] for vid in vehicle_ids]
    # Convert to dicts for CleanVRPOptimizer
    vehicles_dicts = [{
        'id': v.id,
        'capacity': v.capacity,
        'start_location': v.depot_id,
        'end_location': v.depot_id,
        'max_time': getattr(v, 'max_time', 24 * 60)
    } for v in vehicles_from_scenario]

    print("\n================= RUN: MODA_first scenario ================")
    print("Vehicle order:", [v['id'] for v in vehicles_dicts])
    print("Vehicle capacities:", [v['capacity'] for v in vehicles_dicts])
    optimizer = CleanVRPOptimizer(vehicles=vehicles_dicts, locations=None, vrp_instance=scenario)
    optimizer.ride_requests = scenario.ride_requests
    result, status, applied_constraints = optimizer.solve(constraint_level="full", verbose=False)
    print(f"\n=== MODA_first RESULT ===")
    if result:
        print(f"✅ SUCCESS - Status: {status}")
        print(f"   Constraints applied: {applied_constraints}")
        print(f"   Objective value: {result['objective_value']}")
        print(f"   Total distance: {result['total_distance']:.1f} km")
        for vehicle_id, route_data in result['routes'].items():
            route = route_data['route']
            print(f"   Vehicle {vehicle_id}: {route_data}")
    else:
        print(f"❌ FAILED - Status: {status}")
        print(f"   Constraints applied: {applied_constraints}")
    """Test the clean optimizer with different constraint levels."""
    import sys
    import os
    
    print("🧪 Testing Clean VRP Optimizer with different constraint levels")
    print("=" * 60)
    
    # Create a simple test scenario
    # 1. Define the Vehicle List
    vehicle_list = [
        {"id": 0, "capacity": 15, "start_location": "A", "end_location": "A"},
        {"id": 1, "capacity": 15, "start_location": "A", "end_location": "A"},
    ]

    # 2. Define the Location List
    location_list = [
        {"id": "A", "demand": 0, "time_window": (0, 0)},  # Depot
        {"id": "B", "demand": -1, "time_window": (7, 12), "pickup": "C"},
        {"id": "C", "demand": 1, "time_window": (7, 12), "delivery": "B"},
        {"id": "D", "demand": 2, "time_window": (8, 15)},
        {"id": "E", "demand": 1, "time_window": (9, 14)},
    ]
    
    optimizer = CleanVRPOptimizer(
        vehicles=vehicle_list,
        locations=location_list,
        distance_matrix_provider="google"
    )
    
    # Test each constraint level
    levels = ["none", "capacity", "pickup_delivery", "time_windows", "full"]
    
    for level in levels:
        print(f"\n{'='*20} TESTING LEVEL: {level.upper()} {'='*20}")
        
        try:
            solution = optimizer.solve(constraint_level=level)
            
            if solution:
                print(f"✅ SUCCESS - {level} constraints work!")
                print(f"   Objective value: {solution['objective_value']}")
                print(f"   Total distance: {solution['total_distance']}")
                
                # Show first few routes
                for vehicle_id, route_data in list(solution['routes'].items())[:2]:
                    route = route_data['route']
                    print(f"   {vehicle_id}: {len(route)} stops")
                    for stop in route[:3]:
                        print(f"     - {stop['location_id']} (load: {stop['load']})")
                    if len(route) > 3:
                        print(f"     - ... and {len(route)-3} more stops")
            else:
                print(f"❌ FAILED - {level} constraints cause infeasibility!")
                print("   🛑 STOPPING HERE to debug")
                break
                
        except Exception as e:
            print(f"💥 ERROR at {level} level: {str(e)}")
            print("   🛑 STOPPING HERE to debug")
            break


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    #test_moda_inverted_scenario()
    test_moda_small_scenario()
