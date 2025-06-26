"""
Clean VRP Optimizer - Built step by step to debug constraint issues
"""
import logging
import time
from typing import Dict, List, Tuple, Optional
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import matplotlib.pyplot as plt
import numpy as np

try:
    from hybrid_travel_calculator import hybrid_calculator
    HYBRID_CALCULATOR_AVAILABLE = True
except ImportError:
    HYBRID_CALCULATOR_AVAILABLE = False
    hybrid_calculator = None

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
        
        # Add travel matrices for enhanced calculations
        self.travel_time_matrix = None
        self.distance_matrix = None
    
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

    def _setup_multi_day_nodes(self, location_list, max_days, verbose):
        """Setup virtual nodes for multi-day scheduling with 'continue from where you left off' behavior."""
        print(f"\n📅 Setting up multi-day nodes for {max_days} days...")
        print(f"  🚛 TRUE multi-day: vehicles continue from their last location (no forced depot returns)")
        
        original_locations = location_list.copy()
        num_original = len(original_locations)
        
        # Create virtual night/morning node pairs for day transitions
        # These are "virtual" nodes that represent sleeping/waking at the same location
        night_nodes = []
        morning_nodes = []
        num_nights = max_days - 1  # If 3 days, need 2 nights
        
        for i in range(num_nights):
            # Night node: represents "end of day i+1" - can be anywhere
            night_node_id = num_original + i
            night_location = {
                'id': f'night_day_{i+1}',
                'x': 0,  # Virtual coordinates - distance will be handled specially
                'y': 0,
                'demand': 0,
                'service_time': 0,  # No service time for overnight stay
                'address': f"End of day {i+1} (sleep wherever vehicle stops)",
                'time_window': (0, 1440),  # Can arrive anytime
                'is_night_node': True,
                'is_virtual': True,  # Mark as virtual node
                'day': i + 1
            }
            location_list.append(night_location)
            night_nodes.append(night_node_id)
            
            # Morning node: represents "start of day i+2" - same location as night
            morning_node_id = num_original + num_nights + i
            morning_location = {
                'id': f'morning_day_{i+2}',
                'x': 0,  # Virtual coordinates - distance will be handled specially
                'y': 0,
                'demand': 0,
                'service_time': 0,  # No service time for morning start
                'address': f"Start of day {i+2} (wake up where vehicle slept)",
                'time_window': (0, 1440),  # Can start anytime
                'is_morning_node': True,
                'is_virtual': True,  # Mark as virtual node
                'day': i + 2
            }
            location_list.append(morning_location)
            morning_nodes.append(morning_node_id)
        
        print(f"  Created {len(night_nodes)} virtual night nodes: {night_nodes}")
        print(f"  Created {len(morning_nodes)} virtual morning nodes: {morning_nodes}")
        print(f"  Total locations: {len(location_list)} (was {num_original})")
        print(f"  💡 Night→Morning pairs represent sleeping/waking at the same location")
        
        return location_list, night_nodes, morning_nodes

    def solve(self, constraint_level: str = "none", verbose: bool = True, use_hybrid_calculator: bool = False, max_days: int = 1, time_limit: int = 120) -> Optional[Dict]:
        """
        Solves the VRP with the specified level of constraints.
        
        Constraint levels:
        - "none": Just distance minimization
        - "capacity": Add capacity constraints  
        - "pickup_delivery": Add pickup-delivery constraints
        - "time_windows": Add time window constraints
        - "full": All constraints
        
        Parameters:
        - verbose: If False, suppresses OR-Tools search logging
        - use_hybrid_calculator: If True, uses hybrid travel calculator for realistic travel times
        - max_days: Maximum number of days to schedule (1 = single day, >1 = multi-day with overnight stays)
        - time_limit: Solver time limit in seconds (default: 120)
        """
        print(f"\n🚀 Solving with constraint level: {constraint_level}")
        if max_days > 1:
            print(f"📅 Multi-day scheduling enabled: up to {max_days} days")

        # Build travel matrices if using hybrid calculator
        if use_hybrid_calculator:
            self._build_travel_matrices(use_hybrid_calculator=True)

        # Add comprehensive sanity check before solving
        self._print_comprehensive_sanity_check(constraint_level)

        location_list = self.locations
        vehicle_list = self.vehicles
        
        # Multi-day setup: Add dummy nodes for overnight stays and morning starts
        original_num_locations = len(location_list)
        if max_days > 1:
            location_list, night_nodes, morning_nodes = self._setup_multi_day_nodes(
                location_list, max_days, verbose
            )
        else:
            night_nodes = []
            morning_nodes = []

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
         
            from_loc = location_list[from_node]
            to_loc = location_list[to_node]
            
            # Handle virtual nodes for multi-day scheduling
            from_is_virtual = from_loc.get('is_virtual', False)
            to_is_virtual = to_loc.get('is_virtual', False)
            from_is_night = from_loc.get('is_night_node', False)
            to_is_night = to_loc.get('is_night_node', False)
            from_is_morning = from_loc.get('is_morning_node', False)
            to_is_morning = to_loc.get('is_morning_node', False)
            
            # Virtual node distance rules for "continue from where you left off":
            # 1. Any location → night node: 0 distance (sleep where you are)
            # 2. Night node → morning node: 0 distance (wake up where you slept)
            # 3. Morning node → any location: 0 distance (already at the location)
            if to_is_night:
                return 0  # No distance to "sleep" - stay where you are
            elif from_is_night and to_is_morning:
                return 0  # No distance from night to morning - same location
            elif from_is_morning:
                return 0  # No distance from morning start - already at location
            elif from_is_virtual or to_is_virtual:
                return 0  # Any other virtual node transition
            
            # Use pre-built distance matrix if available (from hybrid calculator)
            if self.distance_matrix is not None:
                distance = self.distance_matrix[from_node][to_node]
                
                # Add penalty for unnecessary returns to depot (depot is usually index 0)
                # This discourages vehicles from making empty trips back to depot
                if from_node != 0 and to_node == 0:  # Going to depot from non-depot location
                    distance += 10000  # Add penalty to discourage unnecessary returns
                    
                return distance
            
            # Fallback to regular distance calculation for real locations
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
            self._add_time_window_constraints(routing, manager, location_list, vehicle_list, night_nodes, morning_nodes)
            applied_constraints.append("time_windows")
            
        elif constraint_level == "full":

            pickup_delivery_count = self._add_pickup_delivery_constraints(routing, manager, location_list, vehicle_list)
            applied_constraints.append(f"pickup_delivery_{pickup_delivery_count}")

            self._add_capacity_constraints(routing, manager, location_list, vehicle_list)
            applied_constraints.append("capacity")

            
            self._add_time_window_constraints(routing, manager, location_list, vehicle_list, night_nodes, morning_nodes)
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
        
        # Add multi-day constraints if enabled
        if max_days > 1 and night_nodes and morning_nodes:
            self._add_multi_day_constraints(routing, manager, location_list, night_nodes, morning_nodes, max_days)
        
          # 5. Set search parameters with constraint-focused strategy
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        
        # Use PATH_CHEAPEST_ARC for better constraint satisfaction
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        
        # Use AUTOMATIC for better optimization with constraints
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC
        )
        
        search_parameters.time_limit.seconds = time_limit  # Configurable time limit
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
        print(f"  - Local search: AUTOMATIC")
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
        
        # Track solve time
        start_time = time.time()
        solution = routing.SolveWithParameters(search_parameters)
        solve_time = time.time() - start_time
        
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
            print("✅ SOLUTION FOUND!")
            print(f"   Objective: {solution.ObjectiveValue()}")
            print(f"   Time: {solve_time:.2f}s")
            result = self._extract_solution(routing, manager, solution, location_list, vehicle_list, constraint_level, vehicle_idx_to_vehicle)
            
            # Add solve time to result
            result['solve_time'] = solve_time
            
            # Enhanced success message based on constraint level
            constraint_suffix = f" with {constraint_level} constraints" if constraint_level != "none" else ""
            print(f"✅ SUCCESS WITH {constraint_level.upper()} LEVEL{constraint_suffix}!")
            
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
        
    def _add_time_window_constraints(self, routing, manager, location_list, vehicle_list, night_nodes=None, morning_nodes=None):
        """Add time window constraints with multi-day support."""
        print("\n⏰ Adding time window constraints...")
        
        # Create time callback with multi-day support
        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            from_loc = location_list[from_node]
            to_loc = location_list[to_node]
            
            # Handle virtual nodes for multi-day scheduling
            from_is_virtual = from_loc.get('is_virtual', False)
            to_is_virtual = to_loc.get('is_virtual', False)
            from_is_night = from_loc.get('is_night_node', False)
            to_is_night = to_loc.get('is_night_node', False)
            from_is_morning = from_loc.get('is_morning_node', False)
            to_is_morning = to_loc.get('is_morning_node', False)
            
            # Virtual node time rules for "continue from where you left off":
            # 1. Any location → night node: 0 time (instant sleep)
            # 2. Night node → morning node: 12 hours (overnight stay)
            # 3. Morning node → any location: 0 time (already at location)
            if to_is_night:
                # Going to sleep - no travel time, just service time at current location
                service_time = from_loc.get('service_time', 0)
                return int(service_time)
            elif from_is_night and to_is_morning:
    # Overnight stay - 12 hours sleep time
                overnight_time = 12 * 60  # 12 hours in minutes
                return overnight_time
            elif from_is_morning:
                # Starting from morning node - no travel time, already at location
                return 0
            elif from_is_virtual or to_is_virtual:
                # Any other virtual node transition
                            return 0
            
            # Regular travel time calculation for real locations
            if self.travel_time_matrix is not None:
                # Use pre-built travel time matrix from hybrid calculator
                travel_time = self.travel_time_matrix[from_node][to_node]
                
                # Add service time at the "to" location
                service_time = to_loc.get('service_time', 0)
                if hasattr(to_loc, 'service_time') and to_loc.service_time is not None:
                    service_time = int(to_loc.service_time)
                else:
                    # Standard service time: 15 minutes per stop (except for depots)
                    if 'depot' in to_loc.get('id', '').lower():
                        service_time = 5  # 5 minutes at depot for loading/unloading
                    else:
                        service_time = 15  # 15 minutes at pickup/dropoff locations
                
                total_time = int(travel_time + service_time)
                return total_time
            
            # Fallback to Euclidean distance calculation
            from_x = from_loc.get('x', 0)
            from_y = from_loc.get('y', 0)
            to_x = to_loc.get('x', 0)
            to_y = to_loc.get('y', 0)
            
            # Calculate Euclidean distance
            distance = ((to_x - from_x) ** 2 + (to_y - from_y) ** 2) ** 0.5
            
            # Convert coordinate distance to kilometers 
            distance_km = distance * 111
            
            # Calculate travel time based on 70 km/h average speed
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
        
    def _add_multi_day_constraints(self, routing, manager, location_list, night_nodes, morning_nodes, max_days):
        """Add multi-day scheduling constraints with 'continue from where you left off' behavior."""
        print(f"\n📅 Adding multi-day constraints for {max_days} days...")
        print("    🚛 Vehicles continue from where they ended the previous day (realistic multi-day)")
        
        # Allow night and morning nodes to be dropped for free (they're optional)
        for node in night_nodes:
            routing.AddDisjunction([manager.NodeToIndex(node)], 0)
            print(f"    Night node {node} can be dropped for free")
            
        for node in morning_nodes:
            routing.AddDisjunction([manager.NodeToIndex(node)], 0)
            print(f"    Morning node {node} can be dropped for free")
        
        # Create counting dimension to enforce day ordering
        routing.AddConstantDimension(
            1,  # increment by 1 at each node
            len(location_list) + 1,  # max count is visit every node
            True,  # start count at zero
            "Counting"
        )
        count_dimension = routing.GetDimensionOrDie('Counting')
        print("    Created counting dimension for day ordering")
        
        # Add constraints to enforce proper day ordering and continuity
        solver = routing.solver()
        
        # Enforce ordering of night nodes (day 1 night before day 2 night, etc.)
        for i in range(len(night_nodes)):
            inode = night_nodes[i]
            iidx = manager.NodeToIndex(inode)
            iactive = routing.ActiveVar(iidx)
            
            for j in range(i + 1, len(night_nodes)):
                # Make night node i come before night node j using count dimension
                jnode = night_nodes[j]
                jidx = manager.NodeToIndex(jnode)
                jactive = routing.ActiveVar(jidx)
                
                # If both are active, i must come before j
                solver.Add(iactive >= jactive)
                solver.Add(count_dimension.CumulVar(iidx) * iactive * jactive <=
                          count_dimension.CumulVar(jidx) * iactive * jactive)
            
            # If night node is active AND it's not the last night,
            # it must transition to corresponding morning node
            if i < len(morning_nodes):
                i_morning_idx = manager.NodeToIndex(morning_nodes[i])
                # Force night node and corresponding morning node to be both active or inactive
                solver.Add(iactive == routing.ActiveVar(i_morning_idx))
                
                # Force morning node to immediately follow night node in sequence
                solver.Add(count_dimension.CumulVar(iidx) + 1 ==
                          count_dimension.CumulVar(i_morning_idx))
                
                # Add pickup-delivery constraint to ensure same vehicle handles the pair
                routing.AddPickupAndDelivery(iidx, i_morning_idx)
                
                print(f"      Night {inode} ↔ Morning {morning_nodes[i]} paired directly via constraints")
        
        # Enforce ordering of morning nodes
        for i in range(len(morning_nodes)):
            inode = morning_nodes[i]
            iidx = manager.NodeToIndex(inode)
            iactive = routing.ActiveVar(iidx)
            
            for j in range(i + 1, len(morning_nodes)):
                # Make morning node i come before morning node j
                jnode = morning_nodes[j]
                jidx = manager.NodeToIndex(jnode)
                jactive = routing.ActiveVar(jidx)
                
                solver.Add(iactive >= jactive)
                solver.Add(count_dimension.CumulVar(iidx) * iactive * jactive <=
                          count_dimension.CumulVar(jidx) * iactive * jactive)
        
        # Add constraints to properly handle time at overnight stays
        time_dimension = routing.GetDimensionOrDie('Time')
        
        # For each night node (if active), ensure it's within day time bounds
        for i, node in enumerate(night_nodes):
            idx = manager.NodeToIndex(node)
            
            # Link to morning node to enforce continuity
            if i < len(morning_nodes):
                morning_idx = manager.NodeToIndex(morning_nodes[i])
                # Ensure the time difference between night and morning is ~12 hours
                # This is already handled by the time callback which returns overnight time for night->morning transitions
                
        print(f"✅ Multi-day constraints added:")
        print(f"    - {len(night_nodes)} night nodes with free dropping")
        print(f"    - {len(morning_nodes)} morning nodes with free dropping")
        print(f"    - Counting dimension for day ordering")
        print(f"    - Strong night-morning linking constraints")
        print(f"    - Realistic multi-day: vehicles continue from where they left off")
        
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
        print(f"🔍 Starting route analysis for level {constraint_level}...")
        
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
            
            # Add debug info for final route time
            if has_time and time_dimension:
                try:
                    final_time_var = time_dimension.CumulVar(routing.End(vehicle_idx))
                    final_time = solution.Value(final_time_var)
                    route_length = 0
                    temp_index = routing.Start(vehicle_idx)
                    while not routing.IsEnd(temp_index):
                        route_length += 1
                        temp_index = solution.Value(routing.NextVar(temp_index))
                    print(f"       DEBUG: Vehicle {vehicle_idx} - Final time: {final_time}, Route length: {route_length}")
                except:
                    pass
            
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
                    # Calculate actual distance for tracking, even for virtual nodes
                    prev_node = manager.IndexToNode(previous_index)
                    curr_node = manager.IndexToNode(index)
                    prev_loc = location_list[prev_node]
                    curr_loc = location_list[curr_node]
                    
                    # Check if this is a virtual node transition
                    prev_is_virtual = prev_loc.get('is_virtual', False)
                    curr_is_virtual = curr_loc.get('is_virtual', False)
                    
                    # For distance tracking, we need to handle virtual nodes specially
                    # Virtual nodes represent staying at the same location, so:
                    # - Real → Virtual: distance from real location to where vehicle sleeps (0 if sleeping in place)
                    # - Virtual → Virtual: 0 distance (night to morning at same location)  
                    # - Virtual → Real: 0 distance (already at the location after waking up)
                    # - Real → Real: normal distance calculation
                    
                    if prev_is_virtual and curr_is_virtual:
                        # Virtual to virtual (night to morning) - no distance
                        arc_cost_km = 0.0
                    elif prev_is_virtual and not curr_is_virtual:
                        # Virtual to real (morning to real location) - no distance, already there
                        arc_cost_km = 0.0
                    elif not prev_is_virtual and curr_is_virtual:
                        # Real to virtual (real location to night) - no distance, sleep in place
                        arc_cost_km = 0.0
                    else:
                        # Real to real - calculate actual distance
                        prev_x = prev_loc.get('x', 0)
                        prev_y = prev_loc.get('y', 0)
                        curr_x = curr_loc.get('x', 0)
                        curr_y = curr_loc.get('y', 0)
                        
                        # Calculate Euclidean distance
                        distance = ((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2) ** 0.5
                        distance_km = distance * 111  # Convert to km
                        arc_cost_km = distance_km
                    
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
            # Calculate actual distance by tracking real locations only
            real_locations = []
            for stop in route:
                # Find the location details
                for loc in location_list:
                    if loc['id'] == stop['location_id']:
                        if not loc.get('is_virtual', False):
                            real_locations.append(loc)
                        break
            
            # Calculate distance between consecutive real locations
            actual_distance = 0.0
            for i in range(1, len(real_locations)):
                prev_loc = real_locations[i-1]
                curr_loc = real_locations[i]
                
                prev_x = prev_loc.get('x', 0)
                prev_y = prev_loc.get('y', 0)
                curr_x = curr_loc.get('x', 0)
                curr_y = curr_loc.get('y', 0)
                
                # Calculate Euclidean distance
                distance = ((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2) ** 0.5
                distance_km = distance * 111  # Convert to km
                actual_distance += distance_km
            
            routes[vehicle['id']] = {
                'route': route,
                'distance': actual_distance,  # Use actual distance between real locations
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
            
            total_distance += actual_distance  # Use actual distance instead of route_distance
            total_time += route_time
            
            # Enhanced route analysis like in enhanced optimizer
            vehicle_capacity = vehicle.get('capacity', 0)
            vehicle_max_time = vehicle.get('max_time', 540)
            
            print(f"    Vehicle {vehicle['id']}: {len(route)} stops, {route_time}min total ({route_time/vehicle_max_time*100:.1f}% of {vehicle_max_time}min limit)")
            print(f"       📏 Distance: {actual_distance:.2f} km")
            print(f"       🚗 Driving time: {driving_time}min, Service time: {total_service_time}min")
            print(f"       🔍 Math check: {driving_time}min + {total_service_time}min = {driving_time + total_service_time}min (OR-Tools: {route_time}min)")
            
            if has_capacity and len(route) > 1:
                capacity_utilization = (max_manual_load / vehicle_capacity * 100) if vehicle_capacity > 0 else 0
                remaining_capacity = vehicle_capacity - max_manual_load
                
                print(f"       📦 Max load: {max_manual_load}kg ({capacity_utilization:.1f}% of {vehicle_capacity}kg capacity)")
                
                if capacity_utilization < 50:
                    print(f"       📊 Low capacity utilization: {capacity_utilization:.1f}%")
                elif capacity_utilization > 90:
                    print(f"       📊 High capacity utilization: {capacity_utilization:.1f}%")
                else:
                    print(f"       📊 Good capacity utilization: {capacity_utilization:.1f}%")
                    
                print(f"       📦 Remaining capacity: {remaining_capacity}kg")
                
                if max_manual_load > vehicle_capacity:
                    print(f"       ⚠️ WARNING: Max load {max_manual_load}kg exceeds capacity {vehicle_capacity}kg!")
                else:
                    print(f"       ✅ Load within capacity limits")
        
        print("✅ Route analysis completed successfully")
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
        impossible_pairs = 0
        tight_pairs = 0
        if hasattr(self, 'ride_requests') and self.ride_requests:
            print(f"\n🔄 PICKUP-DROPOFF FEASIBILITY:")
            
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
    
    def _build_travel_matrices(self, use_hybrid_calculator: bool = False):
        """Pre-build travel time and distance matrices using hybrid approach when available."""
        if use_hybrid_calculator and HYBRID_CALCULATOR_AVAILABLE:
            print("🌍 Building realistic travel time matrix using hybrid approach...")
            print("   (Haversine + OSRM correction factor - much faster than full OSRM)")
            
            # Extract coordinates from locations
            coordinates = []
            for location in self.locations:
                coordinates.append((location.get('y', 0), location.get('x', 0)))  # (lat, lon)
            
            # Get corrected travel time matrix using hybrid approach
            self.travel_time_matrix = hybrid_calculator.get_corrected_travel_time_matrix(coordinates)
            
            # Build corresponding distance matrix (we still need this for cost calculation)
            n = len(coordinates)
            self.distance_matrix = [[0 for _ in range(n)] for _ in range(n)]
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        self.distance_matrix[i][j] = 0
                    else:
                        distance_km = hybrid_calculator.calculate_haversine_distance(coordinates[i], coordinates[j])
                        self.distance_matrix[i][j] = int(distance_km * 1000)  # meters for OR-Tools
            
            print(f"✅ Hybrid travel matrices built: {n}x{n} locations")
            
            # Add summary like in enhanced optimizer
            avg_travel_time = sum(sum(row) for row in self.travel_time_matrix) / (n * n)
            min_travel_time = min(min(row) for row in self.travel_time_matrix)
            max_travel_time = max(max(row) for row in self.travel_time_matrix)
            
            print(f"📊 Hybrid Travel Time Matrix Summary:")
            print(f"  Locations: {n}")
            print(f"  Total routes: {n * n}")
            print(f"  Average travel time: {avg_travel_time:.1f} minutes")
            print(f"  Min travel time: {min_travel_time:.0f} minutes")
            print(f"  Max travel time: {max_travel_time:.0f} minutes")
        else:
            if use_hybrid_calculator and not HYBRID_CALCULATOR_AVAILABLE:
                print("⚠️ Hybrid calculator requested but not available, falling back to simple calculation")
            
            # Fallback to simple calculation
            self.travel_time_matrix = None
            self.distance_matrix = None

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
    result1, status1, applied_constraints1 = optimizer1.solve(constraint_level="full", verbose=False, use_hybrid_calculator=True)
    print(f"\n=== RUN 1 RESULT ===")
    if result1:
        print(f"✅ Solution found - Objective: {result1['objective_value']}")
        print(f"   Total distance: {result1['total_distance']:.1f} km")
        print(f"   Solve time: {result1.get('solve_time', 'N/A'):.2f}s")
        vehicles_used = len([v for v in result1['routes'].values() if len(v['route']) > 1])
        print(f"   Vehicles used: {vehicles_used}/{len(vehicles_dicts)}")
    else:
        print(f"❌ No solution found - Status: {status1}")

    # --- Test different max_days values on the MODA scenario ---
    max_days_to_test = [1, 2, 3]
    results = {}
    
    for max_days in max_days_to_test:
        print(f"\n--- Testing {max_days} day{'s' if max_days > 1 else ''} ---")
        optimizer = CleanVRPOptimizer(vehicles=vehicles_dicts, locations=None, vrp_instance=scenario)
        optimizer.ride_requests = scenario.ride_requests
        result, status, constraints = optimizer.solve(
            constraint_level="full", 
            verbose=False, 
            use_hybrid_calculator=True,
            max_days=max_days
        )
        
        results[max_days] = {
            'result': result,
            'status': status,
            'constraints': constraints
        }
        
        if result:
            print(f"✅ {max_days}-day solution found")
            print(f"   Objective: {result['objective_value']}")
            print(f"   Total distance: {result['total_distance']:.1f} km")
            print(f"   Solve time: {result.get('solve_time', 'N/A'):.2f}s")
        else:
            print(f"❌ {max_days}-day solution failed: {status}")
    
    # --- Compare all results ---
    print(f"\n📊 COMPARISON: Different max_days values")
    print("="*60)
    successful_results = [(days, data) for days, data in results.items() if data['result'] is not None]
    
    if successful_results:
        best_result = min(successful_results, key=lambda x: x[1]['result']['objective_value'])
        print(f"🏆 Best result: {best_result[0]} days with objective {best_result[1]['result']['objective_value']}")
        
        for days, data in successful_results:
            result = data['result']
            print(f"  {days} day{'s' if days > 1 else ''}: Objective {result['objective_value']}, Distance {result['total_distance']:.1f}km")
    else:
        print("❌ No successful results to compare")


def test_multi_day_scenario():
    """Test multi-day scheduling capability."""
    print("\n" + "="*80)
    print("🧪 TESTING MULTI-DAY VRP SCHEDULING")
    print("="*80)
    
    # Create a challenging test scenario that REQUIRES multi-day scheduling
    locations = [
        {'id': 'depot', 'x': 0, 'y': 0, 'demand': 0, 'service_time': 0, 'time_window': (0, 1440), 'address': 'Main Depot'},
        {'id': 'location_1', 'x': 0.5, 'y': 0.5, 'demand': 10, 'service_time': 60, 'time_window': (60, 1440), 'address': 'Location 1'},
        {'id': 'location_2', 'x': 1.0, 'y': 1.0, 'demand': 15, 'service_time': 60, 'time_window': (120, 1440), 'address': 'Location 2'},
        {'id': 'location_3', 'x': 1.5, 'y': 1.5, 'demand': 20, 'service_time': 60, 'time_window': (180, 1440), 'address': 'Location 3'},
        {'id': 'location_4', 'x': 2.0, 'y': 2.0, 'demand': 25, 'service_time': 60, 'time_window': (240, 1440), 'address': 'Location 4'},
        {'id': 'location_5', 'x': 2.5, 'y': 2.5, 'demand': 30, 'service_time': 60, 'time_window': (300, 1440), 'address': 'Location 5'},
        {'id': 'location_6', 'x': 3.0, 'y': 3.0, 'demand': 35, 'service_time': 60, 'time_window': (360, 1440), 'address': 'Location 6'},
    ]
    
    vehicles = [
        {'id': 'vehicle_1', 'capacity': 300, 'start_location': 'depot', 'end_location': 'depot', 'max_time': 180}
    ]
    
    print(f"📍 Test scenario:")
    print(f"  - {len(locations)} locations (including depot)")
    print(f"  - {len(vehicles)} vehicle with 3-hour daily limit (very restrictive!)")
    print(f"  - Locations spread out with 1-hour service times")
    print(f"  - Single day should struggle, multi-day should visit more locations")
    
    # Test single day (should struggle with far locations)
    print(f"\n--- Single Day Test ---")
    optimizer_1day = CleanVRPOptimizer(vehicles=vehicles, locations=locations)
    result_1day, status_1day, constraints_1day = optimizer_1day.solve(
        constraint_level="time_windows", 
        verbose=False, 
        max_days=1
    )
    
    if result_1day:
        print(f"✅ Single day solution found")
        print(f"   Objective: {result_1day['objective_value']}")
        for vehicle_id, route_data in result_1day['routes'].items():
            print(f"   {vehicle_id}: {len(route_data['route'])} stops")
    else:
        print(f"❌ Single day solution failed: {status_1day}")
    
    # Test multi-day (should handle far locations better)
    print(f"\n--- Multi-Day Test (3 days) ---")
    optimizer_3day = CleanVRPOptimizer(vehicles=vehicles, locations=locations)
    result_3day, status_3day, constraints_3day = optimizer_3day.solve(
        constraint_level="time_windows", 
        verbose=False, 
        max_days=3
    )
    
    if result_3day:
        print(f"✅ Multi-day solution found")
        print(f"   Objective: {result_3day['objective_value']}")
        print(f"   Total distance: {result_3day['total_distance']:.1f} km")
        
        day_transitions = 0
        for vehicle_id, route_data in result_3day['routes'].items():
            route = route_data['route']
            print(f"   {vehicle_id}: {len(route)} stops, distance: {route_data['distance']:.1f} km")
            # Show day transitions
            for i, stop in enumerate(route):
                location_id = stop['location_id']
                if 'night' in location_id or 'morning' in location_id:
                    day_transitions += 1
                    print(f"     Stop {i+1}: {location_id} (day transition at {stop['arrival_time']}min)")
        
        print(f"   🌙 Total day transitions: {day_transitions}")
    else:
        print(f"❌ Multi-day solution failed: {status_3day}")
    
    # Compare solutions if both exist
    if result_1day and result_3day:
        print(f"\n📊 COMPARISON: Single Day vs Multi-Day")
        improvement = (result_1day['objective_value'] - result_3day['objective_value']) / result_1day['objective_value'] * 100
        print(f"   Single Day  - Objective: {result_1day['objective_value']:,}, Distance: {result_1day['total_distance']:.1f} km")
        print(f"   Multi-Day   - Objective: {result_3day['objective_value']:,}, Distance: {result_3day['total_distance']:.1f} km")
        if improvement > 0:
            print(f"   🎉 Multi-day improved objective by {improvement:.1f}%")
        else:
            print(f"   📊 Single-day was better by {-improvement:.1f}%")
    
    # Plot the multi-day solution if available
    if result_3day:
        try:
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
            
            print(f"\n🎨 Creating multi-day route visualization...")
            
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            
            # Create location lookup for coordinates
            location_lookup = {loc['id']: (loc['x'], loc['y']) for loc in locations}
            
            # Plot all locations first
            for loc in locations:
                if loc['id'] == 'depot':
                    ax.scatter(loc['x'], loc['y'], c='red', s=200, marker='s', label='Depot', zorder=5)
                    ax.annotate('DEPOT', (loc['x'], loc['y']), xytext=(5, 5), 
                               textcoords='offset points', fontsize=10, fontweight='bold')
                else:
                    ax.scatter(loc['x'], loc['y'], c='lightblue', s=100, marker='o', zorder=3)
                    ax.annotate(f'{loc["id"]}\n(D:{loc["demand"]})', (loc['x'], loc['y']), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            # Define colors for different days
            day_colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
            
            # Process routes to identify days and plot them
            for vehicle_id, route_data in result_3day['routes'].items():
                route = route_data['route']
                if len(route) <= 1:  # Skip empty or depot-only routes
                    continue
                
                print(f"   📍 Plotting route for {vehicle_id}...")
                
                # Identify day segments by looking at time_windows and cumulative time
                day_segments = []
                current_day = 0
                current_segment = []
                last_departure_time = 0
                
                for i, stop in enumerate(route):
                    location_id = stop['location_id']
                    
                    # Skip depot at start/end for segmentation (but include in plotting)
                    if location_id == 'depot' and (i == 0 or i == len(route) - 1):
                        current_segment.append(stop)
                        continue
                    
                    arrival_time = stop.get('arrival_time', 0)
                    
                    # Check for day transition (large time gap or time reset)
                    if arrival_time < last_departure_time or (arrival_time - last_departure_time > 12 * 60):  # 12 hours gap
                        if current_segment and len(current_segment) > 1:
                            day_segments.append((current_day, current_segment.copy()))
                            current_segment = [current_segment[0]]  # Keep depot for next day
                        current_day += 1
                    
                    current_segment.append(stop)
                    last_departure_time = stop.get('departure_time', arrival_time)
                
                # Add final segment
                if current_segment and len(current_segment) > 1:
                    day_segments.append((current_day, current_segment))
                
                # If no clear day transitions found, split route into reasonable segments
                if not day_segments and route and len(route) > 3:
                    # Simple fallback: split by route length
                    route_length = len(route)
                    if route_length > 10:  # Multiple days likely
                        segment_size = max(4, route_length // 3)  # Aim for 3 days
                        for day in range(3):
                            start_idx = day * segment_size
                            end_idx = min((day + 1) * segment_size + 1, route_length)  # +1 for overlap
                            if start_idx < route_length:
                                segment = route[start_idx:end_idx]
                                if len(segment) > 1:
                                    day_segments.append((day, segment))
                    else:
                        day_segments = [(0, route)]
                elif not day_segments:
                    day_segments = [(0, route)]
                
                # Plot each day segment with different color
                for day_num, segment in day_segments:
                    if len(segment) < 2:
                        continue
                    
                    color = day_colors[day_num % len(day_colors)]
                    
                    # Get coordinates for this segment
                    segment_coords = []
                    segment_locations = []
                    for stop in segment:
                        loc_id = stop['location_id']
                        if loc_id in location_lookup:
                            segment_coords.append(location_lookup[loc_id])
                            segment_locations.append(loc_id)
                    
                    if len(segment_coords) >= 2:
                        # Plot route lines for this day
                        x_coords = [coord[0] for coord in segment_coords]
                        y_coords = [coord[1] for coord in segment_coords]
                        
                        # Only add label if not already added for this day
                        existing_labels = [l.get_label() for l in ax.get_lines()]
                        day_label = f'Day {day_num + 1}'
                        use_label = day_label if day_label not in existing_labels else None
                        
                        ax.plot(x_coords, y_coords, color=color, linewidth=2, alpha=0.7, label=use_label)
                        
                        # Add arrows to show direction
                        for i in range(len(x_coords) - 1):
                            dx = x_coords[i+1] - x_coords[i]
                            dy = y_coords[i+1] - y_coords[i]
                            if abs(dx) > 0.1 or abs(dy) > 0.1:  # Only add arrow if movement is significant
                                ax.annotate('', xy=(x_coords[i+1], y_coords[i+1]), 
                                           xytext=(x_coords[i], y_coords[i]),
                                           arrowprops=dict(arrowstyle='->', color=color, alpha=0.8, lw=1))
                        
                        print(f"     Day {day_num + 1}: {len(segment)} stops ({', '.join(segment_locations[:5])}{', ...' if len(segment_locations) > 5 else ''}), color: {color}")
            
            ax.set_xlabel('X Coordinate (km)')
            ax.set_ylabel('Y Coordinate (km)')
            ax.set_title(f'Multi-Day VRP Solution\n{len(locations)-1} locations, {len(vehicles)} vehicle, 3 days max')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Add statistics text box
            stats_text = f"Objective: {result_3day['objective_value']:,}\n"
            stats_text += f"Total Distance: {result_3day['total_distance']:.1f} km\n"
            stats_text += f"Total Routes: {len([r for r in result_3day['routes'].values() if len(r['route']) > 1])}"
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
                   verticalalignment='top', fontsize=9)
            
            plt.tight_layout()
            
            # Save the plot
            plot_filename = 'multi_day_vrp_solution.png'
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"   💾 Plot saved as: {plot_filename}")
            
            plt.show()
            
        except ImportError:
            print(f"   ⚠️  Matplotlib not available - skipping plot generation")
        except Exception as e:
            print(f"   ❌ Error creating plot: {e}")
    
    print(f"\n🏁 Multi-day test completed!")


if __name__ == "__main__":
    import logging
    import argparse
    import time
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Clean VRP Optimizer with Multi-Day Scheduling')
    
    # Scenario selection
    parser.add_argument('--scenario', type=str, default='multi_day_test',
                        choices=['multi_day_test', 'furgoni', 'moda_small'],
                        help='Which scenario to run (default: multi_day_test)')
    
    # Multi-day scheduling options
    parser.add_argument('--days', type=int, default=1,
                        help='Number of days to schedule (1=single day, >1=multi-day). Default: 1')
    parser.add_argument('--start-hour', type=int, default=6,
                        help='Daily start hour (0-23). Default: 6 (6 AM)')
    parser.add_argument('--end-hour', type=int, default=18,
                        help='Daily end hour (0-23). Default: 18 (6 PM)')
    parser.add_argument('--max-vehicle-time', type=int, default=None,
                        help='Maximum vehicle working time per day in minutes. Default: uses scenario default')
    
    # Constraint level
    parser.add_argument('--constraints', type=str, default='time_windows',
                        choices=['none', 'capacity', 'pickup_delivery', 'time_windows', 'full'],
                        help='Constraint level to apply (default: time_windows)')
    
    # Solver options
    parser.add_argument('--time-limit', type=int, default=120,
                        help='Solver time limit in seconds (default: 120)')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Enable detailed solver logging')
    parser.add_argument('--hybrid-calculator', action='store_true', default=False,
                        help='Use hybrid travel time calculator if available')
    
    # Plotting options
    parser.add_argument('--plot', action='store_true', default=False,
                        help='Generate and save route plots')
    parser.add_argument('--plot-filename', type=str, default='vrp_solution.png',
                        help='Filename for saved plot (default: vrp_solution.png)')
    
    # Analysis options
    parser.add_argument('--compare-days', action='store_true', default=False,
                        help='Compare solutions for 1, 2, and 3 days')
    parser.add_argument('--summary', action='store_true', default=False,
                        help='Show detailed scenario summary before solving')
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    
    print("🚚 CLEAN VRP OPTIMIZER WITH COMMAND-LINE INTERFACE")
    print("=" * 60)
    print(f"📊 Configuration:")
    print(f"   - Scenario: {args.scenario}")
    print(f"   - Days: {args.days}")
    print(f"   - Working hours: {args.start_hour}:00 - {args.end_hour}:00")
    print(f"   - Constraints: {args.constraints}")
    print(f"   - Time limit: {args.time_limit}s")
    print(f"   - Hybrid calculator: {args.hybrid_calculator}")
    print(f"   - Plot results: {args.plot}")
    
    if args.compare_days:
        print(f"   - Compare multiple days: Yes")
    
    def run_scenario_with_args(scenario_name, vehicles, locations, vrp_instance=None, ride_requests=None):
        """Run a scenario with the command-line arguments."""
        
        # Override vehicle max_time if specified
        if args.max_vehicle_time is not None:
            for vehicle in vehicles:
                vehicle['max_time'] = args.max_vehicle_time
            print(f"   🕒 Override vehicle max time: {args.max_vehicle_time} minutes")
        
        # Create optimizer
        optimizer = CleanVRPOptimizer(vehicles=vehicles, locations=locations, vrp_instance=vrp_instance)
        if ride_requests:
            optimizer.ride_requests = ride_requests
        
        # Show summary if requested
        if args.summary:
            optimizer._print_comprehensive_sanity_check(args.constraints)
        
        # Run optimization
        print(f"\n🚀 Running {scenario_name} scenario...")
        start_time = time.time()
        
        result, status, constraints = optimizer.solve(
            constraint_level=args.constraints,
            verbose=args.verbose,
            use_hybrid_calculator=args.hybrid_calculator,
            max_days=args.days,
            time_limit=args.time_limit
        )
        
        solve_time = time.time() - start_time
        
        # Display results
        print(f"\n📊 RESULTS for {scenario_name}:")
        print("=" * 50)
        
        if result:
            print(f"✅ Solution found!")
            print(f"   🎯 Objective: {result['objective_value']:,}")
            print(f"   📏 Total distance: {result['total_distance']:.1f} km")
            print(f"   ⏱️  Solve time: {solve_time:.2f}s")
            print(f"   🚛 Vehicle utilization:")
            
            active_routes = 0
            for vehicle_id, route_data in result['routes'].items():
                if len(route_data['route']) > 1:  # More than just depot
                    active_routes += 1
                    route_stops = len(route_data['route'])
                    route_distance = route_data['distance']
                    route_time = route_data['time']
                    print(f"      {vehicle_id}: {route_stops} stops, {route_distance:.1f} km, {route_time:.0f} min")
            
            print(f"   📈 Success rate: {active_routes}/{len(vehicles)} vehicles used")
            
            # Validation summary
            if 'validation_results' in result:
                valid_constraints = sum(result['validation_results'].values())
                total_constraints = len(result['validation_results'])
                print(f"   ✅ Constraint validation: {valid_constraints}/{total_constraints} passed")
            
            # Plot if requested
            if args.plot:
                try:
                    import matplotlib.pyplot as plt
                    print(f"\n🎨 Generating plot...")
                    optimizer.plot_solution(result, f"{scenario_name} Solution")
                    plt.savefig(args.plot_filename, dpi=300, bbox_inches='tight')
                    print(f"   💾 Plot saved as: {args.plot_filename}")
                except ImportError:
                    print(f"   ⚠️ Matplotlib not available for plotting")
                except Exception as e:
                    print(f"   ❌ Plotting error: {e}")
        else:
            print(f"❌ No solution found")
            print(f"   Status: {status}")
            print(f"   Solve time: {solve_time:.2f}s")
            print(f"   💡 Try: --days {args.days + 1} or --time-limit {args.time_limit * 2}")
        
        return result, solve_time
    
    # Run the selected scenario
    if args.scenario == 'multi_day_test':
        # Create test scenario with configurable parameters
        locations = [
            {'id': 'depot', 'x': 0, 'y': 0, 'demand': 0, 'service_time': 0, 'time_window': (0, 1440), 'address': 'Main Depot'},
            {'id': 'location_1', 'x': 0.5, 'y': 0.5, 'demand': 10, 'service_time': 60, 'time_window': (60, 1440), 'address': 'Location 1'},
            {'id': 'location_2', 'x': 1.0, 'y': 1.0, 'demand': 15, 'service_time': 60, 'time_window': (120, 1440), 'address': 'Location 2'},
            {'id': 'location_3', 'x': 1.5, 'y': 1.5, 'demand': 20, 'service_time': 60, 'time_window': (180, 1440), 'address': 'Location 3'},
            {'id': 'location_4', 'x': 2.0, 'y': 2.0, 'demand': 25, 'service_time': 60, 'time_window': (240, 1440), 'address': 'Location 4'},
            {'id': 'location_5', 'x': 2.5, 'y': 2.5, 'demand': 30, 'service_time': 60, 'time_window': (300, 1440), 'address': 'Location 5'},
            {'id': 'location_6', 'x': 3.0, 'y': 3.0, 'demand': 35, 'service_time': 60, 'time_window': (360, 1440), 'address': 'Location 6'},
        ]
        
        # Calculate working hours in minutes
        working_hours = (args.end_hour - args.start_hour) * 60
        vehicle_max_time = args.max_vehicle_time if args.max_vehicle_time else min(180, working_hours)  # Default 3 hours or working day
        
        vehicles = [
            {'id': 'vehicle_1', 'capacity': 300, 'start_location': 'depot', 'end_location': 'depot', 'max_time': vehicle_max_time}
        ]
        
        if args.compare_days:
            print(f"\n🔄 COMPARING DIFFERENT DAY CONFIGURATIONS")
            print("=" * 60)
            
            results = {}
            for test_days in [1, 2, 3]:
                print(f"\n--- Testing {test_days} day{'s' if test_days > 1 else ''} ---")
                
                # Temporarily override args.days
                original_days = args.days
                args.days = test_days
                
                result, solve_time = run_scenario_with_args(
                    f"Multi-Day Test ({test_days} days)", 
                    vehicles, 
                    locations
                )
                
                results[test_days] = {'result': result, 'solve_time': solve_time}
                
                # Restore original days
                args.days = original_days
            
            # Compare results
            print(f"\n📊 COMPARISON SUMMARY:")
            print("=" * 50)
            successful_results = [(days, data) for days, data in results.items() if data['result'] is not None]
            
            if successful_results:
                best_result = min(successful_results, key=lambda x: x[1]['result']['objective_value'])
                print(f"🏆 Best solution: {best_result[0]} days (objective: {best_result[1]['result']['objective_value']:,})")
                
                for days, data in successful_results:
                    result = data['result']
                    solve_time = data['solve_time']
                    print(f"   {days} day{'s' if days > 1 else ''}: Obj {result['objective_value']:,}, "
                          f"Dist {result['total_distance']:.1f}km, Time {solve_time:.2f}s")
            else:
                print("❌ No successful solutions found")
        else:
            run_scenario_with_args("Multi-Day Test", vehicles, locations)
    
    elif args.scenario == 'furgoni':
        try:
            from vrp_scenarios import create_furgoni_scenario
            scenario = create_furgoni_scenario()
            
            # Convert vehicles and get ride requests
            vehicle_ids = list(scenario.vehicles.keys())
            vehicles_from_scenario = [scenario.vehicles[vid] for vid in vehicle_ids]
            vehicles_dicts = [{
                'id': v.id,
                'capacity': v.capacity,
                'start_location': v.depot_id,
                'end_location': v.depot_id,
                'max_time': getattr(v, 'max_time', 24 * 60)
            } for v in vehicles_from_scenario]
            
            run_scenario_with_args("Furgoni", vehicles_dicts, None, scenario, scenario.ride_requests)
            
        except ImportError:
            print("❌ Could not import furgoni scenario. Make sure vrp_scenarios.py is available.")
    
    elif args.scenario == 'moda_small':
        try:
            from vrp_scenarios import create_moda_small_scenario
            scenario = create_moda_small_scenario()
            
            # Convert vehicles and get ride requests
            vehicle_ids = list(scenario.vehicles.keys())
            vehicles_from_scenario = [scenario.vehicles[vid] for vid in vehicle_ids]
            vehicles_dicts = [{
                'id': v.id,
                'capacity': v.capacity,
                'start_location': v.depot_id,
                'end_location': v.depot_id,
                'max_time': getattr(v, 'max_time', 24 * 60)
            } for v in vehicles_from_scenario]
            
            run_scenario_with_args("MODA Small", vehicles_dicts, None, scenario, scenario.ride_requests)
            
        except ImportError:
            print("❌ Could not import MODA small scenario. Make sure vrp_scenarios.py is available.")
    
    print(f"\n🎉 Clean VRP Optimizer execution completed!")
    print(f"💡 Try different options:")
    print(f"   python vrp_optimizer_clean.py --days 3 --constraints full --plot")
    print(f"   python vrp_optimizer_clean.py --scenario furgoni --days 4 --compare-days")
    print(f"   python vrp_optimizer_clean.py --scenario moda_small --constraints full --hybrid-calculator")