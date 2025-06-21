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
    
    def _convert_vehicles_from_instance(self, instance):
        """Convert VRPInstance vehicles to dict format."""
        vehicles = []
        for vehicle in instance.vehicles.values():
            vehicles.append({
                'id': vehicle.id,
                'capacity': vehicle.capacity,
                'start_location': vehicle.depot_id,
                'end_location': vehicle.depot_id,
                'max_time': getattr(vehicle, 'max_time', 1440)
            })
        return vehicles
    
    def _convert_locations_from_instance(self, instance):
        """Convert VRPInstance locations to dict format."""
        locations = []
        for location in instance.locations.values():
            loc_dict = {
                'id': location.id,
                'x': location.x,  # longitude
                'y': location.y,  # latitude
                'demand': location.demand,
                'address': getattr(location, 'address', location.id)
            }
            
            # Add time window if present
            if hasattr(location, 'time_window_start') and hasattr(location, 'time_window_end'):
                loc_dict['time_window'] = (location.time_window_start, location.time_window_end)
            
            # Add service time if present
            if hasattr(location, 'service_time'):
                loc_dict['service_time'] = location.service_time
            
            locations.append(loc_dict)
        return locations

    def solve(self, constraint_level: str = "none") -> Optional[Dict]:
        """
        Solves the VRP with the specified level of constraints.
        
        Constraint levels:
        - "none": Just distance minimization
        - "capacity": Add capacity constraints  
        - "pickup_delivery": Add pickup-delivery constraints
        - "time_windows": Add time window constraints
        - "full": All constraints
        """
        print(f"\n🚀 Solving with constraint level: {constraint_level}")

        location_list = self.locations
        vehicle_list = self.vehicles

        print("📊 Problem size:")
        print(f"  - Locations: {len(location_list)}")
        print(f"  - Vehicles: {len(vehicle_list)}")

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
            
            # For now, just use a simple distance calculation
            # This would be replaced with actual distance calculation from a provider
            from_x = from_loc.get('x', 0)
            from_y = from_loc.get('y', 0)
            to_x = to_loc.get('x', 0)
            to_y = to_loc.get('y', 0)
            
            # Use Manhattan distance for now
            distance = abs(from_x - to_x) + abs(from_y - to_y)
            return int(distance * 1000)  # Convert to integer
        
        distance_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(distance_callback_index)
        
        applied_constraints = ["distance"]
        
        # 4. Add constraints based on level
        if constraint_level == "capacity":
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
            # Add all constraints
            self._add_capacity_constraints(routing, manager, location_list, vehicle_list)
            applied_constraints.append("capacity")
            
            pickup_delivery_count = self._add_pickup_delivery_constraints(routing, manager, location_list, vehicle_list)
            applied_constraints.append(f"pickup_delivery_{pickup_delivery_count}")
            
            self._add_time_window_constraints(routing, manager, location_list, vehicle_list)
            applied_constraints.append("time_windows")
        
        print(f"✅ Constraints applied: {applied_constraints}")
        
        # 5. Set search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.seconds = 30
        
        # 6. Solve
        print("🔍 Solving...")
        solution = routing.SolveWithParameters(search_parameters)
        
        if solution:
            print("✅ Solution found!")
            result = self._extract_solution(routing, manager, solution, location_list, vehicle_list, constraint_level)
            return result, "Success", applied_constraints
        else:
            print("❌ No solution found!")
            return None, "Failed", applied_constraints
            
    def _add_capacity_constraints(self, routing, manager, location_list, vehicle_list):
        """Add capacity constraints step by step."""
        print("\n📦 Adding capacity constraints...")
        
        # Create demand callback
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            location = location_list[from_node]
            
            # Get demand directly from the location dict
            demand = location.get('demand', 0)
            return demand
        
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
          # Get vehicle capacities
        vehicle_capacities = [int(vehicle.get('capacity', 0)) for vehicle in vehicle_list]
        print(f"  Vehicle capacities: {vehicle_capacities}")
          # Add dimension
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,  # Demand callback
            0,  # Null capacity slack - this means we count all demand
            vehicle_capacities,  # Vehicle capacity array
            True,  # Capacity cumulates from start of route
            'Capacity'  # Dimension name
        )
        print("✅ Capacity constraints added")
        
    def _add_pickup_delivery_constraints(self, routing, manager, location_list, vehicle_list):
        """Add pickup and delivery constraints using ride requests."""
        print("\n🔄 Adding pickup-delivery constraints...")
        
        pickup_delivery_count = 0
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
        
        for request_id, request in requests_to_process:
            pickup_location = request.pickup_location
            dropoff_location = request.dropoff_location
            
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
            routing.AddPickupAndDelivery(pickup_index, dropoff_index)
            
            # Ensure the same vehicle handles both pickup and delivery
            routing.solver().Add(routing.VehicleVar(pickup_index) == routing.VehicleVar(dropoff_index))
            
            pickup_delivery_count += 1
        
        print(f"✅ Added {pickup_delivery_count} pickup-delivery pairs")
        return pickup_delivery_count
        
    def _add_time_window_constraints(self, routing, manager, location_list, vehicle_list):
        """Add time window constraints."""
        print("\n⏰ Adding time window constraints...")
        
        # Create time callback
        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            from_loc = location_list[from_node]
            to_loc = location_list[to_node]
            
            # Travel time (Manhattan distance as proxy)
            from_x = from_loc.get('x', 0)
            from_y = from_loc.get('y', 0)
            to_x = to_loc.get('x', 0)
            to_y = to_loc.get('y', 0)
            
            travel_time = abs(from_x - to_x) + abs(from_y - to_y)
            
            # Add service time at the "from" location
            service_time = from_loc.get('service_time', 0)
            
            return int(travel_time + service_time)
        
        time_callback_index = routing.RegisterTransitCallback(time_callback)
        
        # Add time dimension
        max_time = 1440  # Default to 24 hours
        for vehicle in vehicle_list:
            vehicle_max_time = vehicle.get('max_time', 1440)
            if vehicle_max_time > max_time:
                max_time = vehicle_max_time
        
        routing.AddDimension(
            time_callback_index,  # Time callback
            30,                   # Allow 30 min slack
            max_time,             # Maximum time per vehicle
            False,                # Don't force start cumul to zero
            'Time'                # Dimension name
        )
        
        time_dimension = routing.GetDimensionOrDie('Time')
        
        # Add time windows for locations
        for i, location in enumerate(location_list):
            if 'time_window' in location:
                time_window_start, time_window_end = location['time_window']
                time_dimension.CumulVar(i).SetRange(time_window_start, time_window_end)
                print(f"    Time window for {location['id']}: [{time_window_start}, {time_window_end}]")
        
        print("✅ Time window constraints added")
        
    def _extract_solution(self, routing, manager, solution, location_list, vehicle_list, constraint_level: str = "none") -> Dict:        
        """Extract and format the solution."""
        print("\n📋 Extracting solution...")
        
        routes = {}
        total_distance = 0
        total_time = 0
        # Check which dimensions are available
        has_capacity = constraint_level in ["capacity", "full"]
        has_time = constraint_level in ["time_windows", "full"]
        
        for vehicle_idx in range(len(vehicle_list)):
            vehicle = vehicle_list[vehicle_idx]
            route = []
            
            index = routing.Start(vehicle_idx)
            route_distance = 0
            route_time = 0
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                location = location_list[node_index]
                
                # Safely get dimension values
                arrival_time = 0
                load = 0
                
                # Check if Time dimension exists and get arrival time
                if has_time:
                    try:
                        time_dimension = routing.GetDimensionOrDie('Time')
                        arrival_time = solution.Value(time_dimension.CumulVar(index))
                    except:
                        arrival_time = 0
                else:
                    arrival_time = 0
                
                # Check if Capacity dimension exists and get load
                if has_capacity:
                    try:
                        capacity_dimension = routing.GetDimensionOrDie('Capacity')
                        load = solution.Value(capacity_dimension.CumulVar(index))
                    except:
                        load = 0
                else:
                    load = 0
                
                route.append({
                    'location_id': location['id'],
                    'location_name': location.get('address', location['id']),
                    'coordinates': (location.get('x', 0), location.get('y', 0)),
                    'arrival_time': arrival_time,
                    'load': load
                })
                
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                
                if not routing.IsEnd(index):
                    route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_idx)
            
            # Add final location
            if not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                location = location_list[node_index]
                
                # Safely get dimension values for final location
                arrival_time = 0
                load = 0
                
                # Check if Time dimension exists and get arrival time
                if has_time:
                    try:
                        time_dimension = routing.GetDimensionOrDie('Time')
                        arrival_time = solution.Value(time_dimension.CumulVar(index))
                    except:
                        arrival_time = 0
                else:
                    arrival_time = 0
                
                # Check if Capacity dimension exists and get load
                if has_capacity:
                    try:
                        capacity_dimension = routing.GetDimensionOrDie('Capacity')
                        load = solution.Value(capacity_dimension.CumulVar(index))
                    except:
                        load = 0
                else:
                    load = 0
                
                route.append({
                    'location_id': location['id'],
                    'location_name': location.get('address', location['id']),
                    'coordinates': (location.get('x', 0), location.get('y', 0)),
                    'arrival_time': arrival_time,
                    'load': load
                })
            
            routes[vehicle['id']] = {
                'route': route,
                'distance': route_distance,
                'time': route_time
            }
            
            total_distance += route_distance
            total_time += route_time
            
            print(f"  Vehicle {vehicle['id']}: {len(route)} stops, distance: {route_distance}")
        
        return {
            'status': 'success',
            'routes': routes,
            'total_distance': total_distance,
            'total_time': total_time,
            'objective_value': solution.ObjectiveValue()
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
                                             connectionstyle="arc3,rad=0.1"),
                               zorder=1)
        
        # Plot pickup-delivery connections (thin dashed lines)
        if isinstance(self.ride_requests, list):
            for request in self.ride_requests:
                pickup_loc = None
                dropoff_loc = None
                
                for loc in self.locations:
                    if loc['id'] == request.pickup_location:
                        pickup_loc = loc
                    elif loc['id'] == request.dropoff_location:
                        dropoff_loc = loc
                
                if pickup_loc and dropoff_loc:
                    plt.plot([pickup_loc['x'], dropoff_loc['x']], 
                            [pickup_loc['y'], dropoff_loc['y']], 
                            '--', color='gray', alpha=0.5, linewidth=1)
        
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
        
        plt.title(f"{title}\nTotal Distance: {result['total_distance']}, Objective: {result['objective_value']}")
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

def test_moda_small_scenario():
    """Test the clean optimizer with MODA_small VRPPD scenario."""
    print("🧪 Testing Clean VRP Optimizer with MODA_small scenario")
    print("=" * 60)
    
    # Import the scenario generator
    try:
        from vrp_scenarios import create_moda_small_scenario
    except ImportError:
        print("❌ Could not import vrp_scenarios. Make sure the file is available.")
        return
    
    # Create the MODA_small scenario
    print("Creating MODA_small scenario...")
    scenario = create_moda_small_scenario()
    
    print(f"📊 Scenario details:")
    print(f"  - Locations: {len(scenario.locations)}")
    print(f"  - Vehicles: {len(scenario.vehicles)}")
    print(f"  - Ride requests: {len(scenario.ride_requests)}")
    
    # Create optimizer with the scenario
    optimizer = CleanVRPOptimizer(vrp_instance=scenario)
    
    # Test pickup-delivery constraints only (ignoring capacity, time windows, service time)
    print(f"\n{'='*20} TESTING PICKUP-DELIVERY ONLY {'='*20}")
    
    try:
        result, status, applied_constraints = optimizer.solve(constraint_level="pickup_delivery")
        
        if result:
            print(f"✅ SUCCESS - MODA_small VRPPD solved!")
            print(f"   Status: {status}")
            print(f"   Constraints applied: {applied_constraints}")
            print(f"   Objective value: {result['objective_value']}")
            print(f"   Total distance: {result['total_distance']}")
              # Show route summary
            for vehicle_id, route_data in result['routes'].items():
                route = route_data['route']
                if len(route) > 2:  # More than just start-end
                    print(f"   Vehicle {vehicle_id}: {len(route)} stops, distance: {route_data['distance']}")
                    for i, stop in enumerate(route[:5]):  # Show first 5 stops
                        print(f"     {i+1}. {stop['location_id']} (load: {stop['load']})")
                    if len(route) > 5:
                        print(f"     ... and {len(route)-5} more stops")
                else:
                    print(f"   Vehicle {vehicle_id}: No stops assigned")
              # Plot solution
            optimizer.plot_solution(result, title="MODA_small VRPPD Solution")
        else:
            print(f"❌ FAILED - Status: {status}")
            print(f"   Constraints applied: {applied_constraints}")
            
    except Exception as e:
        print(f"💥 ERROR: {str(e)}")
        import traceback
        traceback.print_exc()


def test_constraint_levels():
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
    test_moda_small_scenario()
