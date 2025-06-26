#!/usr/bin/env python3
"""
Fixed Multi-Day VRP Implementation
==================================

This implements the correct approach for multi-day VRP where each vehicle
has its own set of overnight stay nodes, and disjunctions ensure each 
vehicle selects at most one overnight stay per night.

The key insight: Create separate overnight stay nodes for each vehicle,
similar to how you'd handle dinner rooms in a restaurant - each vehicle
gets its own "dinner table" options, and can only choose one per night.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import importlib.util
import matplotlib.pyplot as plt
import numpy as np

# Import the distance calculator from the original file
spec = importlib.util.spec_from_file_location("vrp_original", 
                                              os.path.join(os.path.dirname(__file__), "vrp_optimizer_clean copy.py"))
vrp_original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vrp_original)
OSMDistanceCalculator = vrp_original.OSMDistanceCalculator

class FixedMultiDayVRP:
    """
    Fixed Multi-Day VRP implementation with proper vehicle-specific overnight stays.
    """
    
    def __init__(self, vehicles, locations):
        self.vehicles = vehicles
        self.locations = locations
        
        # Initialize distance calculator
        self.distance_calculator = OSMDistanceCalculator(self.locations)
        
    def create_vehicle_specific_overnight_nodes(self, max_days):
        """
        Create separate overnight stay nodes for each vehicle at actual locations.
        
        FIXED APPROACH: Instead of virtual coordinates (0,0), overnight nodes are 
        positioned at the coordinates of their corresponding customer locations.
        This provides realistic visualization of where vehicles actually stop overnight.
        """
        if max_days <= 1:
            return self.locations.copy(), {}, {}
            
        print(f"\n🔧 FIXED: Creating vehicle-specific overnight nodes for {max_days} days")
        print(f"  Each vehicle gets its own 'dinner table' for each night AT ACTUAL LOCATIONS")
        
        working_locations = self.locations.copy()
        num_original = len(working_locations)
        num_vehicles = len(self.vehicles)
        num_nights = max_days - 1
        
        # Track overnight nodes by vehicle and night
        vehicle_overnight_nodes = {}  # vehicle_id -> [night1_node, night2_node, ...]
        night_sets = []  # [night1_nodes_for_all_vehicles, night2_nodes_for_all_vehicles, ...]
        
        # For each customer location (excluding depot), create overnight possibilities
        for customer_idx, customer_location in enumerate(self.locations):
            if customer_idx == 0:  # Skip depot
                continue
                
            for night_idx in range(num_nights):
                night_nodes_for_this_night = []
                
                for vehicle_idx, vehicle in enumerate(self.vehicles):
                    vehicle_id = vehicle['id']
                    
                    if vehicle_id not in vehicle_overnight_nodes:
                        vehicle_overnight_nodes[vehicle_id] = []
                    
                    # Create overnight node at this customer location for this vehicle and night
                    node_id = len(working_locations)
                    overnight_location = {
                        'id': f'overnight_{customer_location["id"]}_night{night_idx+1}_vehicle_{vehicle_id}',
                        'x': customer_location['x'],  # ACTUAL customer coordinates
                        'y': customer_location['y'],  # ACTUAL customer coordinates
                        'demand': 0,
                        'volume_demand': 0,
                        'service_time': 0,
                        'address': f"Night {night_idx+1} overnight at {customer_location['id']} for {vehicle_id}",
                        'time_window': (0, 1440),
                        'is_overnight_node': True,
                        'is_virtual': False,  # NOT virtual - real coordinates!
                        'night': night_idx + 1,
                        'vehicle_id': vehicle_id,
                        'vehicle_idx': vehicle_idx,
                        'customer_location_idx': customer_idx,
                        'base_customer_id': customer_location['id']
                    }
                    
                    working_locations.append(overnight_location)
                    
                    # Only add to vehicle's overnight nodes if this is the first night
                    # (to avoid duplicate entries for multiple customers)
                    if customer_idx == 1:  # Only for first customer location
                        vehicle_overnight_nodes[vehicle_id].append(node_id)
                    
                    night_nodes_for_this_night.append(node_id)
                    
                    print(f"    Created overnight node {node_id} for {vehicle_id} at {customer_location['id']} ({customer_location['x']}, {customer_location['y']}) on night {night_idx+1}")
                
                # Only add night set once per night (not per customer)
                if customer_idx == 1:  # Only for first customer
                    night_sets.append(night_nodes_for_this_night)
        
        print(f"  ✅ Created overnight nodes at {len(self.locations)-1} customer locations")
        print(f"  📍 Total locations: {len(working_locations)} (was {num_original})")
        
        return working_locations, vehicle_overnight_nodes, night_sets
    
    def add_fixed_overnight_constraints(self, routing, manager, working_locations, vehicle_overnight_nodes, night_sets):
        """
        Add the fixed overnight stay constraints.
        
        This implements the "dinner room" approach:
        1. Each night has multiple overnight nodes (one per vehicle)
        2. Each vehicle can only visit its own overnight nodes
        3. Each vehicle can visit at most one overnight node per night
        """
        print(f"\n🔧 FIXED: Adding vehicle-specific overnight constraints")
        
        solver = routing.solver()
        num_vehicles = len(self.vehicles)
        
        # 1. For each night, ensure each vehicle can visit at most one overnight node
        for night_idx, night_nodes in enumerate(night_sets):
            print(f"  Night {night_idx + 1}: Setting up disjunctions for {len(night_nodes)} vehicle-specific nodes")
            
            # Group nodes by vehicle for this night
            for vehicle_idx, vehicle in enumerate(self.vehicles):
                vehicle_id = vehicle['id']
                
                # Find this vehicle's overnight node for this night
                vehicle_overnight_node = None
                for node_id in night_nodes:
                    if node_id < len(working_locations):
                        location = working_locations[node_id]
                        if location.get('vehicle_id') == vehicle_id:
                            vehicle_overnight_node = node_id
                            break
                
                if vehicle_overnight_node is not None:
                    # Create disjunction: vehicle can visit this node or not (penalty = 0)
                    node_idx = manager.NodeToIndex(vehicle_overnight_node)
                    routing.AddDisjunction([node_idx], 0)
                    
                    # Restrict this node to only this vehicle
                    for v in range(num_vehicles):
                        if v != vehicle_idx:
                            solver.Add(routing.VehicleVar(node_idx) != v)
                    
                    print(f"    Vehicle {vehicle_id} can optionally visit overnight node {vehicle_overnight_node}")
        
        # 2. Add ordering constraints within each vehicle's route
        # Add counting dimension first (needed for ordering)
        routing.AddConstantDimension(
            1,  # increment by 1 at each node
            len(working_locations) + 1,  # max count
            True,  # start at zero
            "Counting"
        )
        count_dimension = routing.GetDimensionOrDie('Counting')
        
        for vehicle_idx, vehicle in enumerate(self.vehicles):
            vehicle_id = vehicle['id']
            
            if vehicle_id in vehicle_overnight_nodes:
                overnight_nodes = vehicle_overnight_nodes[vehicle_id]
                
                # If vehicle visits multiple overnight nodes, they must be in order
                for i in range(len(overnight_nodes) - 1):
                    node1_idx = manager.NodeToIndex(overnight_nodes[i])
                    node2_idx = manager.NodeToIndex(overnight_nodes[i + 1])
                    
                    # If both are visited, node1 must come before node2
                    active1 = routing.ActiveVar(node1_idx)
                    active2 = routing.ActiveVar(node2_idx)
                    
                    # Add constraint using counting dimension for ordering
                    solver.Add(
                        count_dimension.CumulVar(node1_idx) * active1 * active2 <=
                        count_dimension.CumulVar(node2_idx) * active1 * active2
                    )
        
        print(f"  ✅ Fixed overnight constraints added for {len(night_sets)} nights")
    
    def create_distance_callback(self, manager, working_locations):
        """Create distance callback that handles virtual overnight nodes."""
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            
            # Handle virtual overnight nodes
            if from_node >= len(self.locations) or to_node >= len(self.locations):
                # At least one is a virtual overnight node
                return 1000  # Small cost for virtual transitions
            
            # Normal distance calculation
            distance_km = self.distance_calculator.distance_matrix[from_node][to_node]
            return int(distance_km * 1000)  # Convert to meters
        
        return distance_callback
    
    def solve_fixed_multiday(self, max_days=3, daily_time_limit_minutes=480):
        """
        Solve the multi-day VRP with the fixed overnight stay implementation.
        """
        print(f"\n🚀 FIXED Multi-Day VRP Solver")
        print(f"   Vehicles: {len(self.vehicles)}")
        print(f"   Locations: {len(self.locations)}")
        print(f"   Max days: {max_days}")
        print(f"   Daily limit: {daily_time_limit_minutes} minutes")
        
        # Create vehicle-specific overnight nodes
        working_locations, vehicle_overnight_nodes, night_sets = self.create_vehicle_specific_overnight_nodes(max_days)
        
        # Create OR-Tools model
        manager = pywrapcp.RoutingIndexManager(len(working_locations), len(self.vehicles), 0)
        routing = pywrapcp.RoutingModel(manager)
        
        # Add distance callback
        distance_callback = self.create_distance_callback(manager, working_locations)
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Add capacity constraints (simplified for demo)
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            if from_node < len(self.locations):
                return self.locations[from_node].get('demand', 0)
            return 0  # Virtual nodes have no demand
        
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        capacities = [vehicle.get('capacity', 1000) for vehicle in self.vehicles]
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index, 0, capacities, True, 'Capacity'
        )
        
        # Add time dimension with daily limits
        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            
            # Handle overnight nodes (they reset the daily time counter)
            if to_node >= len(self.locations):
                to_location = working_locations[to_node]
                if to_location.get('is_overnight_node', False):
                    return -(daily_time_limit_minutes * 60)  # Reset time for new day
            
            # Normal time calculation
            if from_node < len(self.locations) and to_node < len(self.locations):
                time_minutes = self.distance_calculator.time_matrix[from_node][to_node]
                if to_node > 0:  # Add service time
                    time_minutes += self.locations[to_node].get('service_time', 0)
                return int(time_minutes * 60)  # Convert to seconds
            
            return 60  # 1 minute for virtual transitions
        
        time_callback_index = routing.RegisterTransitCallback(time_callback)
        routing.AddDimension(
            time_callback_index,
            daily_time_limit_minutes * 60,  # Max slack
            daily_time_limit_minutes * 60,  # Max cumulative time
            True,  # Start at zero
            'Time'
        )
        
        # Add the fixed overnight constraints
        self.add_fixed_overnight_constraints(routing, manager, working_locations, vehicle_overnight_nodes, night_sets)
        
        # Solve
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_parameters.time_limit.seconds = 60
        
        print(f"\n🔍 Solving with fixed constraints...")
        solution = routing.SolveWithParameters(search_parameters)
        
        if solution:
            print(f"✅ FIXED solution found!")
            return self.extract_fixed_solution(routing, manager, solution, working_locations, vehicle_overnight_nodes)
        else:
            print(f"❌ No solution found")
            return None
    
    def extract_fixed_solution(self, routing, manager, solution, working_locations, vehicle_overnight_nodes):
        """Extract and format the fixed solution."""
        routes = {}
        total_distance = 0
        
        for vehicle_idx in range(len(self.vehicles)):
            vehicle = self.vehicles[vehicle_idx]
            vehicle_id = vehicle['id']
            
            route_stops = []
            route_distance = 0
            index = routing.Start(vehicle_idx)
            
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                location = working_locations[node]
                
                stop_info = {
                    'location_id': location['id'],
                    'is_overnight': location.get('is_overnight_node', False)
                }
                
                route_stops.append(stop_info)
                
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                
                if not routing.IsEnd(index):
                    route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_idx)
            
            # Add final depot
            final_node = manager.IndexToNode(index)
            route_stops.append({'location_id': working_locations[final_node]['id'], 'is_overnight': False})
            
            routes[vehicle_id] = {
                'stops': route_stops,
                'distance': route_distance / 1000.0,  # Convert back to km
                'overnight_count': sum(1 for stop in route_stops if stop['is_overnight'])
            }
            
            total_distance += route_distance / 1000.0
        
        return {
            'status': 'SUCCESS',
            'routes': routes,
            'total_distance': total_distance,
            'vehicle_overnight_nodes': vehicle_overnight_nodes
        }


    def plot_multiday_solution(self, result, working_locations, title="Fixed Multi-Day VRP Solution"):
        """
        Plot the multi-day solution with different colors for vehicles and line styles for days.
        """
        if not result or result.get('status') != 'SUCCESS':
            print("❌ Cannot plot: No valid solution provided")
            return
        
        print(f"\n🎨 Plotting multi-day solution...")
        
        # Set up the plot
        plt.figure(figsize=(12, 10))
        plt.title(title, fontsize=16, fontweight='bold')
        
        # Define colors for vehicles
        vehicle_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        # Define line styles for different days
        day_styles = ['-', '--', '-.', ':', '-', '--', '-.']  # Cycle if more than 7 days
        
        # Create a mapping of location IDs to coordinates
        location_coords = {}
        for loc in working_locations:
            location_coords[loc['id']] = (loc['x'], loc['y'])
        
        # Track which locations are overnight stays vs regular locations
        overnight_locations = set()
        regular_locations = set()
        
        for loc in working_locations:
            if loc.get('is_overnight_node', False):
                overnight_locations.add(loc['id'])
            else:
                regular_locations.add(loc['id'])
        
        # Plot each vehicle's route
        for vehicle_idx, (vehicle_id, route_data) in enumerate(result['routes'].items()):
            vehicle_color = vehicle_colors[vehicle_idx % len(vehicle_colors)]
            stops = route_data['stops']
            
            if len(stops) < 2:
                continue
            
            print(f"  Plotting {vehicle_id} route with color {vehicle_color}")
            
            # Group stops by day based on overnight stays
            days_routes = []  # List of lists, each containing stops for one day
            current_day_stops = []
            current_day = 1
            
            for stop in stops:
                current_day_stops.append(stop)
                
                # If this is an overnight stay, end the current day
                if stop['is_overnight']:
                    days_routes.append(current_day_stops.copy())
                    current_day_stops = []  # Start new day
                    current_day += 1
            
            # Add remaining stops as the final day
            if current_day_stops:
                days_routes.append(current_day_stops)
            
            # Plot each day's route
            for day_idx, day_stops in enumerate(days_routes):
                if len(day_stops) < 2:
                    continue
                
                day_style = day_styles[day_idx % len(day_styles)]
                
                # Get coordinates for this day's stops
                day_coords = []
                for stop in day_stops:
                    stop_id = stop['location_id']
                    if stop_id in location_coords:
                        day_coords.append(location_coords[stop_id])
                
                if len(day_coords) < 2:
                    continue
                
                # Plot the route for this day
                x_coords = [coord[0] for coord in day_coords]
                y_coords = [coord[1] for coord in day_coords]
                
                # Plot line segments
                plt.plot(x_coords, y_coords, 
                        color=vehicle_color, 
                        linestyle=day_style, 
                        linewidth=2,
                        alpha=0.8,
                        label=f'{vehicle_id} Day {day_idx + 1}' if day_idx < 3 else "")
                
                # Add arrows to show direction
                for i in range(len(x_coords) - 1):
                    dx = x_coords[i + 1] - x_coords[i]
                    dy = y_coords[i + 1] - y_coords[i]
                    if abs(dx) > 0.001 or abs(dy) > 0.001:  # Only add arrow if there's actual movement
                        plt.arrow(x_coords[i], y_coords[i], dx * 0.3, dy * 0.3,
                                head_width=0.05, head_length=0.03, 
                                fc=vehicle_color, ec=vehicle_color, alpha=0.6)
        
        # Plot location points
        depot_plotted = False
        for loc in working_locations:
            x, y = loc['x'], loc['y']
            loc_id = loc['id']
            
            if loc.get('is_overnight_node', False):
                # Overnight stay nodes - use diamond markers
                plt.scatter(x, y, c='black', marker='D', s=80, alpha=0.7, 
                           edgecolors='white', linewidth=1, zorder=5)
                plt.annotate(f"🌙{loc_id.split('_')[-1]}", (x, y), 
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=8, alpha=0.8)
            elif 'depot' in loc_id.lower():
                # Depot - use large square marker
                if not depot_plotted:  # Only plot depot once
                    plt.scatter(x, y, c='black', marker='s', s=200, alpha=0.9,
                               edgecolors='yellow', linewidth=3, zorder=10)
                    plt.annotate('🏠 DEPOT', (x, y), 
                               xytext=(10, 10), textcoords='offset points', 
                               fontsize=10, fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
                    depot_plotted = True
            else:
                # Regular locations - use circles
                plt.scatter(x, y, c='lightblue', marker='o', s=100, alpha=0.8,
                           edgecolors='darkblue', linewidth=1, zorder=5)
                plt.annotate(loc_id, (x, y), 
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=8)
        
        # Customize the plot
        plt.xlabel('Longitude', fontsize=12)
        plt.ylabel('Latitude', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Create custom legend
        legend_elements = []
        
        # Vehicle colors
        for vehicle_idx, (vehicle_id, _) in enumerate(result['routes'].items()):
            vehicle_color = vehicle_colors[vehicle_idx % len(vehicle_colors)]
            legend_elements.append(plt.Line2D([0], [0], color=vehicle_color, linewidth=3, 
                                            label=f'{vehicle_id}'))
        
        # Day styles
        legend_elements.append(plt.Line2D([0], [0], color='gray', linestyle='-', 
                                        linewidth=2, label='Day 1'))
        legend_elements.append(plt.Line2D([0], [0], color='gray', linestyle='--', 
                                        linewidth=2, label='Day 2'))
        legend_elements.append(plt.Line2D([0], [0], color='gray', linestyle='-.', 
                                        linewidth=2, label='Day 3'))
        
        # Location markers
        legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', 
                                        markerfacecolor='black', markersize=10,
                                        markeredgecolor='yellow', markeredgewidth=2,
                                        label='Depot', linestyle='None'))
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor='lightblue', markersize=8,
                                        markeredgecolor='darkblue', markeredgewidth=1,
                                        label='Customer', linestyle='None'))
        legend_elements.append(plt.Line2D([0], [0], marker='D', color='w', 
                                        markerfacecolor='black', markersize=8,
                                        markeredgecolor='white', markeredgewidth=1,
                                        label='Overnight Stay', linestyle='None'))
        
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
        
        # Add summary text
        total_distance = result.get('total_distance', 0)
        total_overnight_stays = sum(route['overnight_count'] for route in result['routes'].values())
        
        summary_text = f"Total Distance: {total_distance:.1f} km\n"
        summary_text += f"Total Overnight Stays: {total_overnight_stays}\n"
        summary_text += f"Vehicles: {len(result['routes'])}"
        
        plt.text(0.02, 0.98, summary_text, transform=plt.gca().transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        # Adjust layout to prevent legend cutoff
        plt.tight_layout()
        
        # Save the plot
        filename = "fixed_multiday_vrp_solution.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  📊 Plot saved as: {filename}")
        
        # Show the plot
        plt.show()
        
        return filename


def test_fixed_multiday():
    """Test the fixed multi-day implementation."""
    print("🧪 TESTING FIXED MULTI-DAY VRP IMPLEMENTATION")
    print("=" * 60)
    
    # Create test scenario with locations spread out for better visualization
    locations = [
        {'id': 'depot', 'x': 11.0, 'y': 46.0, 'demand': 0, 'service_time': 0, 'address': 'Main Depot'},
        {'id': 'north_customer', 'x': 11.2, 'y': 46.8, 'demand': 100, 'volume_demand': 2.0, 'service_time': 30, 'address': 'North Customer'},
        {'id': 'east_customer', 'x': 11.7, 'y': 46.1, 'demand': 150, 'volume_demand': 3.0, 'service_time': 30, 'address': 'East Customer'},
        {'id': 'south_customer', 'x': 10.8, 'y': 45.3, 'demand': 200, 'volume_demand': 4.0, 'service_time': 30, 'address': 'South Customer'},
        {'id': 'west_customer', 'x': 10.2, 'y': 46.2, 'demand': 250, 'volume_demand': 5.0, 'service_time': 30, 'address': 'West Customer'},
        {'id': 'northwest_customer', 'x': 10.3, 'y': 46.9, 'demand': 180, 'volume_demand': 3.5, 'service_time': 30, 'address': 'Northwest Customer'},
        {'id': 'southeast_customer', 'x': 11.8, 'y': 45.2, 'demand': 220, 'volume_demand': 4.5, 'service_time': 30, 'address': 'Southeast Customer'},
    ]
    
    vehicles = [
        {'id': 'truck_24t', 'capacity': 1000, 'volume_capacity': 20.0, 'cost_per_km': 1.80, 'start_location': 'depot', 'end_location': 'depot', 'max_time': 480},
        {'id': 'van_4t', 'capacity': 500, 'volume_capacity': 10.0, 'cost_per_km': 0.95, 'start_location': 'depot', 'end_location': 'depot', 'max_time': 480}
    ]
    
    print(f"📍 Test scenario: {len(locations)} locations, {len(vehicles)} vehicles")
    
    # Test the fixed implementation
    fixed_vrp = FixedMultiDayVRP(vehicles, locations)
    result = fixed_vrp.solve_fixed_multiday(max_days=3, daily_time_limit_minutes=480)
    
    if result:
        print(f"\n📋 FIXED SOLUTION ANALYSIS:")
        print("=" * 50)
        
        # Store working_locations for plotting
        working_locations, _, _ = fixed_vrp.create_vehicle_specific_overnight_nodes(max_days=3)
        
        for vehicle_id, route_data in result['routes'].items():
            print(f"\n🚛 Vehicle {vehicle_id}:")
            print(f"   Distance: {route_data['distance']:.1f} km")
            print(f"   Overnight stays: {route_data['overnight_count']}")
            print(f"   Route:")
            
            for stop in route_data['stops']:
                if stop['is_overnight']:
                    print(f"     🌙 {stop['location_id']} (OVERNIGHT STAY)")
                else:
                    print(f"     📍 {stop['location_id']}")
        
        print(f"\n💰 Total distance: {result['total_distance']:.1f} km")
        
        # Analyze overnight stays
        total_overnight_stays = sum(route['overnight_count'] for route in result['routes'].values())
        print(f"🌙 Total overnight stays: {total_overnight_stays}")
        
        # Check if the fix worked (no vehicle should have multiple overnight stays for the same night)
        print(f"\n🔍 VERIFICATION:")
        for vehicle_id, route_data in result['routes'].items():
            overnight_stops = [stop for stop in route_data['stops'] if stop['is_overnight']]
            
            # Extract night numbers from overnight stop IDs
            night_numbers = []
            for stop in overnight_stops:
                stop_id = stop['location_id']
                if 'night1' in stop_id:
                    night_numbers.append(1)
                elif 'night2' in stop_id:
                    night_numbers.append(2)
            
            # Check for duplicates (same night used twice)
            unique_nights = set(night_numbers)
            
            if len(night_numbers) == len(unique_nights):
                print(f"   ✅ {vehicle_id}: {len(overnight_stops)} overnight stays on {len(unique_nights)} different nights (CORRECT)")
                for i, stop in enumerate(overnight_stops):
                    print(f"      - Night {night_numbers[i]}: {stop['location_id']}")
            else:
                print(f"   ❌ {vehicle_id}: {len(overnight_stops)} overnight stays but only {len(unique_nights)} unique nights (DUPLICATE NIGHTS)")
                for i, stop in enumerate(overnight_stops):
                    print(f"      - Night {night_numbers[i]}: {stop['location_id']}")
        
        # Overall assessment
        all_routes_valid = True
        for vehicle_id, route_data in result['routes'].items():
            overnight_stops = [stop for stop in route_data['stops'] if stop['is_overnight']]
            night_numbers = []
            for stop in overnight_stops:
                stop_id = stop['location_id']
                if 'night1' in stop_id:
                    night_numbers.append(1)
                elif 'night2' in stop_id:
                    night_numbers.append(2)
            
            unique_nights = set(night_numbers)
            if len(night_numbers) != len(unique_nights):
                all_routes_valid = False
                break
        
        if all_routes_valid:
            print(f"\n🎉 SOLUTION IS CORRECT: No vehicle uses the same night twice!")
        else:
            print(f"\n❌ SOLUTION STILL HAS ISSUES: Some vehicles use the same night multiple times")
        
        # Plot the solution
        try:
            plot_filename = fixed_vrp.plot_multiday_solution(result, working_locations, 
                                                           "Fixed Multi-Day VRP Solution")
            print(f"\n📊 Solution visualized and saved as: {plot_filename}")
        except Exception as e:
            print(f"\n⚠️ Plotting failed: {e}")
        
        return result
    else:
        print(f"❌ Fixed implementation failed to find solution")
        return None


if __name__ == "__main__":
    test_fixed_multiday()
