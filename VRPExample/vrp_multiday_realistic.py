#!/usr/bin/env python3
"""
Realistic Multi-Day VRP Implementation
=====================================

This implements multi-day VRP where overnight stops are positioned at actual
customer locations where vehicles need to stop due to time constraints.

Based on the MWE approach but with realistic customer locations and scenarios.
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

class RealisticMultiDayVRP:
    """
    Realistic Multi-Day VRP with overnight stops at actual customer locations.
    """
    
    def __init__(self, vehicles, locations):
        self.vehicles = vehicles
        self.locations = locations
        
        # Initialize distance calculator
        self.distance_calculator = OSMDistanceCalculator(self.locations)
        
        self.daily_time_limit_minutes = 480  # 8 hours
        self.max_days = 3
        
    def create_overnight_node_pairs(self):
        """
        Create night/morning node pairs at customer locations.
        
        For each customer location that might require overnight stay, create:
        - Night node: End of day at this location
        - Morning node: Resume from same location next day
        
        Both positioned at the customer's actual coordinates.
        """
        print(f"\n🔧 Creating overnight node pairs at customer locations...")
        
        expanded_locations = self.locations.copy()
        overnight_pairs = {}  # location_idx -> [(night_node_idx, morning_node_idx), ...]
        overnight_node_info = {}  # node_idx -> info dict
        
        # For each customer location (skip depot), create potential overnight nodes  
        for loc_idx, location in enumerate(self.locations):
            if loc_idx == 0:  # Skip depot
                continue
                
            overnight_pairs[loc_idx] = []
            
            # Create night/morning pairs for each potential day
            for day in range(self.max_days - 1):  # max_days-1 nights possible
                # Night node (end of day at customer location)
                night_node_idx = len(expanded_locations)
                night_location = {
                    'id': f'night_{location["id"]}_day{day+1}',
                    'x': location['x'],  # Same coordinates as customer
                    'y': location['y'],
                    'demand': 0,
                    'volume_demand': 0,
                    'service_time': 0,
                    'address': f"Overnight at {location['address']} (Day {day+1})",
                    'time_window': (0, 1440),
                    'is_overnight_node': True,
                    'is_night_node': True,
                    'is_morning_node': False,
                    'original_location_idx': loc_idx,
                    'day': day + 1,
                    'pair_type': 'night'
                }
                expanded_locations.append(night_location)
                overnight_node_info[night_node_idx] = night_location
                
                # Morning node (start of next day at same location)
                morning_node_idx = len(expanded_locations)
                morning_location = {
                    'id': f'morning_{location["id"]}_day{day+2}',
                    'x': location['x'],  # Same coordinates as customer
                    'y': location['y'],
                    'demand': 0,
                    'volume_demand': 0, 
                    'service_time': 0,
                    'address': f"Resume from {location['address']} (Day {day+2})",
                    'time_window': (0, 1440),
                    'is_overnight_node': True,
                    'is_night_node': False,
                    'is_morning_node': True,
                    'original_location_idx': loc_idx,
                    'day': day + 2,
                    'pair_type': 'morning'
                }
                expanded_locations.append(morning_location)
                overnight_node_info[morning_node_idx] = morning_location
                
                # Store the pair
                overnight_pairs[loc_idx].append((night_node_idx, morning_node_idx))
                
                print(f"    Created overnight pair at {location['id']} ({location['x']:.1f}, {location['y']:.1f}):")
                print(f"      Night node {night_node_idx}: {night_location['id']}")
                print(f"      Morning node {morning_node_idx}: {morning_location['id']}")
        
        print(f"  ✅ Expanded from {len(self.locations)} to {len(expanded_locations)} locations")
        
        return expanded_locations, overnight_pairs, overnight_node_info
    
    def create_time_callback(self, manager, expanded_locations):
        """Create time callback with overnight reset logic."""
        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            
            from_loc = expanded_locations[from_node]
            to_loc = expanded_locations[to_node]
            
            # Night to morning transition at same location (overnight rest)
            if (from_loc.get('is_night_node') and to_loc.get('is_morning_node') and
                from_loc.get('original_location_idx') == to_loc.get('original_location_idx')):
                # Reset daily time counter for new day
                return -self.daily_time_limit_minutes * 60
            
            # Calculate travel time between locations
            if from_node < len(self.locations) and to_node < len(self.locations):
                # Both are original locations
                time_minutes = self.distance_calculator.time_matrix[from_node][to_node]
            else:
                # Handle overnight nodes
                from_orig_idx = from_loc.get('original_location_idx', from_node)
                to_orig_idx = to_loc.get('original_location_idx', to_node)
                
                if from_orig_idx < len(self.locations) and to_orig_idx < len(self.locations):
                    time_minutes = self.distance_calculator.time_matrix[from_orig_idx][to_orig_idx]
                else:
                    time_minutes = 1  # Minimal time for transitions
            
            # Add service time
            service_time = to_loc.get('service_time', 0)
            total_time = time_minutes + service_time
            
            return int(total_time * 60)  # Convert to seconds
        
        return time_callback
    
    def solve_realistic_multiday(self):
        """Solve the realistic multi-day VRP."""
        print(f"\n🚀 Realistic Multi-Day VRP Solver")
        print(f"   Vehicles: {len(self.vehicles)}")
        print(f"   Locations: {len(self.locations)}")
        print(f"   Daily limit: {self.daily_time_limit_minutes} minutes")
        
        # Create overnight node pairs
        expanded_locations, overnight_pairs, overnight_node_info = self.create_overnight_node_pairs()
        
        # Create OR-Tools model
        manager = pywrapcp.RoutingIndexManager(len(expanded_locations), len(self.vehicles), 0)
        routing = pywrapcp.RoutingModel(manager)
        
        # Distance callback
        distance_callback = self.create_distance_callback(manager, expanded_locations)
        distance_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(distance_callback_index)
        
        # Capacity constraints
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            if from_node < len(self.locations):
                return self.locations[from_node].get('demand', 0)
            return 0  # Overnight nodes have no demand
        
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        capacities = [vehicle.get('capacity', 1000) for vehicle in self.vehicles]
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index, 0, capacities, True, 'Capacity'
        )
        
        # Time dimension with daily limits and overnight resets
        time_callback = self.create_time_callback(manager, expanded_locations)
        time_callback_index = routing.RegisterTransitCallback(time_callback)
        routing.AddDimension(
            time_callback_index,
            self.daily_time_limit_minutes * 60,  # Max slack (daily reset)
            self.daily_time_limit_minutes * 60 * self.max_days,  # Max cumulative
            True,  # Start at zero
            'Time'
        )
        
        # Add overnight constraints
        self.add_overnight_constraints(routing, manager, expanded_locations, overnight_pairs, overnight_node_info)
        
        # Solve
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_parameters.time_limit.seconds = 60
        
        print(f"\n🔍 Solving...")
        solution = routing.SolveWithParameters(search_parameters)
        
        if solution:
            print(f"✅ Solution found!")
            return self.extract_realistic_solution(routing, manager, solution, expanded_locations, overnight_node_info)
        else:
            print(f"❌ No solution found")
            return None
    
    def create_distance_callback(self, manager, expanded_locations):
        """Create distance callback."""
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            
            # Use original distance matrix for calculations
            if from_node < len(self.locations) and to_node < len(self.locations):
                distance_km = self.distance_calculator.distance_matrix[from_node][to_node]
            else:
                # Handle overnight nodes by using their original location indices
                from_loc = expanded_locations[from_node]
                to_loc = expanded_locations[to_node]
                
                from_orig_idx = from_loc.get('original_location_idx', from_node)
                to_orig_idx = to_loc.get('original_location_idx', to_node)
                
                if from_orig_idx < len(self.locations) and to_orig_idx < len(self.locations):
                    distance_km = self.distance_calculator.distance_matrix[from_orig_idx][to_orig_idx]
                else:
                    distance_km = 0.1  # Small cost for virtual transitions
            
            return int(distance_km * 1000)  # Convert to meters
        
        return distance_callback
    
    def add_overnight_constraints(self, routing, manager, expanded_locations, overnight_pairs, overnight_node_info):
        """Add overnight constraints."""
        print(f"\n🔧 Adding overnight constraints...")
        
        solver = routing.solver()
        
        # 1. Make overnight nodes optional
        for node_idx, node_info in overnight_node_info.items():
            index = manager.NodeToIndex(node_idx)
            routing.AddDisjunction([index], 0)  # Zero penalty
            
        # 2. Ensure night/morning pairing
        for loc_idx, pairs in overnight_pairs.items():
            for night_idx, morning_idx in pairs:
                night_index = manager.NodeToIndex(night_idx)
                morning_index = manager.NodeToIndex(morning_idx)
                
                # Both must be assigned to same vehicle
                solver.Add(routing.VehicleVar(night_index) == routing.VehicleVar(morning_index))
        
        print(f"  ✅ Overnight constraints added")
    
    def extract_realistic_solution(self, routing, manager, solution, expanded_locations, overnight_node_info):
        """Extract and format the realistic solution."""
        routes = {}
        total_distance = 0
        
        print(f"\n📋 REALISTIC SOLUTION ANALYSIS:")
        print("=" * 50)
        
        for vehicle_idx in range(len(self.vehicles)):
            vehicle = self.vehicles[vehicle_idx]
            vehicle_id = vehicle['id']
            
            route_stops = []
            route_distance = 0
            
            index = routing.Start(vehicle_idx)
            
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                location = expanded_locations[node]
                
                stop_info = {
                    'location_id': location['id'],
                    'coordinates': (location['x'], location['y']),
                    'is_overnight': location.get('is_overnight_node', False),
                    'overnight_type': None
                }
                
                if location.get('is_night_node'):
                    stop_info['overnight_type'] = 'night'
                elif location.get('is_morning_node'):
                    stop_info['overnight_type'] = 'morning'
                
                route_stops.append(stop_info)
                
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                
                if not routing.IsEnd(index):
                    route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_idx)
            
            # Add final depot
            final_node = manager.IndexToNode(index)
            route_stops.append({
                'location_id': expanded_locations[final_node]['id'],
                'coordinates': (expanded_locations[final_node]['x'], expanded_locations[final_node]['y']),
                'is_overnight': False,
                'overnight_type': None
            })
            
            routes[vehicle_id] = {
                'stops': route_stops,
                'distance': route_distance / 1000.0,
                'overnight_count': sum(1 for stop in route_stops if stop['overnight_type'] == 'night')
            }
            
            total_distance += route_distance / 1000.0
            
            # Print route details
            print(f"\n🚛 {vehicle_id}:")
            print(f"   Distance: {routes[vehicle_id]['distance']:.1f} km")
            print(f"   Overnight stays: {routes[vehicle_id]['overnight_count']}")
            print(f"   Route:")
            
            for stop in route_stops:
                coords = stop['coordinates']
                if stop['overnight_type'] == 'night':
                    print(f"     🌙 {stop['location_id']} at ({coords[0]:.1f}, {coords[1]:.1f}) (NIGHT)")
                elif stop['overnight_type'] == 'morning':
                    print(f"     🌅 {stop['location_id']} at ({coords[0]:.1f}, {coords[1]:.1f}) (MORNING)")
                else:
                    print(f"     📍 {stop['location_id']} at ({coords[0]:.1f}, {coords[1]:.1f})")
        
        print(f"\n💰 Total distance: {total_distance:.1f} km")
        
        return {
            'status': 'SUCCESS',
            'routes': routes,
            'total_distance': total_distance,
            'expanded_locations': expanded_locations,
            'overnight_node_info': overnight_node_info
        }
    
    def plot_realistic_solution(self, result, title="Realistic Multi-Day VRP Solution"):
        """Plot the realistic solution with overnight stops at actual locations."""
        if not result or result.get('status') != 'SUCCESS':
            print("❌ Cannot plot: No valid solution provided")
            return
        
        print(f"\n🎨 Plotting realistic multi-day solution...")
        
        plt.figure(figsize=(14, 10))
        plt.title(title, fontsize=16, fontweight='bold')
        
        # Colors for vehicles
        vehicle_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        
        # Line styles for different days
        day_styles = ['-', '--', '-.', ':']
        
        # Plot original customer locations
        for i, loc in enumerate(self.locations):
            x, y = loc['x'], loc['y']
            if i == 0:  # Depot
                plt.scatter(x, y, c='black', marker='s', s=300, alpha=0.9,
                           edgecolors='yellow', linewidth=3, zorder=10)
                plt.annotate('🏠 DEPOT', (x, y), xytext=(15, 15), textcoords='offset points',
                           fontsize=12, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
            else:  # Customer
                plt.scatter(x, y, c='lightblue', marker='o', s=200, alpha=0.8,
                           edgecolors='darkblue', linewidth=2, zorder=5)
                plt.annotate(loc['id'], (x, y), xytext=(10, 10), textcoords='offset points',
                           fontsize=10, fontweight='bold')
        
        # Plot vehicle routes
        for vehicle_idx, (vehicle_id, route_data) in enumerate(result['routes'].items()):
            vehicle_color = vehicle_colors[vehicle_idx % len(vehicle_colors)]
            stops = route_data['stops']
            
            if len(stops) < 2:
                continue
            
            print(f"  Plotting {vehicle_id} route with color {vehicle_color}")
            
            # Split route into days based on overnight stops
            days_routes = []
            current_day_stops = []
            
            for stop in stops:
                current_day_stops.append(stop)
                
                # If this is a night stop, end the current day
                if stop['overnight_type'] == 'night':
                    days_routes.append(current_day_stops.copy())
                    current_day_stops = []
                # Skip morning stops for route visualization (same location as night)
                elif stop['overnight_type'] == 'morning':
                    continue
            
            # Add remaining stops as final day
            if current_day_stops:
                days_routes.append(current_day_stops)
            
            # Plot each day's route
            for day_idx, day_stops in enumerate(days_routes):
                if len(day_stops) < 2:
                    continue
                
                day_style = day_styles[day_idx % len(day_styles)]
                
                # Get coordinates for this day
                day_coords = [stop['coordinates'] for stop in day_stops]
                
                if len(day_coords) >= 2:
                    x_coords = [coord[0] for coord in day_coords]
                    y_coords = [coord[1] for coord in day_coords]
                    
                    # Plot route line
                    plt.plot(x_coords, y_coords, color=vehicle_color, linestyle=day_style,
                            linewidth=3, alpha=0.8, 
                            label=f'{vehicle_id} Day {day_idx + 1}' if day_idx < 3 else "")
                    
                    # Add direction arrows
                    for i in range(len(x_coords) - 1):
                        dx = x_coords[i + 1] - x_coords[i]
                        dy = y_coords[i + 1] - y_coords[i]
                        if abs(dx) > 0.01 or abs(dy) > 0.01:
                            plt.arrow(x_coords[i], y_coords[i], dx * 0.3, dy * 0.3,
                                    head_width=0.1, head_length=0.08,
                                    fc=vehicle_color, ec=vehicle_color, alpha=0.7)
            
            # Mark overnight locations with special stars
            overnight_coords = [stop['coordinates'] for stop in stops if stop['overnight_type'] == 'night']
            for coord in overnight_coords:
                plt.scatter(coord[0], coord[1], c='red', marker='*', s=400,
                           alpha=0.9, edgecolors='white', linewidth=2, zorder=15)
                plt.annotate('⭐ OVERNIGHT', coord, xytext=(20, 20),
                           textcoords='offset points', fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.8),
                           color='white', fontweight='bold')
        
        # Customize plot
        plt.xlabel('Longitude', fontsize=12)
        plt.ylabel('Latitude', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Custom legend
        legend_elements = []
        
        # Vehicle colors
        for vehicle_idx, (vehicle_id, _) in enumerate(result['routes'].items()):
            vehicle_color = vehicle_colors[vehicle_idx % len(vehicle_colors)]
            legend_elements.append(plt.Line2D([0], [0], color=vehicle_color, linewidth=3,
                                            label=f'{vehicle_id}'))
        
        # Day styles
        for day_idx in range(min(3, self.max_days)):
            day_style = day_styles[day_idx]
            legend_elements.append(plt.Line2D([0], [0], color='gray', linestyle=day_style,
                                            linewidth=2, label=f'Day {day_idx + 1}'))
        
        # Location markers
        legend_elements.extend([
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='black',
                      markersize=12, markeredgecolor='yellow', markeredgewidth=2,
                      label='Depot', linestyle='None'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue',
                      markersize=10, markeredgecolor='darkblue', markeredgewidth=2,
                      label='Customer', linestyle='None'),
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
                      markersize=15, markeredgecolor='white', markeredgewidth=2,
                      label='Overnight Stay', linestyle='None')
        ])
        
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
        
        # Add summary
        total_distance = result.get('total_distance', 0)
        total_overnight_stays = sum(route['overnight_count'] for route in result['routes'].values())
        
        summary_text = f"🚚 Total Distance: {total_distance:.1f} km\n"
        summary_text += f"🌙 Overnight Stays: {total_overnight_stays}\n"
        summary_text += f"⏰ Daily Limit: {self.daily_time_limit_minutes} min\n"
        summary_text += f"🚛 Vehicles: {len(result['routes'])}"
        
        plt.text(0.02, 0.98, summary_text, transform=plt.gca().transAxes,
                verticalalignment='top', fontsize=11,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.9))
        
        plt.tight_layout()
        
        # Save plot
        filename = "realistic_multiday_vrp_solution.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  📊 Plot saved as: {filename}")
        
        plt.show()
        return filename


def test_realistic_multiday():
    """Test the realistic multi-day VRP implementation."""
    print("🧪 TESTING REALISTIC MULTI-DAY VRP IMPLEMENTATION")
    print("=" * 60)
    
    # Create test scenario with realistic customer distribution
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
    
    # Test the realistic implementation
    realistic_vrp = RealisticMultiDayVRP(vehicles, locations)
    result = realistic_vrp.solve_realistic_multiday()
    
    if result:
        # Plot the solution
        plot_filename = realistic_vrp.plot_realistic_solution(result, 
                                                             "Realistic Multi-Day VRP - Overnight at Customer Locations")
        print(f"\n📊 Solution plotted and saved as: {plot_filename}")
        
        # Verify overnight positioning
        print(f"\n🔍 OVERNIGHT POSITIONING VERIFICATION:")
        print("=" * 50)
        
        for vehicle_id, route_data in result['routes'].items():
            overnight_stops = [stop for stop in route_data['stops'] 
                             if stop['overnight_type'] == 'night']
            
            if overnight_stops:
                print(f"\n🚛 {vehicle_id} overnight stops AT CUSTOMER LOCATIONS:")
                for stop in overnight_stops:
                    coords = stop['coordinates']
                    print(f"   🌙 {stop['location_id']} at ({coords[0]:.1f}, {coords[1]:.1f})")
                    
                    # Find which customer location this corresponds to
                    for orig_loc in locations:
                        if abs(orig_loc['x'] - coords[0]) < 0.01 and abs(orig_loc['y'] - coords[1]) < 0.01:
                            print(f"       📍 This is at customer: {orig_loc['id']} ({orig_loc['address']})")
                            break
            else:
                print(f"\n🚛 {vehicle_id}: No overnight stops needed")
        
        return result
    else:
        print("❌ Failed to solve realistic multi-day VRP")
        return None


if __name__ == "__main__":
    test_realistic_multiday()
