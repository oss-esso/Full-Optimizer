#!/usr/bin/env python3
"""
Multi-Day VRP Minimal Working Example (MWE)
==========================================

This implements a realistic multi-day VRP where overnight stops are created
at actual locations when routes exceed daily driving limits.

Key Features:
- 4 vehicles, 4 locations (one too far for single day)
- 8-hour daily driving limit
- Overnight nodes positioned at actual stop locations
- Night/Morning node pairs for seamless day transitions
- Visual plotting with correct overnight positioning
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import matplotlib.pyplot as plt
import numpy as np
import math

class MultiDayVRPMWE:
    """
    Minimal Working Example of Multi-Day VRP with realistic overnight positioning.
    """
    
    def __init__(self):
        # Define simple test scenario - make it easier to solve
        self.locations = [
            {'id': 'depot', 'x': 0.0, 'y': 0.0, 'demand': 0, 'service_time': 0},
            {'id': 'close_A', 'x': 1.0, 'y': 0.0, 'demand': 50, 'service_time': 30},  # 30min service
            {'id': 'close_B', 'x': 0.0, 'y': 1.0, 'demand': 50, 'service_time': 30},  # 30min service
            {'id': 'far_C', 'x': 6.0, 'y': 0.0, 'demand': 100, 'service_time': 60},  # Far location needing overnight
        ]
        
        self.vehicles = [
            {'id': 'vehicle_1', 'capacity': 1000},
            {'id': 'vehicle_2', 'capacity': 1000},
        ]
        
        # 8-hour daily limit (480 minutes) - but travel to far_C takes 6*60=360min each way
        self.daily_time_limit_minutes = 480
        self.max_days = 2  # Simplified to 2 days
        
        print(f"🎯 MWE Setup:")
        print(f"   Locations: {len(self.locations)}")
        print(f"   Vehicles: {len(self.vehicles)}")
        print(f"   Daily limit: {self.daily_time_limit_minutes} minutes")
        print(f"   Max days: {self.max_days}")
        print(f"   Note: Travel to far_C = 360min each way, needs overnight!")
    
    def calculate_distance(self, loc1, loc2):
        """Calculate Euclidean distance between two locations."""
        dx = loc1['x'] - loc2['x']
        dy = loc1['y'] - loc2['y']
        return math.sqrt(dx * dx + dy * dy)
    
    def calculate_travel_time(self, loc1, loc2):
        """Calculate travel time in minutes (assuming 1 unit distance = 60 minutes)."""
        distance = self.calculate_distance(loc1, loc2)
        return int(distance * 60)  # 60 minutes per unit distance
    
    def create_expanded_node_list(self):
        """
        Create expanded node list with night/morning node pairs.
        
        For each location that might require overnight stay, we create:
        - Night node: End of day at this location
        - Morning node: Resume from same location next day
        """
        print(f"\n🔧 Creating expanded node list with night/morning pairs...")
        
        expanded_locations = self.locations.copy()
        night_morning_pairs = {}  # location_idx -> [(night_node_idx, morning_node_idx), ...]
        overnight_node_info = {}  # node_idx -> info dict
        
        # For each non-depot location, create potential overnight nodes
        for loc_idx, location in enumerate(self.locations):
            if loc_idx == 0:  # Skip depot
                continue
                
            night_morning_pairs[loc_idx] = []
            
            # Create night/morning pairs for each potential day
            for day in range(self.max_days - 1):  # max_days-1 nights possible
                # Night node (end of day)
                night_node_idx = len(expanded_locations)
                night_location = {
                    'id': f'night_{location["id"]}_day{day+1}',
                    'x': location['x'],  # Same coordinates as original location
                    'y': location['y'],
                    'demand': 0,  # No additional demand
                    'service_time': 0,  # No service time
                    'is_night_node': True,
                    'is_morning_node': False,
                    'original_location_idx': loc_idx,
                    'day': day + 1,
                    'pair_type': 'night'
                }
                expanded_locations.append(night_location)
                overnight_node_info[night_node_idx] = night_location
                
                # Morning node (start of next day)
                morning_node_idx = len(expanded_locations)
                morning_location = {
                    'id': f'morning_{location["id"]}_day{day+2}',
                    'x': location['x'],  # Same coordinates as original location
                    'y': location['y'],
                    'demand': 0,  # No additional demand
                    'service_time': 0,  # No service time
                    'is_night_node': False,
                    'is_morning_node': True,
                    'original_location_idx': loc_idx,
                    'day': day + 2,
                    'pair_type': 'morning'
                }
                expanded_locations.append(morning_location)
                overnight_node_info[morning_node_idx] = morning_location
                
                # Store the pair
                night_morning_pairs[loc_idx].append((night_node_idx, morning_node_idx))
                
                print(f"    Created night/morning pair for {location['id']} day {day+1}-{day+2}:")
                print(f"      Night node {night_node_idx}: {night_location['id']} at ({location['x']}, {location['y']})")
                print(f"      Morning node {morning_node_idx}: {morning_location['id']} at ({location['x']}, {location['y']})")
        
        print(f"  ✅ Expanded from {len(self.locations)} to {len(expanded_locations)} nodes")
        
        return expanded_locations, night_morning_pairs, overnight_node_info
    
    def create_time_callback(self, manager, expanded_locations):
        """Create time callback with overnight reset logic."""
        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            
            from_loc = expanded_locations[from_node]
            to_loc = expanded_locations[to_node]
            
            # Calculate base travel time
            if from_node < len(self.locations) and to_node < len(self.locations):
                # Both are original locations
                travel_time = self.calculate_travel_time(from_loc, to_loc)
            elif from_loc.get('is_night_node') and to_loc.get('is_morning_node'):
                # Night to morning transition (same location)
                if (from_loc.get('original_location_idx') == to_loc.get('original_location_idx') and
                    from_loc.get('day') == to_loc.get('day') - 1):
                    # Overnight rest: reset daily time counter
                    return -self.daily_time_limit_minutes * 60  # Reset time dimension
                else:
                    return 999999  # Invalid transition
            elif to_loc.get('is_night_node') or to_loc.get('is_morning_node'):
                # Transition to overnight node
                if to_loc.get('original_location_idx') is not None:
                    orig_to_loc = self.locations[to_loc['original_location_idx']]
                    travel_time = self.calculate_travel_time(from_loc, orig_to_loc)
                else:
                    travel_time = 60  # Default
            elif from_loc.get('is_night_node') or from_loc.get('is_morning_node'):
                # Transition from overnight node
                if from_loc.get('original_location_idx') is not None:
                    orig_from_loc = self.locations[from_loc['original_location_idx']]
                    travel_time = self.calculate_travel_time(orig_from_loc, to_loc)
                else:
                    travel_time = 60  # Default
            else:
                # Default case
                travel_time = 60
            
            # Add service time
            service_time = to_loc.get('service_time', 0)
            total_time = travel_time + service_time
            
            return int(total_time * 60)  # Convert to seconds
        
        return time_callback
    
    def add_overnight_constraints(self, routing, manager, expanded_locations, night_morning_pairs, overnight_node_info):
        """Add simplified constraints for overnight stays."""
        print(f"\n🔧 Adding overnight constraints...")
        
        solver = routing.solver()
        
        # 1. Make overnight nodes optional with zero penalty
        for node_idx, node_info in overnight_node_info.items():
            index = manager.NodeToIndex(node_idx)
            routing.AddDisjunction([index], 0)  # Zero penalty for not using
            print(f"    Made {node_info['id']} optional")
        
        # 2. Simple pairing: if night is used, morning must be used by same vehicle
        for loc_idx, pairs in night_morning_pairs.items():
            for night_idx, morning_idx in pairs:
                night_index = manager.NodeToIndex(night_idx)
                morning_index = manager.NodeToIndex(morning_idx)
                
                # Simple constraint: both must have same vehicle assignment
                solver.Add(routing.VehicleVar(night_index) == routing.VehicleVar(morning_index))
                
                print(f"    Paired night node {night_idx} with morning node {morning_idx}")
        
        print(f"  ✅ Overnight constraints added")
    
    def solve_multiday_mwe(self):
        """Solve the multi-day VRP MWE."""
        print(f"\n🚀 Solving Multi-Day VRP MWE...")
        
        # Create expanded node list
        expanded_locations, night_morning_pairs, overnight_node_info = self.create_expanded_node_list()
        
        # Create OR-Tools model
        manager = pywrapcp.RoutingIndexManager(len(expanded_locations), len(self.vehicles), 0)
        routing = pywrapcp.RoutingModel(manager)
        
        # Distance callback
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            
            from_loc = expanded_locations[from_node]
            to_loc = expanded_locations[to_node]
            
            if from_node < len(self.locations) and to_node < len(self.locations):
                distance = self.calculate_distance(from_loc, to_loc)
            elif (to_loc.get('is_night_node') or to_loc.get('is_morning_node') and 
                  to_loc.get('original_location_idx') is not None):
                orig_to_loc = self.locations[to_loc['original_location_idx']]
                distance = self.calculate_distance(from_loc, orig_to_loc)
            else:
                distance = 0.1  # Small cost for virtual transitions
            
            return int(distance * 1000)  # Convert to meters for OR-Tools
        
        distance_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(distance_callback_index)
        
        # Capacity constraints
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            if from_node < len(self.locations):
                return expanded_locations[from_node].get('demand', 0)
            return 0  # Overnight nodes have no demand
        
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        capacities = [vehicle.get('capacity', 1000) for vehicle in self.vehicles]
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index, 0, capacities, True, 'Capacity'
        )
        
        # Time dimension with daily limits
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
        self.add_overnight_constraints(routing, manager, expanded_locations, night_morning_pairs, overnight_node_info)
        
        # Solve
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_parameters.time_limit.seconds = 30
        
        print(f"\n🔍 Solving...")
        solution = routing.SolveWithParameters(search_parameters)
        
        if solution:
            print(f"✅ Solution found!")
            return self.extract_solution(routing, manager, solution, expanded_locations, overnight_node_info)
        else:
            print(f"❌ No solution found")
            return None
    
    def extract_solution(self, routing, manager, solution, expanded_locations, overnight_node_info):
        """Extract solution with overnight stop analysis."""
        routes = {}
        total_distance = 0
        
        print(f"\n📋 SOLUTION ANALYSIS:")
        print("=" * 50)
        
        for vehicle_idx in range(len(self.vehicles)):
            vehicle = self.vehicles[vehicle_idx]
            vehicle_id = vehicle['id']
            
            route_stops = []
            route_distance = 0
            route_time = 0
            
            index = routing.Start(vehicle_idx)
            
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                location = expanded_locations[node]
                
                stop_info = {
                    'location_id': location['id'],
                    'coordinates': (location['x'], location['y']),
                    'is_overnight': location.get('is_night_node', False) or location.get('is_morning_node', False),
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
                'distance': route_distance / 1000.0,  # Convert back to km
                'overnight_count': sum(1 for stop in route_stops if stop['is_overnight'] and stop['overnight_type'] == 'night')
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
    
    def plot_solution(self, result, title="Multi-Day VRP MWE Solution"):
        """Plot the solution with overnight stops at correct locations."""
        if not result or result.get('status') != 'SUCCESS':
            print("❌ Cannot plot: No valid solution provided")
            return
        
        print(f"\n🎨 Plotting MWE solution...")
        
        plt.figure(figsize=(12, 10))
        plt.title(title, fontsize=16, fontweight='bold')
        
        # Colors for vehicles
        vehicle_colors = ['red', 'blue', 'green', 'orange']
        
        # Plot original locations first
        for i, loc in enumerate(self.locations):
            x, y = loc['x'], loc['y']
            if i == 0:  # Depot
                plt.scatter(x, y, c='black', marker='s', s=200, alpha=0.9,
                           edgecolors='yellow', linewidth=3, zorder=10)
                plt.annotate('DEPOT', (x, y), xytext=(10, 10), textcoords='offset points',
                           fontsize=12, fontweight='bold')
            else:  # Customer
                plt.scatter(x, y, c='lightblue', marker='o', s=150, alpha=0.8,
                           edgecolors='darkblue', linewidth=2, zorder=5)
                plt.annotate(loc['id'], (x, y), xytext=(5, 5), textcoords='offset points',
                           fontsize=10)
        
        # Plot routes
        for vehicle_idx, (vehicle_id, route_data) in enumerate(result['routes'].items()):
            vehicle_color = vehicle_colors[vehicle_idx % len(vehicle_colors)]
            stops = route_data['stops']
            
            if len(stops) < 2:
                continue
            
            print(f"  Plotting {vehicle_id} route with color {vehicle_color}")
            
            # Get coordinates for all stops
            route_coords = []
            overnight_coords = []
            
            for stop in stops:
                coord = stop['coordinates']
                route_coords.append(coord)
                
                # Mark overnight locations
                if stop['overnight_type'] == 'night':
                    overnight_coords.append(coord)
            
            # Plot route line
            if len(route_coords) >= 2:
                x_coords = [coord[0] for coord in route_coords]
                y_coords = [coord[1] for coord in route_coords]
                
                plt.plot(x_coords, y_coords, color=vehicle_color, linewidth=3,
                        alpha=0.7, label=f'{vehicle_id}')
                
                # Add arrows for direction
                for i in range(len(x_coords) - 1):
                    dx = x_coords[i + 1] - x_coords[i]
                    dy = y_coords[i + 1] - y_coords[i]
                    if abs(dx) > 0.1 or abs(dy) > 0.1:
                        plt.arrow(x_coords[i], y_coords[i], dx * 0.3, dy * 0.3,
                                head_width=0.2, head_length=0.15,
                                fc=vehicle_color, ec=vehicle_color, alpha=0.8)
            
            # Plot overnight stops
            for coord in overnight_coords:
                plt.scatter(coord[0], coord[1], c='red', marker='*', s=300,
                           alpha=0.9, edgecolors='white', linewidth=2, zorder=15)
                plt.annotate('OVERNIGHT', coord, xytext=(15, 15),
                           textcoords='offset points', fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7),
                           color='white', fontweight='bold')
        
        # Customize plot
        plt.xlabel('X Coordinate', fontsize=12)
        plt.ylabel('Y Coordinate', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper left')
        
        # Add summary
        total_distance = result.get('total_distance', 0)
        total_overnight_stays = sum(route['overnight_count'] for route in result['routes'].values())
        
        summary_text = f"Total Distance: {total_distance:.1f} km\n"
        summary_text += f"Overnight Stays: {total_overnight_stays}\n"
        summary_text += f"Daily Limit: {self.daily_time_limit_minutes} min"
        
        plt.text(0.02, 0.98, summary_text, transform=plt.gca().transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        filename = "multiday_vrp_mwe_solution.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  📊 Plot saved as: {filename}")
        
        plt.show()
        return filename


def test_multiday_mwe():
    """Test the Multi-Day VRP MWE."""
    print("🧪 TESTING MULTI-DAY VRP MINIMAL WORKING EXAMPLE")
    print("=" * 60)
    
    # Create and solve
    vrp_mwe = MultiDayVRPMWE()
    result = vrp_mwe.solve_multiday_mwe()
    
    if result:
        # Plot the solution
        plot_filename = vrp_mwe.plot_solution(result)
        print(f"\n📊 Solution plotted and saved as: {plot_filename}")
        
        # Verify overnight positioning
        print(f"\n🔍 OVERNIGHT POSITIONING VERIFICATION:")
        print("=" * 40)
        
        for vehicle_id, route_data in result['routes'].items():
            overnight_stops = [stop for stop in route_data['stops'] 
                             if stop['overnight_type'] == 'night']
            
            if overnight_stops:
                print(f"\n🚛 {vehicle_id} overnight stops:")
                for stop in overnight_stops:
                    coords = stop['coordinates']
                    print(f"   🌙 {stop['location_id']} at ({coords[0]:.1f}, {coords[1]:.1f})")
            else:
                print(f"\n🚛 {vehicle_id}: No overnight stops needed")
        
        return result
    else:
        print("❌ Failed to solve MWE")
        return None


if __name__ == "__main__":
    test_multiday_mwe()
