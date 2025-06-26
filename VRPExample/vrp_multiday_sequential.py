#!/usr/bin/env python3
"""
Sequential Multi-Day VRP Implementation
======================================

This implements a day-by-day approach to multi-day VRP:
1. Generate overnight nodes for current day
2. Solve optimization for current day only
3. Keep only overnight locations reached
4. Use overnight locations as starting points for next day
5. Repeat until all days are solved

This approach scales much better for weekly scenarios and is more realistic.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import importlib.util
import matplotlib.pyplot as plt
import numpy as np
import copy

# Import the distance calculator from the original file
spec = importlib.util.spec_from_file_location("vrp_original", 
                                              os.path.join(os.path.dirname(__file__), "vrp_optimizer_clean copy.py"))
vrp_original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vrp_original)
OSMDistanceCalculator = vrp_original.OSMDistanceCalculator

class SequentialMultiDayVRP:
    """
    Sequential Multi-Day VRP that solves day by day.
    """
    
    def __init__(self, vehicles, locations):
        self.vehicles = vehicles
        self.locations = locations
        
        # Initialize distance calculator
        self.distance_calculator = OSMDistanceCalculator(self.locations)
        
        self.daily_time_limit_minutes = 480  # 8 hours per day
        
        # Track the multi-day solution
        self.daily_solutions = {}  # day -> solution
        self.vehicle_states = {}   # vehicle_id -> current_state
        
        # Initialize vehicle states
        for vehicle in self.vehicles:
            self.vehicle_states[vehicle['id']] = {
                'current_location_idx': 0,  # Start at depot
                'overnight_position': None,  # Track overnight positions
                'remaining_capacity': vehicle.get('capacity', 1000),
                'remaining_volume': vehicle.get('volume_capacity', 20.0),
                'is_active': True
            }
    
    def create_overnight_nodes_for_day(self, day_num, unvisited_customers):
        """
        Create overnight nodes for a specific day.
        Only create overnight nodes at unvisited customer locations.
        """
        print(f"\n🌙 Creating overnight nodes for Day {day_num}...")
        
        day_locations = []
        
        # Add depot (always available)
        day_locations.append(self.locations[0])
        
        # Add unvisited customers
        for customer_idx in unvisited_customers:
            day_locations.append(self.locations[customer_idx])
        
        # Add overnight nodes at each customer location
        overnight_node_info = {}
        for i, customer_idx in enumerate(unvisited_customers):
            customer_location = self.locations[customer_idx]
            
            # Create overnight node at this customer location
            overnight_location = {
                'id': f'overnight_{customer_location["id"]}_day{day_num}',
                'x': customer_location['x'],
                'y': customer_location['y'],
                'demand': 0,
                'volume_demand': 0,
                'service_time': 0,
                'address': f"Overnight at {customer_location['address']} (Day {day_num})",
                'time_window': (0, 1440),
                'is_overnight_node': True,
                'original_customer_idx': customer_idx,
                'base_customer_id': customer_location['id']
            }
            
            day_locations.append(overnight_location)
            overnight_node_info[len(day_locations) - 1] = overnight_location
            
            print(f"    Added overnight option at {customer_location['id']} ({customer_location['x']:.1f}, {customer_location['y']:.1f})")
        
        print(f"  ✅ Day {day_num}: {len(unvisited_customers)} customers + {len(overnight_node_info)} overnight options = {len(day_locations)} total locations")
        
        return day_locations, overnight_node_info
    
    def solve_single_day(self, day_num, day_locations, active_vehicles):
        """
        Solve VRP for a single day with adaptive overnight logic.
        No capacity constraints, focus on time limits with dynamic overnight stops.
        """
        print(f"\n🚀 Solving Day {day_num} with {len(active_vehicles)} vehicles...")
        
        if not active_vehicles:
            print("  ⚠️ No active vehicles for this day")
            return None
        
        # Simple approach: try to visit as many customers as possible within time limit
        # If time limit is reached, create overnight stop at current position
        
        day_routes = {}
        visited_customers = set()
        overnight_locations = {}
        
        print(f"\n📋 Day {day_num} Adaptive Solution:")
        print("=" * 40)
        
        available_customers = [loc for i, loc in enumerate(day_locations) if i > 0 and not loc.get('is_overnight_node', False)]
        unvisited = list(range(len(available_customers)))
        
        for vehicle_idx, vehicle in enumerate(active_vehicles):
            vehicle_id = vehicle['id']
            current_time = 0  # Minutes from start of day
            route_stops = []
            route_distance = 0
            total_demand = 0
            
            # Determine starting position - either depot or last overnight position
            vehicle_state = self.vehicle_states[vehicle_id]
            if 'overnight_position' in vehicle_state and vehicle_state['overnight_position'] is not None:
                # Start from overnight position
                start_pos_idx = vehicle_state['overnight_position']
                start_location = self.locations[start_pos_idx]
                current_pos = start_pos_idx
                print(f"  🚛 {vehicle_id} starting from overnight position:")
                print(f"    🌙 Starting at {start_location['id']} ({start_location['x']:.1f}, {start_location['y']:.1f}) | Day {day_num}")
            else:
                # Start from depot
                depot = day_locations[0]
                current_pos = 0
                start_location = depot
                print(f"  🚛 {vehicle_id} starting route:")
                print(f"    ⏰ 00:00 - Start at {depot['id']} | Current driving time: {current_time:.1f} min")
            
            # Add starting point to route
            route_stops.append({
                'location_id': start_location['id'],
                'coordinates': (start_location['x'], start_location['y']),
                'is_overnight': False,
                'demand': 0,
                'original_customer_idx': current_pos,
                'arrival_time': current_time
            })
            
            # Visit customers until time limit reached
            customers_for_this_vehicle = unvisited[:len(unvisited)//len(active_vehicles) + (1 if vehicle_idx < len(unvisited)%len(active_vehicles) else 0)]
            unvisited = unvisited[len(customers_for_this_vehicle):]
            
            if not customers_for_this_vehicle:
                print(f"    No customers assigned to {vehicle_id}")
                # If started from overnight position, stay there; if from depot, stay at depot
                if current_pos != 0:  # Not at depot
                    print(f"    Staying at current position (no customers to visit)")
                else:
                    print(f"    {current_time:05.1f} - No work, staying at depot")
            else:
                for customer_rel_idx in customers_for_this_vehicle:
                    customer = available_customers[customer_rel_idx]
                    customer_abs_idx = customer.get('original_customer_idx', customer_rel_idx + 1)
                    
                    # Get current position coordinates
                    if current_pos < len(self.locations):
                        current_coords = (self.locations[current_pos]['x'], self.locations[current_pos]['y'])
                    else:
                        current_coords = (start_location['x'], start_location['y'])
                    
                    print(f"    📍 Currently at position {current_pos} {current_coords} (driving time so far: {current_time:.1f} min)")
                    
                    # Calculate travel time to this customer
                    try:
                        if current_pos < len(self.locations) and customer_abs_idx < len(self.locations):
                            travel_time = self.distance_calculator.time_matrix[current_pos][customer_abs_idx]
                            travel_distance = self.distance_calculator.distance_matrix[current_pos][customer_abs_idx]
                        else:
                            # Fallback
                            current_loc = self.locations[current_pos] if current_pos < len(self.locations) else start_location
                            dx = current_loc['x'] - customer['x']
                            dy = current_loc['y'] - customer['y']
                            travel_distance = (dx**2 + dy**2)**0.5 * 100
                            travel_time = travel_distance * 1.5
                    except:
                        travel_time = 60  # Default fallback
                        travel_distance = 40
                    
                    service_time = customer.get('service_time', 0)
                    total_stop_time = travel_time + service_time
                    
                    # Check if we can complete this stop and return to depot within time limit
                    return_time = 0
                    try:
                        return_time = self.distance_calculator.time_matrix[customer_abs_idx][0]
                    except:
                        depot_coords = self.locations[0]
                        dx = customer['x'] - depot_coords['x']
                        dy = customer['y'] - depot_coords['y']
                        return_distance = (dx**2 + dy**2)**0.5 * 100
                        return_time = return_distance * 1.5
                    
                    projected_time_after_customer = current_time + total_stop_time
                    projected_time_with_return = projected_time_after_customer + return_time
                    
                    print(f"      🔍 Evaluating customer {customer['id']}:")
                    print(f"         Travel time to customer: {travel_time:.1f} min")
                    print(f"         Service time: {service_time:.1f} min")
                    print(f"         Return time to depot: {return_time:.1f} min")
                    print(f"         Current driving time: {current_time:.1f} min")
                    print(f"         Time after this customer: {projected_time_after_customer:.1f} min")
                    print(f"         Time with return to depot: {projected_time_with_return:.1f} min")
                    print(f"         Daily limit: {self.daily_time_limit_minutes} min")
                    
                    if projected_time_with_return > self.daily_time_limit_minutes:
                        # Time limit would be exceeded - create overnight stop ON THE ROAD towards this customer
                        print(f"      🚨 TIME LIMIT EXCEEDED! Creating overnight stop ON THE ROAD")
                        
                        # Calculate how much time we can use to travel towards this customer
                        remaining_time = self.daily_time_limit_minutes - current_time
                        print(f"         Remaining time available: {remaining_time:.1f} min")
                        
                        # Get current position coordinates
                        if current_pos < len(self.locations):
                            current_coords = (self.locations[current_pos]['x'], self.locations[current_pos]['y'])
                        else:
                            current_coords = (start_location['x'], start_location['y'])
                        
                        if remaining_time > 30:  # Only move if we have meaningful time left
                            # Calculate how far we can travel towards the customer
                            # Assume average speed for calculating position on route
                            max_travel_time = remaining_time - 30  # Leave 30 min buffer for setup
                            travel_ratio = min(max_travel_time / travel_time, 0.8) if travel_time > 0 else 0  # Max 80% of the way
                            
                            # Calculate intermediate position on the road to customer
                            dx = customer['x'] - current_coords[0]
                            dy = customer['y'] - current_coords[1]
                            
                            overnight_x = current_coords[0] + (dx * travel_ratio)
                            overnight_y = current_coords[1] + (dy * travel_ratio)
                            
                            print(f"         Moving {travel_ratio:.1%} of the way towards {customer['id']}")
                            print(f"         From: {current_coords}")
                            print(f"         To overnight position: ({overnight_x:.1f}, {overnight_y:.1f})")
                            
                            # Update time and distance for partial travel
                            partial_travel_time = max_travel_time
                            partial_travel_distance = travel_distance * travel_ratio
                            current_time += partial_travel_time
                            route_distance += partial_travel_distance
                            
                            # Create temporary location for overnight position
                            temp_overnight_idx = len(self.locations)  # Use next available index
                            overnight_location = {
                                'id': f'road_overnight_{vehicle_id}_day{day_num}',
                                'x': overnight_x,
                                'y': overnight_y,
                                'demand': 0,
                                'is_overnight_node': True,
                                'is_road_position': True,
                                'target_customer': customer['id'],
                                'original_customer_idx': temp_overnight_idx
                            }
                            
                            # Add this position to locations temporarily for next day
                            self.locations.append(overnight_location)
                            overnight_locations[vehicle_id] = temp_overnight_idx
                            
                        else:
                            # Not enough time to move significantly, stay at current position
                            print(f"         Not enough time to move significantly, staying at current position")
                            overnight_x = current_coords[0]
                            overnight_y = current_coords[1]
                            
                            overnight_location = {
                                'id': f'overnight_pos_{vehicle_id}_day{day_num}',
                                'x': overnight_x,
                                'y': overnight_y,
                                'demand': 0,
                                'is_overnight_node': True,
                                'original_customer_idx': current_pos
                            }
                            overnight_locations[vehicle_id] = current_pos
                        
                        print(f"         Would NOT visit {customer['id']} today")
                        
                        route_stops.append({
                            'location_id': overnight_location['id'],
                            'coordinates': (overnight_location['x'], overnight_location['y']),
                            'is_overnight': True,
                            'demand': 0,
                            'original_customer_idx': overnight_locations[vehicle_id],
                            'arrival_time': current_time
                        })
                        
                        print(f"      🌙 {vehicle_id} stays overnight ON THE ROAD at ({overnight_location['x']:.1f}, {overnight_location['y']:.1f})")
                        print(f"         Final driving time for day: {current_time:.1f} min")
                        
                        break
                    else:
                        # Normal case - customer can be reached within regular time limit
                        # Visit this customer
                        current_time += total_stop_time
                        route_distance += travel_distance
                        total_demand += customer.get('demand', 0)
                        current_pos = customer_abs_idx
                        
                        visited_customers.add(customer_abs_idx)
                        
                        route_stops.append({
                            'location_id': customer['id'],
                            'coordinates': (customer['x'], customer['y']),
                            'is_overnight': False,
                            'demand': customer.get('demand', 0),
                            'original_customer_idx': customer_abs_idx,
                            'arrival_time': current_time
                        })
                        
                        print(f"      ✅ Visit {customer['id']} (demand: {customer.get('demand', 0)}) - driving time now: {current_time:.1f} min")
                        print(f"         Position after visit: ({customer['x']:.1f}, {customer['y']:.1f})")
            
            # Return to depot if no overnight stop
            if vehicle_id not in overnight_locations:
                try:
                    return_time = self.distance_calculator.time_matrix[current_pos][0] if current_pos < len(self.locations) else 30
                    return_distance = self.distance_calculator.distance_matrix[current_pos][0] if current_pos < len(self.locations) else 20
                except:
                    return_time = 30
                    return_distance = 20
                
                current_time += return_time
                route_distance += return_distance
                
                depot_location = self.locations[0]  # Get depot location
                route_stops.append({
                    'location_id': depot_location['id'],
                    'coordinates': (depot_location['x'], depot_location['y']),
                    'is_overnight': False,
                    'demand': 0,
                    'original_customer_idx': 0,
                    'arrival_time': current_time
                })
                
                print(f"    {current_time:05.1f} - Return to depot")
            
            day_routes[vehicle_id] = {
                'stops': route_stops,
                'distance': route_distance,
                'total_demand': total_demand,
                'ends_with_overnight': vehicle_id in overnight_locations,
                'total_time': current_time
            }
            
            # Update vehicle state
            self.vehicle_states[vehicle_id]['remaining_capacity'] -= total_demand
            
            print(f"  ✅ {vehicle_id}: {route_distance:.1f} km, {total_demand} demand, {current_time:.1f} min, overnight: {vehicle_id in overnight_locations}")
        
        # --- REPORT: Travel time from depot to each location ---
        print(f"\n🕒 Travel time from depot to each location (Day {day_num}):")
        print(f"{'Location':<30} {'Type':<15} {'Time from Depot (min)':>20}")
        print('-'*70)
        depot_idx = 0
        for idx, loc in enumerate(day_locations):
            # Try to get original index for time matrix lookup
            orig_idx = loc.get('original_customer_idx', idx)
            # Fallback: if not in range, use idx
            if orig_idx is None or orig_idx >= len(self.locations):
                orig_idx = idx
            # Try to get time from depot to this location
            try:
                if depot_idx < len(self.locations) and orig_idx < len(self.locations):
                    time_min = self.distance_calculator.time_matrix[depot_idx][orig_idx]
                else:
                    # Fallback: Euclidean distance * 1.5 (min)
                    dx = self.locations[depot_idx]['x'] - loc['x']
                    dy = self.locations[depot_idx]['y'] - loc['y']
                    dist = (dx**2 + dy**2)**0.5 * 100
                    time_min = dist * 1.5
            except Exception:
                time_min = -1
            # Type
            if idx == 0:
                loc_type = 'Depot'
            elif loc.get('is_overnight_node', False):
                loc_type = 'Overnight Node'
            else:
                loc_type = 'Customer'
            print(f"{loc['id']:<30} {loc_type:<15} {time_min:>20.1f}")
        print('-'*70)
        
        return {
            'day': day_num,
            'routes': day_routes,
            'visited_customers': visited_customers,
            'overnight_locations': overnight_locations
        }
    
    def solve_sequential_multiday(self, max_days=3):
        """
        Solve multi-day VRP sequentially, day by day.
        """
        print(f"\n🚀 SEQUENTIAL MULTI-DAY VRP SOLVER")
        print(f"   Vehicles: {len(self.vehicles)}")
        print(f"   Locations: {len(self.locations)}")
        print(f"   Max days: {max_days}")
        print(f"   Daily limit: {self.daily_time_limit_minutes} minutes")
        
        # --- REPORT: Travel time from depot to each location (original scenario) ---
        print(f"\n🕒 Travel time from depot to each location (Scenario Overview):")
        print(f"{'Location':<30} {'Type':<15} {'Time from Depot (min)':>20}")
        print('-'*70)
        depot_idx = 0
        for idx, loc in enumerate(self.locations):
            try:
                if depot_idx < len(self.locations) and idx < len(self.locations):
                    time_min = self.distance_calculator.time_matrix[depot_idx][idx]
                else:
                    dx = self.locations[depot_idx]['x'] - loc['x']
                    dy = self.locations[depot_idx]['y'] - loc['y']
                    dist = (dx**2 + dy**2)**0.5 * 100
                    time_min = dist * 1.5
            except Exception:
                time_min = -1
            if idx == 0:
                loc_type = 'Depot'
            else:
                loc_type = 'Customer'
            print(f"{loc['id']:<30} {loc_type:<15} {time_min:>20.1f}")
        print('-'*70)
        
        # Track unvisited customers (exclude depot)
        unvisited_customers = set(range(1, len(self.locations)))
        all_visited_customers = set()
        
        for day_num in range(1, max_days + 1):
            print(f"\n{'='*60}")
            print(f"🗓️ SOLVING DAY {day_num}")
            print(f"{'='*60}")
            
            if not unvisited_customers:
                print(f"  ✅ All customers visited! Stopping at day {day_num - 1}")
                break
            
            # Get active vehicles for this day
            active_vehicles = []
            for vehicle in self.vehicles:
                if self.vehicle_states[vehicle['id']]['is_active']:
                    active_vehicles.append(vehicle)
            
            if not active_vehicles:
                print(f"  ⚠️ No active vehicles remaining for day {day_num}")
                break

            # --- Check if all unvisited customers can be served in one day (no overnight needed) ---
            # Build day_locations with just depot + unvisited customers
            test_day_locations = [self.locations[0]] + [self.locations[idx] for idx in unvisited_customers]
            # Estimate total demand and total time for all customers
            total_demand = sum(self.locations[idx].get('demand', 0) for idx in unvisited_customers)
            total_capacity = sum(self.vehicle_states[v['id']]['remaining_capacity'] for v in active_vehicles)
            
            # Estimate total time needed (rough approximation)
            total_time_estimate = 0
            for customer_idx in unvisited_customers:
                # Time from depot to customer + service time
                depot_to_customer_time = self.distance_calculator.time_matrix[0][customer_idx]
                service_time = self.locations[customer_idx].get('service_time', 0)
                total_time_estimate += depot_to_customer_time + service_time
            
            # Check if can be solved in one day: capacity OK AND time feasible
            capacity_ok = total_demand <= total_capacity
            time_feasible = total_time_estimate <= self.daily_time_limit_minutes * len(active_vehicles) * 0.8  # 80% utilization
            
            print(f"  📊 Feasibility check:")
            print(f"    Total demand: {total_demand} / {total_capacity} (capacity)")
            print(f"    Est. time: {total_time_estimate:.1f} / {self.daily_time_limit_minutes * len(active_vehicles) * 0.8:.1f} min")
            print(f"    Capacity OK: {capacity_ok}, Time feasible: {time_feasible}")
            
            if capacity_ok and time_feasible:
                print("  🟢 All customers can be served in one day. Skipping overnight nodes.")
                # Create day_locations with proper original_customer_idx mapping
                day_locations = []
                # Add depot
                depot_copy = copy.deepcopy(self.locations[0])
                depot_copy['original_customer_idx'] = 0
                day_locations.append(depot_copy)
                # Add customers with proper mapping
                for customer_idx in unvisited_customers:
                    customer_copy = copy.deepcopy(self.locations[customer_idx])
                    customer_copy['original_customer_idx'] = customer_idx
                    day_locations.append(customer_copy)
            else:
                print("  🔴 Multi-day solution needed (adaptive overnight logic).")
                # Create day_locations with just depot + unvisited customers
                day_locations = []
                # Add depot
                depot_copy = copy.deepcopy(self.locations[0])
                depot_copy['original_customer_idx'] = 0
                day_locations.append(depot_copy)
                # Add customers with proper mapping
                for customer_idx in unvisited_customers:
                    customer_copy = copy.deepcopy(self.locations[customer_idx])
                    customer_copy['original_customer_idx'] = customer_idx
                    day_locations.append(customer_copy)
            
            # Solve this day with adaptive logic
            day_solution = self.solve_single_day(day_num, day_locations, active_vehicles)
            
            if not day_solution:
                print(f"  ❌ Failed to solve day {day_num}")
                break
            
            # Store the solution
            self.daily_solutions[day_num] = day_solution
            
            # Update state for next day
            visited_today = day_solution['visited_customers']
            overnight_vehicles = day_solution['overnight_locations']
            
            # Remove visited customers from unvisited set
            unvisited_customers -= visited_today
            all_visited_customers |= visited_today
            
            # Update vehicle positions for next day
            for vehicle_id, overnight_location_idx in overnight_vehicles.items():
                # Vehicle starts next day at overnight location
                self.vehicle_states[vehicle_id]['current_location_idx'] = overnight_location_idx
                self.vehicle_states[vehicle_id]['overnight_position'] = overnight_location_idx
                print(f"    📍 {vehicle_id} will start Day {day_num + 1} at location {overnight_location_idx}")
            
            # Vehicles that didn't stay overnight return to depot (inactive for remaining days)
            for vehicle in active_vehicles:
                vehicle_id = vehicle['id']
                if vehicle_id not in overnight_vehicles:
                    self.vehicle_states[vehicle_id]['is_active'] = False
                    self.vehicle_states[vehicle_id]['current_location_idx'] = 0  # Return to depot
                    self.vehicle_states[vehicle_id]['overnight_position'] = None
                    print(f"    🏠 {vehicle_id} returned to depot (inactive for remaining days)")
            
            print(f"\n  📊 Day {day_num} Summary:")
            print(f"    Customers visited today: {len(visited_today)}")
            print(f"    Customers remaining: {len(unvisited_customers)}")
            print(f"    Vehicles with overnight: {len(overnight_vehicles)}")
            print(f"    Active vehicles for next day: {sum(1 for v in self.vehicle_states.values() if v['is_active'])}")
        
        print(f"\n🎯 SEQUENTIAL SOLUTION COMPLETE!")
        print(f"   Days solved: {len(self.daily_solutions)}")
        print(f"   Total customers visited: {len(all_visited_customers)} / {len(self.locations) - 1}")
        
        return self.compile_full_solution()
    
    def compile_full_solution(self):
        """Compile all daily solutions into a complete multi-day solution."""
        full_solution = {
            'status': 'SUCCESS',
            'daily_solutions': self.daily_solutions,
            'total_days': len(self.daily_solutions),
            'vehicle_routes': {}
        }
        
        # Compile vehicle routes across all days
        for vehicle in self.vehicles:
            vehicle_id = vehicle['id']
            full_route = []
            total_distance = 0
            total_overnight_stays = 0
            
            for day_num in sorted(self.daily_solutions.keys()):
                day_solution = self.daily_solutions[day_num]
                
                if vehicle_id in day_solution['routes']:
                    day_route = day_solution['routes'][vehicle_id]
                    
                    # Add day marker
                    full_route.append({
                        'location_id': f'=== DAY {day_num} ===',
                        'coordinates': (0, 0),
                        'is_day_marker': True,
                        'is_overnight': False
                    })
                    
                    # Add day's stops
                    full_route.extend(day_route['stops'])
                    total_distance += day_route['distance']
                    
                    if day_route['ends_with_overnight']:
                        total_overnight_stays += 1
            
            full_solution['vehicle_routes'][vehicle_id] = {
                'full_route': full_route,
                'total_distance': total_distance,
                'total_overnight_stays': total_overnight_stays,
                'days_active': len([d for d in self.daily_solutions.keys() 
                                 if vehicle_id in self.daily_solutions[d]['routes']])
            }
        
        return full_solution
    
    def plot_sequential_solution(self, solution, title="Sequential Multi-Day VRP Solution"):
        """Plot the sequential multi-day solution."""
        if not solution or solution.get('status') != 'SUCCESS':
            print("❌ Cannot plot: No valid solution provided")
            return
        
        print(f"\n🎨 Plotting sequential multi-day solution...")
        
        plt.figure(figsize=(16, 12))
        plt.title(title, fontsize=18, fontweight='bold')
        
        # Colors for vehicles and days
        vehicle_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        day_styles = ['-', '--', '-.', ':']
        
        # Plot original customer locations
        for i, loc in enumerate(self.locations):
            x, y = loc['x'], loc['y']
            if i == 0:  # Depot
                plt.scatter(x, y, c='black', marker='s', s=400, alpha=0.9,
                           edgecolors='yellow', linewidth=4, zorder=10)
                plt.annotate('🏠 DEPOT', (x, y), xytext=(15, 15), textcoords='offset points',
                           fontsize=14, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
            else:  # Customer
                plt.scatter(x, y, c='lightblue', marker='o', s=250, alpha=0.8,
                           edgecolors='darkblue', linewidth=2, zorder=5)
                plt.annotate(loc['id'], (x, y), xytext=(10, 10), textcoords='offset points',
                           fontsize=11, fontweight='bold')
        
        # Plot vehicle routes by day
        for vehicle_idx, (vehicle_id, vehicle_data) in enumerate(solution['vehicle_routes'].items()):
            vehicle_color = vehicle_colors[vehicle_idx % len(vehicle_colors)]
            full_route = vehicle_data['full_route']
            
            if len(full_route) < 2:
                continue
            
            print(f"  Plotting {vehicle_id} route with color {vehicle_color}")
            
            # Split route by days
            current_day_route = []
            day_routes = []
            current_day = 1
            
            for stop in full_route:
                if stop.get('is_day_marker'):
                    if current_day_route:
                        day_routes.append((current_day, current_day_route))
                        current_day_route = []
                    # Extract day number more carefully
                    day_str = stop['location_id'].replace('=', '').strip()
                    try:
                        current_day = int(day_str.split()[1])
                    except:
                        current_day += 1  # Fallback increment
                else:
                    current_day_route.append(stop)
            
            # Add final day
            if current_day_route:
                day_routes.append((current_day, current_day_route))
            
            # Plot each day's route
            for day_num, day_route in day_routes:
                if len(day_route) < 2:
                    continue
                
                day_style = day_styles[(day_num - 1) % len(day_styles)]
                
                # Get coordinates
                day_coords = [stop['coordinates'] for stop in day_route]
                
                if len(day_coords) >= 2:
                    x_coords = [coord[0] for coord in day_coords]
                    y_coords = [coord[1] for coord in day_coords]
                    
                    # Plot route line
                    plt.plot(x_coords, y_coords, color=vehicle_color, linestyle=day_style,
                            linewidth=4, alpha=0.8,
                            label=f'{vehicle_id} Day {day_num}' if day_num <= 3 else "")
                    
                    # Add direction arrows
                    for i in range(len(x_coords) - 1):
                        dx = x_coords[i + 1] - x_coords[i]
                        dy = y_coords[i + 1] - y_coords[i]
                        if abs(dx) > 0.01 or abs(dy) > 0.01:
                            plt.arrow(x_coords[i], y_coords[i], dx * 0.3, dy * 0.3,
                                    head_width=0.08, head_length=0.06,
                                    fc=vehicle_color, ec=vehicle_color, alpha=0.8)
                
                # Mark overnight locations
                overnight_coords = [stop['coordinates'] for stop in day_route if stop['is_overnight']]
                for coord in overnight_coords:
                    plt.scatter(coord[0], coord[1], c='red', marker='*', s=500,
                               alpha=0.9, edgecolors='white', linewidth=3, zorder=15)
                    plt.annotate(f'🌙 Day {day_num}', coord, xytext=(25, 25),
                               textcoords='offset points', fontsize=11,
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.8),
                               color='white', fontweight='bold')
        
        # Customize plot
        plt.xlabel('Longitude', fontsize=14)
        plt.ylabel('Latitude', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Custom legend
        legend_elements = []
        
        # Vehicle colors
        for vehicle_idx, (vehicle_id, _) in enumerate(solution['vehicle_routes'].items()):
            vehicle_color = vehicle_colors[vehicle_idx % len(vehicle_colors)]
            legend_elements.append(plt.Line2D([0], [0], color=vehicle_color, linewidth=4,
                                            label=f'{vehicle_id}'))
        
        # Day styles
        for day_idx in range(min(4, solution['total_days'])):
            day_style = day_styles[day_idx]
            legend_elements.append(plt.Line2D([0], [0], color='gray', linestyle=day_style,
                                            linewidth=3, label=f'Day {day_idx + 1}'))
        
        # Location markers
        legend_elements.extend([
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='black',
                      markersize=12, markeredgecolor='yellow', markeredgewidth=3,
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
        total_distance = sum(route['total_distance'] for route in solution['vehicle_routes'].values())
        total_overnight_stays = sum(route['total_overnight_stays'] for route in solution['vehicle_routes'].values())
        
        summary_text =  f" SEQUENTIAL SOLUTION SUMMARY\n"
        summary_text += f" Total Distance: {total_distance:.1f} km\n"
        summary_text += f" Total Overnight Stays: {total_overnight_stays}\n"
        summary_text += f" Days: {solution['total_days']}\n"
        summary_text += f" Vehicles: {len(solution['vehicle_routes'])}\n"
        summary_text += f" Daily Limit: {self.daily_time_limit_minutes} min"
        
        plt.text(0.02, 0.98, summary_text, transform=plt.gca().transAxes,
                verticalalignment='top', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.9))
        
        plt.tight_layout()
        
        # Save plot
        filename = "sequential_multiday_vrp_solution.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  📊 Plot saved as: {filename}")
        
        plt.show()
        return filename


def test_sequential_multiday():
    """Test the sequential multi-day VRP implementation."""
    print("🧪 TESTING SEQUENTIAL MULTI-DAY VRP IMPLEMENTATION")
    print("=" * 60)
    
    # Create test scenario with realistic customer distribution
    locations = [
        {'id': 'depot', 'x': 11.0, 'y': 46.0, 'demand': 0, 'service_time': 0, 'address': 'Main Depot'},
        {'id': 'north_customer', 'x': 11.2, 'y': 46.8, 'demand': 80, 'volume_demand': 2.0, 'service_time': 20, 'address': 'North Customer'},
        {'id': 'east_customer', 'x': 11.7, 'y': 46.1, 'demand': 100, 'volume_demand': 2.5, 'service_time': 20, 'address': 'East Customer'},
        {'id': 'south_customer', 'x': 10.8, 'y': 45.3, 'demand': 120, 'volume_demand': 3.0, 'service_time': 20, 'address': 'South Customer'},
        {'id': 'west_customer', 'x': 10.2, 'y': 46.2, 'demand': 150, 'volume_demand': 3.5, 'service_time': 20, 'address': 'West Customer'},
        {'id': 'far_customer', 'x': 12.5, 'y': 47.2, 'demand': 200, 'volume_demand': 4.0, 'service_time': 30, 'address': 'Far Customer'},
    ]
    
    vehicles = [
        {'id': 'truck_24t', 'capacity': 500, 'volume_capacity': 12.0, 'cost_per_km': 1.80},
        {'id': 'van_4t', 'capacity': 400, 'volume_capacity': 10.0, 'cost_per_km': 0.95}
    ]
    
    print(f"📍 Test scenario: {len(locations)} locations, {len(vehicles)} vehicles")
    print(f"📦 High demand customers requiring multiple days...")
    
    # Test the sequential implementation
    sequential_vrp = SequentialMultiDayVRP(vehicles, locations)
    solution = sequential_vrp.solve_sequential_multiday(max_days=4)
    
    if solution:
        # Plot the solution
        plot_filename = sequential_vrp.plot_sequential_solution(solution, 
                                                               "Sequential Multi-Day VRP - Day by Day Optimization")
        print(f"\n📊 Solution plotted and saved as: {plot_filename}")
        
        # Detailed analysis
        print(f"\n🔍 DETAILED SEQUENTIAL ANALYSIS:")
        print("=" * 50)
        
        for vehicle_id, vehicle_data in solution['vehicle_routes'].items():
            print(f"\n🚛 {vehicle_id}:")
            print(f"   Total distance: {vehicle_data['total_distance']:.1f} km")
            print(f"   Overnight stays: {vehicle_data['total_overnight_stays']}")
            print(f"   Days active: {vehicle_data['days_active']}")
            print(f"   Route summary:")
            
            current_day = None
            for stop in vehicle_data['full_route']:
                if stop.get('is_day_marker'):
                    current_day = stop['location_id']
                    print(f"     {current_day}")
                elif stop['is_overnight']:
                    coords = stop['coordinates']
                    print(f"       🌙 OVERNIGHT at {stop['location_id']} ({coords[0]:.1f}, {coords[1]:.1f})")
                else:
                    coords = stop['coordinates']
                    print(f"       📍 {stop['location_id']} ({coords[0]:.1f}, {coords[1]:.1f})")
        
        return solution
    else:
        print("❌ Failed to solve sequential multi-day VRP")
        return None


if __name__ == "__main__":
    test_sequential_multiday()
