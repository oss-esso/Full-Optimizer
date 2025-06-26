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
from datetime import datetime

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
        
        # Smart customer assignment: distance-aware with vehicle capacity consideration
        vehicle_assignments = self._assign_customers_to_vehicles(active_vehicles, available_customers, unvisited)
        
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
            
            # Get customers assigned to this vehicle
            customers_for_this_vehicle = vehicle_assignments.get(vehicle_id, [])
            
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
                    
                    # Calculate travel time to this customer using precise method
                    travel_time = self._get_precise_travel_time(current_pos, customer_abs_idx)
                    travel_distance = 0
                    try:
                        if current_pos < len(self.locations) and customer_abs_idx < len(self.locations):
                            travel_distance = self.distance_calculator.distance_matrix[current_pos][customer_abs_idx]
                        else:
                            # Fallback distance calculation
                            current_loc = self.locations[current_pos] if current_pos < len(self.locations) else start_location
                            dx = current_loc['x'] - customer['x']
                            dy = current_loc['y'] - customer['y']
                            travel_distance = (dx**2 + dy**2)**0.5 * 100
                    except:
                        travel_distance = 40.0  # Default fallback
                    
                    # Get precise service time
                    service_time = self._get_precise_service_time(customer_abs_idx)
                    total_stop_time = travel_time + service_time
                    
                    # Calculate precise return time to depot
                    return_time = self._get_precise_travel_time(customer_abs_idx, 0)
                    
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
                # Use precise time calculation for return to depot
                return_time = self._get_precise_travel_time(current_pos, 0)
                try:
                    return_distance = self.distance_calculator.distance_matrix[current_pos][0] if current_pos < len(self.locations) else 20.0
                except:
                    return_distance = 20.0
                
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
            
            # Calculate precise metrics for this route
            precise_metrics = self._calculate_precise_route_metrics(route_stops, include_detailed_breakdown=True)
            
            # Update vehicle state
            self.vehicle_states[vehicle_id]['remaining_capacity'] -= total_demand
            
            print(f"  ✅ {vehicle_id}: {route_distance:.1f} km, {total_demand} demand, {current_time:.1f} min, overnight: {vehicle_id in overnight_locations}")
            print(f"     📊 Precise metrics: Travel: {precise_metrics['travel_time']:.1f}min, Service: {precise_metrics['service_time']:.1f}min")
            print(f"     ⚡ Time efficiency: {precise_metrics['time_utilization']:.1%} of daily limit")
            
            # Store precise metrics in route data
            day_routes[vehicle_id]['precise_metrics'] = precise_metrics
        
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
            
            # Vehicles that didn't stay overnight return to depot (remain active for next day)
            for vehicle in active_vehicles:
                vehicle_id = vehicle['id']
                if vehicle_id not in overnight_vehicles:
                    # Keep vehicle active but reset to depot
                    self.vehicle_states[vehicle_id]['is_active'] = True  # Keep active!
                    self.vehicle_states[vehicle_id]['current_location_idx'] = 0  # Return to depot
                    self.vehicle_states[vehicle_id]['overnight_position'] = None
                    print(f"    🏠 {vehicle_id} returned to depot (available for next day)")
                    
            # Only mark vehicles as inactive if there are no more customers
            remaining_customers = len(unvisited_customers)
            if remaining_customers == 0:
                for vehicle in active_vehicles:
                    vehicle_id = vehicle['id']
                    if vehicle_id not in overnight_vehicles:
                        self.vehicle_states[vehicle_id]['is_active'] = False
                        print(f"    🏁 {vehicle_id} finished - no more customers remaining")
            
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
        """Plot the sequential multi-day solution with enhanced visualization."""
        if not solution or solution.get('status') != 'SUCCESS':
            print("❌ Cannot plot: No valid solution provided")
            return
        
        print(f"\n🎨 Plotting sequential multi-day solution...")
        
        plt.figure(figsize=(20, 14))
        plt.title(title, fontsize=20, fontweight='bold', pad=20)
        
        # Enhanced color schemes
        # Different colors for each day
        day_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
        # Different line styles for each vehicle type
        vehicle_styles = {
            'truck': '-',      # Solid line for trucks
            'van': '--',       # Dashed line for vans  
            'sprinter': '-.',  # Dash-dot line for sprinters
            'default': ':'     # Dotted line for others
        }
        # Different markers for vehicle types
        vehicle_markers = {
            'truck': 's',      # Square for trucks
            'van': '^',        # Triangle for vans
            'sprinter': 'D',   # Diamond for sprinters
            'default': 'o'     # Circle for others
        }
        
        # Plot original customer locations with enhanced styling
        for i, loc in enumerate(self.locations):
            x, y = loc['x'], loc['y']
            if i == 0:  # Depot
                plt.scatter(x, y, c='black', marker='s', s=600, alpha=0.9,
                           edgecolors='yellow', linewidth=5, zorder=20)
                plt.annotate('🏠 DEPOT', (x, y), xytext=(20, 20), textcoords='offset points',
                           fontsize=16, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.9),
                           arrowprops=dict(arrowstyle='->', lw=2))
            else:  # Customer
                plt.scatter(x, y, c='lightblue', marker='o', s=300, alpha=0.8,
                           edgecolors='darkblue', linewidth=2, zorder=10)
                plt.annotate(loc['id'], (x, y), xytext=(12, 12), textcoords='offset points',
                           fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        
        # Helper function to get vehicle style
        def get_vehicle_style(vehicle_id):
            vehicle_lower = vehicle_id.lower()
            if 'truck' in vehicle_lower:
                return vehicle_styles['truck'], vehicle_markers['truck']
            elif 'van' in vehicle_lower:
                return vehicle_styles['van'], vehicle_markers['van']
            elif 'sprinter' in vehicle_lower:
                return vehicle_styles['sprinter'], vehicle_markers['sprinter']
            else:
                return vehicle_styles['default'], vehicle_markers['default']
        
        # Plot vehicle routes by day with enhanced styling
        for vehicle_idx, (vehicle_id, vehicle_data) in enumerate(solution['vehicle_routes'].items()):
            full_route = vehicle_data['full_route']
            
            if len(full_route) < 2:
                continue
            
            # Get vehicle-specific styling
            vehicle_line_style, vehicle_marker = get_vehicle_style(vehicle_id)
            
            print(f"  Plotting {vehicle_id} route with style {vehicle_line_style} and marker {vehicle_marker}")
            
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
            
            # Plot each day's route with day-specific colors
            for day_num, day_route in day_routes:
                if len(day_route) < 2:
                    continue
                
                # Use day-specific color
                day_color = day_colors[(day_num - 1) % len(day_colors)]
                
                # Get coordinates
                day_coords = [stop['coordinates'] for stop in day_route]
                
                if len(day_coords) >= 2:
                    x_coords = [coord[0] for coord in day_coords]
                    y_coords = [coord[1] for coord in day_coords]
                    
                    # Plot route line with enhanced styling
                    plt.plot(x_coords, y_coords, color=day_color, linestyle=vehicle_line_style,
                            linewidth=5, alpha=0.8, 
                            label=f'{vehicle_id} Day {day_num}')
                    
                    # Add route markers at stops
                    plt.scatter(x_coords[1:-1], y_coords[1:-1], 
                              c=day_color, marker=vehicle_marker, s=80, alpha=0.9,
                              edgecolors='white', linewidth=1, zorder=15)
                    
                    # Add enhanced direction arrows
                    for i in range(len(x_coords) - 1):
                        dx = x_coords[i + 1] - x_coords[i]
                        dy = y_coords[i + 1] - y_coords[i]
                        if abs(dx) > 0.01 or abs(dy) > 0.01:
                            # Calculate arrow position (2/3 along the segment)
                            arrow_x = x_coords[i] + dx * 0.67
                            arrow_y = y_coords[i] + dy * 0.67
                            plt.arrow(arrow_x, arrow_y, dx * 0.15, dy * 0.15,
                                    head_width=0.03, head_length=0.02,
                                    fc=day_color, ec=day_color, alpha=0.9, zorder=12)
                
                # Mark overnight locations with enhanced styling
                overnight_coords = [stop['coordinates'] for stop in day_route if stop['is_overnight']]
                for coord in overnight_coords:
                    plt.scatter(coord[0], coord[1], c='red', marker='*', s=800,
                               alpha=0.95, edgecolors='white', linewidth=4, zorder=25)
                    plt.annotate(f'🌙 {vehicle_id}\nDay {day_num}', coord, xytext=(30, 30),
                               textcoords='offset points', fontsize=12,
                               bbox=dict(boxstyle="round,pad=0.4", facecolor="red", alpha=0.9),
                               color='white', fontweight='bold',
                               arrowprops=dict(arrowstyle='->', color='red', lw=2))
        
        # Customize plot with enhanced styling
        plt.xlabel('Longitude', fontsize=16, fontweight='bold')
        plt.ylabel('Latitude', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.4, linestyle='--')
        
        # Create enhanced legend
        legend_elements = []
        
        # Vehicle type legend (by line style and marker)
        unique_vehicle_types = set()
        for vehicle_id in solution['vehicle_routes'].keys():
            vehicle_lower = vehicle_id.lower()
            if 'truck' in vehicle_lower and 'truck' not in unique_vehicle_types:
                unique_vehicle_types.add('truck')
                legend_elements.append(plt.Line2D([0], [0], color='gray', linestyle=vehicle_styles['truck'],
                                                marker=vehicle_markers['truck'], linewidth=4, markersize=8,
                                                label='Truck Routes'))
            elif 'van' in vehicle_lower and 'van' not in unique_vehicle_types:
                unique_vehicle_types.add('van')
                legend_elements.append(plt.Line2D([0], [0], color='gray', linestyle=vehicle_styles['van'],
                                                marker=vehicle_markers['van'], linewidth=4, markersize=8,
                                                label='Van Routes'))
            elif 'sprinter' in vehicle_lower and 'sprinter' not in unique_vehicle_types:
                unique_vehicle_types.add('sprinter')
                legend_elements.append(plt.Line2D([0], [0], color='gray', linestyle=vehicle_styles['sprinter'],
                                                marker=vehicle_markers['sprinter'], linewidth=4, markersize=8,
                                                label='Sprinter Routes'))
        
        # Day color legend
        for day_idx in range(min(len(day_colors), solution['total_days'])):
            day_color = day_colors[day_idx]
            legend_elements.append(plt.Line2D([0], [0], color=day_color, linewidth=5,
                                            label=f'Day {day_idx + 1}'))
        
        # Location markers
        legend_elements.extend([
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='black',
                      markersize=15, markeredgecolor='yellow', markeredgewidth=3,
                      label='Depot', linestyle='None'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue',
                      markersize=12, markeredgecolor='darkblue', markeredgewidth=2,
                      label='Customer', linestyle='None'),
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
                      markersize=20, markeredgecolor='white', markeredgewidth=3,
                      label='Overnight Stay', linestyle='None')
        ])
        
        # Create legend with multiple columns for better organization
        plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), 
                  fontsize=12, ncol=1, framealpha=0.9)
        
        # Add enhanced summary box
        total_distance = sum(route['total_distance'] for route in solution['vehicle_routes'].values())
        total_overnight_stays = sum(route['total_overnight_stays'] for route in solution['vehicle_routes'].values())
        
        summary_text =  f"📊 ENHANCED SEQUENTIAL SOLUTION SUMMARY\n"
        summary_text += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        summary_text += f"🚛 Total Distance: {total_distance:.1f} km\n"
        summary_text += f"🌙 Total Overnight Stays: {total_overnight_stays}\n"
        summary_text += f"📅 Days Required: {solution['total_days']}\n"
        summary_text += f"🚐 Active Vehicles: {len(solution['vehicle_routes'])}\n"
        summary_text += f"⏱️  Daily Time Limit: {self.daily_time_limit_minutes} min\n"
        summary_text += f"📍 Customers Served: {len(self.locations) - 1}\n"
        
        # Add cost estimation if available
        total_cost = 0
        for vehicle_id, route_data in solution['vehicle_routes'].items():
            vehicle = next((v for v in self.vehicles if v['id'] == vehicle_id), None)
            if vehicle and 'cost_per_km' in vehicle:
                total_cost += route_data['total_distance'] * vehicle['cost_per_km']
        
        if total_cost > 0:
            summary_text += f"💰 Estimated Cost: {total_cost:.2f} CHF"
        
        plt.text(0.02, 0.98, summary_text, transform=plt.gca().transAxes,
                verticalalignment='top', fontsize=13, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.8", facecolor="lightblue", alpha=0.95, edgecolor='darkblue'))
        
        plt.tight_layout()
        
        # Save plot with enhanced filename
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"enhanced_sequential_vrp_new_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"  📊 Enhanced plot saved as: {filename}")
        
        plt.show()
        return filename
    
    def print_detailed_time_analysis(self, solution):
        """
        Print detailed time analysis inspired by the original VRP optimizer's precision.
        """
        print(f"\n🕰️ DETAILED TIME ANALYSIS (High Precision):")
        print("=" * 60)
        
        total_travel_time = 0
        total_service_time = 0
        total_efficiency = 0
        active_vehicles = 0
        
        for vehicle_id, vehicle_data in solution['vehicle_routes'].items():
            if vehicle_data['days_active'] > 0:
                active_vehicles += 1
                print(f"\n🚛 {vehicle_id.upper()} - Detailed Time Breakdown:")
                print("-" * 45)
                
                daily_travel_times = []
                daily_service_times = []
                daily_efficiencies = []
                
                # Analyze each day's precise metrics
                for day_num in sorted(self.daily_solutions.keys()):
                    day_solution = self.daily_solutions[day_num]
                    
                    if vehicle_id in day_solution['routes']:
                        day_route = day_solution['routes'][vehicle_id]
                        
                        if 'precise_metrics' in day_route:
                            metrics = day_route['precise_metrics']
                            
                            daily_travel_times.append(metrics['travel_time'])
                            daily_service_times.append(metrics['service_time'])
                            daily_efficiencies.append(metrics['time_utilization'])
                            
                            print(f"  📅 Day {day_num}:")
                            print(f"     🛣️  Travel Time: {metrics['travel_time']:.1f} min")
                            print(f"     🔧 Service Time: {metrics['service_time']:.1f} min")
                            print(f"     ⏱️  Total Time: {metrics['total_time']:.1f} min")
                            print(f"     ⚡ Efficiency: {metrics['time_utilization']:.1%}")
                            print(f"     👥 Customers: {metrics['customer_count']}")
                            
                            if metrics['customer_count'] > 0:
                                print(f"     📊 Avg Time/Customer: {metrics['avg_time_per_customer']:.1f} min")
                
                # Vehicle summary
                vehicle_total_travel = sum(daily_travel_times)
                vehicle_total_service = sum(daily_service_times)
                vehicle_avg_efficiency = sum(daily_efficiencies) / len(daily_efficiencies) if daily_efficiencies else 0
                
                total_travel_time += vehicle_total_travel
                total_service_time += vehicle_total_service
                total_efficiency += vehicle_avg_efficiency
                
                print(f"  📈 Vehicle Summary:")
                print(f"     Total Travel: {vehicle_total_travel:.1f} min")
                print(f"     Total Service: {vehicle_total_service:.1f} min")
                print(f"     Avg Daily Efficiency: {vehicle_avg_efficiency:.1%}")
        
        # Overall analysis
        if active_vehicles > 0:
            avg_efficiency = total_efficiency / active_vehicles
            print(f"\n📊 FLEET TIME EFFICIENCY ANALYSIS:")
            print("-" * 40)
            print(f"Fleet Travel Time: {total_travel_time:.1f} min")
            print(f"Fleet Service Time: {total_service_time:.1f} min") 
            print(f"Fleet Total Time: {total_travel_time + total_service_time:.1f} min")
            print(f"Average Fleet Efficiency: {avg_efficiency:.1%}")
            print(f"Time per Vehicle: {(total_travel_time + total_service_time) / active_vehicles:.1f} min")
            
            # Time distribution analysis
            if total_travel_time + total_service_time > 0:
                travel_percentage = total_travel_time / (total_travel_time + total_service_time) * 100
                service_percentage = total_service_time / (total_travel_time + total_service_time) * 100
                print(f"\n⚖️ TIME DISTRIBUTION:")
                print(f"Travel: {travel_percentage:.1f}% | Service: {service_percentage:.1f}%")
                
                if travel_percentage > 70:
                    print("⚠️  High travel time ratio - consider route optimization")
                elif service_percentage > 60:
                    print("ℹ️  High service time ratio - efficient routing")
    
    def _assign_customers_to_vehicles(self, active_vehicles, available_customers, unvisited_indices):
        """
        OR-Tools-based optimal customer assignment strategy.
        Uses assignment problem solver to find optimal vehicle-customer pairing.
        Considers vehicle suitability, distances, and load balancing.
        """
        print(f"  🧠 OR-Tools Optimal Assignment: {len(available_customers)} customers to {len(active_vehicles)} vehicles")
        
        if not available_customers or not active_vehicles:
            return {v['id']: [] for v in active_vehicles}
        
        # Prepare vehicle information
        vehicle_info = []
        for vehicle in active_vehicles:
            capacity = vehicle.get('capacity', 1000)
            vehicle_type = vehicle['id']
            is_truck = 'truck' in vehicle_type.lower()
            
            # Get starting position for distance calculations
            vehicle_state = self.vehicle_states[vehicle['id']]
            if 'overnight_position' in vehicle_state and vehicle_state['overnight_position'] is not None:
                start_pos = vehicle_state['overnight_position']
            else:
                start_pos = 0  # Depot
                
            vehicle_info.append({
                'id': vehicle['id'],
                'capacity': capacity,
                'start_pos': start_pos,
                'is_truck': is_truck,
                'type': vehicle_type
            })
        
        # Prepare customer information
        customer_info = []
        for idx in unvisited_indices:
            customer = available_customers[idx]
            customer_abs_idx = customer.get('original_customer_idx', idx + 1)
            demand = customer.get('demand', 0)
            
            customer_info.append({
                'idx': idx,
                'id': customer['id'],
                'abs_idx': customer_abs_idx,
                'demand': demand,
                'coordinates': (customer['x'], customer['y'])
            })
        
        # Create assignment problem using OR-Tools
        from ortools.linear_solver import pywraplp
        
        # Create the solver
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            print("  ⚠️  OR-Tools solver not available, falling back to simple assignment")
            return self._fallback_assignment(active_vehicles, available_customers, unvisited_indices)
        
        # Create decision variables: x[i][j] = 1 if customer i is assigned to vehicle j
        num_customers = len(customer_info)
        num_vehicles = len(vehicle_info)
        
        x = {}
        for i in range(num_customers):
            for j in range(num_vehicles):
                x[i, j] = solver.IntVar(0, 1, f'x_{i}_{j}')
        
        # Constraint: Each customer is assigned to exactly one vehicle
        for i in range(num_customers):
            solver.Add(solver.Sum([x[i, j] for j in range(num_vehicles)]) == 1)
        
        # Optional: Add workload balancing constraints
        # Don't let one vehicle get overloaded while others are idle
        max_customers_per_vehicle = max(2, (num_customers + num_vehicles - 1) // num_vehicles + 1)
        for j in range(num_vehicles):
            solver.Add(solver.Sum([x[i, j] for i in range(num_customers)]) <= max_customers_per_vehicle)
        
        # Calculate cost matrix (distance + vehicle suitability + workload balancing)
        cost_matrix = []
        
        # Calculate workload for each vehicle (customers already assigned from previous days)
        vehicle_workloads = []
        for j, vehicle in enumerate(vehicle_info):
            total_customers_assigned = 0
            vehicle_id = vehicle['id']
            # Count customers assigned to this vehicle in previous days
            for day_num, day_solution in self.daily_solutions.items():
                if isinstance(day_solution, dict) and 'routes' in day_solution:
                    if vehicle_id in day_solution['routes']:
                        route = day_solution['routes'][vehicle_id]['stops']
                        # Count customer stops (not depot or overnight)
                        customer_stops = [stop for stop in route if not stop.get('is_overnight', False) and stop['location_id'] != 'depot']
                        total_customers_assigned += len(customer_stops)
            vehicle_workloads.append(total_customers_assigned)
        
        for i, customer in enumerate(customer_info):
            customer_costs = []
            for j, vehicle in enumerate(vehicle_info):
                
                # Calculate distance cost
                try:
                    if vehicle['start_pos'] < len(self.locations):
                        distance = self.distance_calculator.distance_matrix[vehicle['start_pos']][customer['abs_idx']]
                    else:
                        # Fallback calculation
                        start_loc = self.locations[vehicle['start_pos']] if vehicle['start_pos'] < len(self.locations) else self.locations[0]
                        dx = start_loc['x'] - customer['coordinates'][0]
                        dy = start_loc['y'] - customer['coordinates'][1]
                        distance = (dx**2 + dy**2)**0.5 * 100
                except:
                    distance = 100.0
                
                # Base cost is distance
                cost = distance
                
                # Add vehicle suitability factors (stronger preferences)
                # Trucks are better for long distances, vans for short
                if distance > 150 and vehicle['is_truck']:
                    cost *= 0.6  # 40% discount for trucks on long routes
                elif distance <= 100 and not vehicle['is_truck']:
                    cost *= 0.7  # 30% discount for vans on short routes
                elif distance > 200 and not vehicle['is_truck']:
                    cost *= 1.5  # 50% penalty for vans on very long routes
                elif distance <= 80 and vehicle['is_truck']:
                    cost *= 1.3  # 30% penalty for trucks on very short routes
                
                # Add workload balancing factor (prefer vehicles with less prior work)
                workload_penalty = vehicle_workloads[j] * 25  # Penalty per customer already handled
                cost += workload_penalty
                
                # Add capacity consideration
                if customer['demand'] > 100 and vehicle['is_truck']:
                    cost *= 0.9  # Small bonus for trucks handling high-demand customers
                elif customer['demand'] > 120 and not vehicle['is_truck']:
                    cost *= 1.2  # Penalty for vans handling very high demand
                
                customer_costs.append(int(cost * 100))  # Scale to integers
            cost_matrix.append(customer_costs)
        
        # Set objective: minimize total assignment cost
        objective_terms = []
        for i in range(num_customers):
            for j in range(num_vehicles):
                objective_terms.append(cost_matrix[i][j] * x[i, j])
        solver.Minimize(solver.Sum(objective_terms))
        
        # Solve the assignment problem
        print(f"  🔍 Solving assignment problem with {num_customers} customers and {num_vehicles} vehicles...")
        status = solver.Solve()
        
        # Extract results
        assignments = {v['id']: [] for v in vehicle_info}
        
        if status == pywraplp.Solver.OPTIMAL:
            print(f"  ✅ Optimal assignment found! Total cost: {solver.Objective().Value():.0f}")
            
            for i in range(num_customers):
                for j in range(num_vehicles):
                    if x[i, j].solution_value() > 0.5:  # Assigned
                        vehicle_id = vehicle_info[j]['id']
                        customer_idx = customer_info[i]['idx']
                        assignments[vehicle_id].append(customer_idx)
                        
                        distance = cost_matrix[i][j] / 100.0
                        print(f"    → {customer_info[i]['id']} → {vehicle_id} (cost: {distance:.1f})")
        else:
            print(f"  ⚠️  Assignment solver failed (status: {status}), using fallback")
            return self._fallback_assignment(active_vehicles, available_customers, unvisited_indices)
        
        # Print assignment summary
        total_customers = sum(len(customers) for customers in assignments.values())
        print(f"  📊 Assignment Summary: {total_customers}/{len(customer_info)} customers assigned")
        for vehicle_id, customers in assignments.items():
            vehicle_type = next(v['type'] for v in vehicle_info if v['id'] == vehicle_id)
            print(f"    {vehicle_id} ({vehicle_type}): {len(customers)} customers")
        
        return assignments
    
    def _fallback_assignment(self, active_vehicles, available_customers, unvisited_indices):
        """
        Simple fallback assignment when OR-Tools is not available.
        """
        assignments = {v['id']: [] for v in active_vehicles}
        
        # Simple round-robin assignment
        for i, idx in enumerate(unvisited_indices):
            vehicle_idx = i % len(active_vehicles)
            vehicle_id = active_vehicles[vehicle_idx]['id']
            assignments[vehicle_id].append(idx)
            
        return assignments
    
    # ...existing code...
    
    def _calculate_precise_route_metrics(self, route_stops, include_detailed_breakdown=True):
        """
        Calculate precise metrics for a route including detailed time breakdown.
        Inspired by the original VRP optimizer's precision tracking.
        """
        metrics = {
            'total_distance': 0.0,
            'total_time': 0.0,
            'travel_time': 0.0,
            'service_time': 0.0,
            'stop_count': len(route_stops),
            'customer_count': 0
        }
        
        if include_detailed_breakdown:
            metrics['time_breakdown'] = []
            metrics['cumulative_times'] = []
        
        current_time = 0.0
        
        for i in range(len(route_stops) - 1):
            current_stop = route_stops[i]
            next_stop = route_stops[i + 1]
            
            # Get original customer indices for matrix lookup
            current_idx = current_stop.get('original_customer_idx', 0)
            next_idx = next_stop.get('original_customer_idx', 0)
            
            # Calculate travel time and distance
            try:
                if (current_idx < len(self.locations) and 
                    next_idx < len(self.locations) and
                    hasattr(self.distance_calculator, 'time_matrix')):
                    travel_time = self.distance_calculator.time_matrix[current_idx][next_idx]
                    travel_distance = self.distance_calculator.distance_matrix[current_idx][next_idx]
                else:
                    # Fallback calculation
                    dx = next_stop['coordinates'][0] - current_stop['coordinates'][0]
                    dy = next_stop['coordinates'][1] - current_stop['coordinates'][1]
                    travel_distance = (dx**2 + dy**2)**0.5 * 100
                    travel_time = travel_distance * 1.5  # Assume 40 km/h average speed
            except:
                travel_time = 30.0  # Default fallback
                travel_distance = 20.0
            
            # Add travel time
            current_time += travel_time
            metrics['total_distance'] += travel_distance
            metrics['travel_time'] += travel_time
            
            # Add service time at destination (skip depot)
            if not next_stop.get('is_overnight', False) and next_idx != 0:
                # Get service time from location data
                if next_idx < len(self.locations):
                    service_time = self.locations[next_idx].get('service_time', 20.0)
                else:
                    service_time = 20.0  # Default service time
                
                current_time += service_time
                metrics['service_time'] += service_time
                metrics['customer_count'] += 1
            
            if include_detailed_breakdown:
                segment_info = {
                    'from': current_stop['location_id'],
                    'to': next_stop['location_id'],
                    'travel_time': travel_time,
                    'travel_distance': travel_distance,
                    'service_time': service_time if not next_stop.get('is_overnight', False) and next_idx != 0 else 0,
                    'cumulative_time': current_time
                }
                metrics['time_breakdown'].append(segment_info)
                metrics['cumulative_times'].append(current_time)
        
        metrics['total_time'] = current_time
        
        # Calculate efficiency metrics
        if metrics['customer_count'] > 0:
            metrics['avg_time_per_customer'] = metrics['total_time'] / metrics['customer_count']
            metrics['avg_distance_per_customer'] = metrics['total_distance'] / metrics['customer_count']
        else:
            metrics['avg_time_per_customer'] = 0.0
            metrics['avg_distance_per_customer'] = 0.0
        
        # Time utilization (compared to daily limit)
        metrics['time_utilization'] = min(metrics['total_time'] / self.daily_time_limit_minutes, 1.0)
        
        return metrics
    
    def _get_precise_travel_time(self, from_idx, to_idx):
        """
        Get precise travel time between two locations with multiple fallback methods.
        """
        try:
            # Primary method: use calculated time matrix
            if (hasattr(self.distance_calculator, 'time_matrix') and
                from_idx < len(self.distance_calculator.time_matrix) and
                to_idx < len(self.distance_calculator.time_matrix[0])):
                return self.distance_calculator.time_matrix[from_idx][to_idx]
        except:
            pass
        
        try:
            # Secondary method: calculate from distance matrix with average speed
            if (hasattr(self.distance_calculator, 'distance_matrix') and
                from_idx < len(self.distance_calculator.distance_matrix) and
                to_idx < len(self.distance_calculator.distance_matrix[0])):
                distance_km = self.distance_calculator.distance_matrix[from_idx][to_idx]
                avg_speed_kmh = 40.0  # Assume 40 km/h average speed
                return (distance_km / avg_speed_kmh) * 60.0  # Convert to minutes
        except:
            pass
        
        try:
            # Tertiary method: Euclidean distance estimation
            if from_idx < len(self.locations) and to_idx < len(self.locations):
                from_loc = self.locations[from_idx]
                to_loc = self.locations[to_idx]
                dx = to_loc['x'] - from_loc['x']
                dy = to_loc['y'] - from_loc['y']
                euclidean_dist_km = (dx**2 + dy**2)**0.5 * 100  # Convert to km
                return euclidean_dist_km * 1.5  # Assume 40 km/h with 1.5x road factor
        except:
            pass
        
        # Final fallback
        return 60.0  # 1 hour default
    
    def _get_precise_service_time(self, location_idx):
        """
        Get precise service time for a location with fallback.
        """
        try:
            if location_idx < len(self.locations):
                location = self.locations[location_idx]
                return location.get('service_time', 20.0)
        except:
            pass
        
        return 20.0  # Default 20 minutes service time


def test_sequential_multiday():
    """Test the sequential multi-day VRP implementation with large scenario."""
    print("🧪 TESTING SEQUENTIAL MULTI-DAY VRP IMPLEMENTATION (NEW VERSION)")
    print("=" * 60)
    
    # Create large test scenario with realistic customer distribution across Switzerland
    locations = [
        {'id': 'depot', 'x': 47.3769, 'y': 8.5417, 'demand': 0, 'service_time': 0, 'address': 'Main Depot Zurich'},
        
        # North region
        {'id': 'basel_north', 'x': 47.5596, 'y': 7.5886, 'demand': 120, 'volume_demand': 3.2, 'service_time': 25, 'address': 'Basel North'},
        {'id': 'schaffhausen', 'x': 47.6979, 'y': 8.6308, 'demand': 95, 'volume_demand': 2.1, 'service_time': 20, 'address': 'Schaffhausen'},
        {'id': 'winterthur', 'x': 47.5009, 'y': 8.7240, 'demand': 140, 'volume_demand': 3.8, 'service_time': 30, 'address': 'Winterthur'},
        {'id': 'st_gallen', 'x': 47.4245, 'y': 9.3767, 'demand': 180, 'volume_demand': 4.5, 'service_time': 35, 'address': 'St. Gallen'},
        
        # East region  
        {'id': 'chur', 'x': 46.8480, 'y': 9.5330, 'demand': 110, 'volume_demand': 2.8, 'service_time': 25, 'address': 'Chur'},
        {'id': 'davos', 'x': 46.8098, 'y': 9.8368, 'demand': 75, 'volume_demand': 1.9, 'service_time': 20, 'address': 'Davos'},
        {'id': 'st_moritz', 'x': 46.4908, 'y': 9.8355, 'demand': 65, 'volume_demand': 1.7, 'service_time': 18, 'address': 'St. Moritz'},
        
        # South region
        {'id': 'lugano', 'x': 46.0037, 'y': 8.9511, 'demand': 200, 'volume_demand': 5.2, 'service_time': 40, 'address': 'Lugano'},
        {'id': 'locarno', 'x': 46.1712, 'y': 8.7994, 'demand': 85, 'volume_demand': 2.3, 'service_time': 22, 'address': 'Locarno'},
        {'id': 'bellinzona', 'x': 46.1944, 'y': 9.0175, 'demand': 130, 'volume_demand': 3.4, 'service_time': 28, 'address': 'Bellinzona'},
        
        # West region
        {'id': 'geneva', 'x': 46.2044, 'y': 6.1432, 'demand': 250, 'volume_demand': 6.1, 'service_time': 45, 'address': 'Geneva'},
        {'id': 'lausanne', 'x': 46.5197, 'y': 6.6323, 'demand': 190, 'volume_demand': 4.8, 'service_time': 38, 'address': 'Lausanne'},
        {'id': 'montreux', 'x': 46.4312, 'y': 6.9123, 'demand': 100, 'volume_demand': 2.6, 'service_time': 24, 'address': 'Montreux'},
        {'id': 'sion', 'x': 46.2280, 'y': 7.3598, 'demand': 115, 'volume_demand': 3.0, 'service_time': 26, 'address': 'Sion'},
        
        # Central region
        {'id': 'lucerne', 'x': 47.0502, 'y': 8.3093, 'demand': 160, 'volume_demand': 4.1, 'service_time': 32, 'address': 'Lucerne'},
        {'id': 'interlaken', 'x': 46.6863, 'y': 7.8632, 'demand': 90, 'volume_demand': 2.4, 'service_time': 23, 'address': 'Interlaken'},
        {'id': 'bern', 'x': 46.9481, 'y': 7.4474, 'demand': 210, 'volume_demand': 5.4, 'service_time': 42, 'address': 'Bern'},
        {'id': 'fribourg', 'x': 46.8057, 'y': 7.1608, 'demand': 125, 'volume_demand': 3.3, 'service_time': 27, 'address': 'Fribourg'},
    ]
    
    vehicles = [
        {'id': 'truck_40t_alpha', 'capacity': 800, 'volume_capacity': 18.0, 'cost_per_km': 2.20},
        {'id': 'truck_24t_beta', 'capacity': 500, 'volume_capacity': 12.0, 'cost_per_km': 1.80},
        {'id': 'van_7t_gamma', 'capacity': 350, 'volume_capacity': 10.0, 'cost_per_km': 1.20},
        {'id': 'van_4t_delta', 'capacity': 250, 'volume_capacity': 8.0, 'cost_per_km': 0.95},
        {'id': 'sprinter_3t_epsilon', 'capacity': 180, 'volume_capacity': 6.0, 'cost_per_km': 0.75},
    ]
    
    print(f"📍 Large test scenario: {len(locations)} locations, {len(vehicles)} vehicles")
    print(f"📦 Total demand: {sum(loc.get('demand', 0) for loc in locations)} units")
    print(f"📊 Total volume: {sum(loc.get('volume_demand', 0) for loc in locations):.1f} m³")
    print(f"🕒 Total service time: {sum(loc.get('service_time', 0) for loc in locations)} minutes")
    
    # Test the sequential implementation
    sequential_vrp = SequentialMultiDayVRP(vehicles, locations)
    solution = sequential_vrp.solve_sequential_multiday(max_days=7)
    
    if solution:
        # Plot the solution with enhanced visualization
        plot_filename = sequential_vrp.plot_sequential_solution(solution, 
                                                               "Enhanced Sequential Multi-Day VRP (NEW) - Large Switzerland Scenario")
        print(f"\n� Solution plotted and saved as: {plot_filename}")
        
        # Add detailed time analysis
        sequential_vrp.print_detailed_time_analysis(solution)
        
        return solution
    else:
        print("❌ Failed to solve sequential multi-day VRP")
        return None


if __name__ == "__main__":
    test_sequential_multiday()
