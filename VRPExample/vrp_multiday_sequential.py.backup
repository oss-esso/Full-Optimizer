#!/usr/bin/env python3
"""
Sequential Multi-Day VRP Implementation with Cached OSRM Routing
==============================================================

This implements a day-by-day approach to multi-day VRP with realistic OSRM routing:
1. Generate overnight nodes for current day
2. Solve optimization for current day only using cached OSRM routes
3. Keep only overnight locations reached
4. Use overnight locations as starting points for next day
5. Repeat until all days are solved

Features:
- Cached OSRM routing with persistent SQLite database
- Truck speed adjustments based on road composition
- Realistic travel times and distances
- Database grows over time for faster subsequent runs

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
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Import the cached OSRM calculator and truck speed profiles
from route_database import CachedOSRMDistanceCalculator, RouteDatabase
from vrp_scenarios import DEFAULT_TRUCK_SPEED_RATIOS

class SequentialMultiDayVRP:
    """
    Sequential Multi-Day VRP that solves day by day with cached OSRM routing.
    
    Features:
    - Uses cached OSRM routing with persistent SQLite database
    - Applies truck speed adjustments based on road composition
    - Builds comprehensive route database over time
    """
    
    def __init__(self, vehicles, locations, use_truck_speeds=True, db_path="vrp_routes_cache.db"):
        self.vehicles = vehicles
        self.locations = locations
        self.use_truck_speeds = use_truck_speeds
        
        # Store vehicle-specific truck speed ratios
        self.vehicle_speed_ratios = {}
        for vehicle in vehicles:
            vehicle_id = vehicle['id']
            if hasattr(vehicle, 'truck_speed_ratios') or 'truck_speed_ratios' in vehicle:
                speed_ratios = getattr(vehicle, 'truck_speed_ratios', 
                                     vehicle.get('truck_speed_ratios'))
                self.vehicle_speed_ratios[vehicle_id] = speed_ratios
                vehicle_type = vehicle.get('vehicle_type', 'unknown')
                print(f"  🚛 {vehicle_id} ({vehicle_type}): Using specific truck speed ratios")
            else:
                self.vehicle_speed_ratios[vehicle_id] = DEFAULT_TRUCK_SPEED_RATIOS['standard']
                print(f"  🚛 {vehicle_id}: Using default standard truck speed ratios")
        
        # Initialize cached OSRM distance calculator without speed adjustments
        # We'll apply vehicle-specific speed adjustments on-demand
        print(f"🚛 Initializing cached OSRM routing with mixed vehicle types: {use_truck_speeds}")
        self.distance_calculator = CachedOSRMDistanceCalculator(
            locations=self.locations,
            truck_speed_ratios=None,  # Don't apply speed adjustments in matrix
            use_truck_speeds=False,   # We'll handle this per-vehicle
            use_road_composition=True,  # Always use road composition for realism
            db_path=db_path
        )
        
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
    
    def solve_single_day(self, day_num, day_locations, active_vehicles, force_return_to_depot=False):
        """
        Solve VRP for a single day with adaptive overnight logic.
        No capacity constraints, focus on time limits with dynamic overnight stops.
        
        Args:
            day_num: Current day number
            day_locations: Locations available for this day
            active_vehicles: Vehicles available for this day
            force_return_to_depot: If True, all vehicles must end at depot (final day logic)
        """
        print(f"\n🚀 Solving Day {day_num} with {len(active_vehicles)} vehicles...")
        if force_return_to_depot:
            print("  🏠 FINAL DAY: All vehicles must return to depot")
        
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
        all_customer_indices = set(range(len(available_customers)))  # Track all available customers
        assigned_customers = set()  # Track globally assigned customers across all vehicles
        
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
            
            # Visit customers until time limit reached (greedy approach)
            # Each vehicle considers all remaining unvisited customers
            
            # Calculate which customers this vehicle can consider (not yet assigned)
            vehicle_available_customers = all_customer_indices - assigned_customers
            
            if not vehicle_available_customers:
                print(f"    No customers remaining for {vehicle_id}")
                # If started from overnight position, stay there; if from depot, stay at depot
                if current_pos != 0:  # Not at depot
                    print(f"    Staying at current position (no customers to visit)")
                else:
                    print(f"    {current_time:05.1f} - No work, staying at depot")
            else:
                # Greedy assignment: keep taking customers until time/capacity constraints are hit
                while vehicle_available_customers and vehicle_state['remaining_capacity'] > 0:
                    # Find the best customer to visit next (for now, just take the first available)
                    # Convert set to list for indexing, then sort for consistent behavior
                    available_indices = sorted(list(vehicle_available_customers))
                    
                    # Try each remaining customer until we find one we can visit
                    customer_assigned = False
                    for customer_rel_idx in available_indices:
                        customer = available_customers[customer_rel_idx]
                        customer_abs_idx = customer.get('original_customer_idx', customer_rel_idx + 1)
                        
                        # Check capacity constraint first
                        if customer.get('demand', 0) > vehicle_state['remaining_capacity']:
                            print(f"      ❌ {customer['id']} demand ({customer.get('demand', 0)}) exceeds vehicle capacity ({vehicle_state['remaining_capacity']})")
                            continue  # Try next customer
                        
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
                        
                        # First check: Can we reach the customer within the daily limit?
                        if projected_time_after_customer > self.daily_time_limit_minutes:
                            # Can't reach the customer - create overnight stop ON THE ROAD towards this customer
                            print(f"      🚨 CAN'T REACH CUSTOMER! Creating overnight stop ON THE ROAD")
                            
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
                            
                            # Mark that we couldn't assign this customer but found overnight stop
                            customer_assigned = True  # We did something (overnight stop)
                            break  # Exit customer loop for this vehicle
                            
                        else:
                            # We CAN reach the customer - visit them first
                            current_time += total_stop_time
                            route_distance += travel_distance
                            total_demand += customer.get('demand', 0)
                            current_pos = customer_abs_idx
                            
                            # Update vehicle capacity
                            vehicle_state['remaining_capacity'] -= customer.get('demand', 0)
                            
                            # Remove customer from unvisited list
                            # Mark customer as assigned globally 
                            assigned_customers.add(customer_rel_idx)
                            # Update vehicle available customers
                            vehicle_available_customers = all_customer_indices - assigned_customers
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
                            
                            # Mark that we successfully assigned this customer
                            customer_assigned = True
                            
                            # Now check if we can return to depot or need overnight stop on the way back
                            if projected_time_with_return > self.daily_time_limit_minutes:
                                print(f"      🚨 CAN'T RETURN TO DEPOT! Will need overnight stop on return journey")
                                # Stop processing more customers - we need to start return journey
                                break
                            else:
                                print(f"      ✅ Can complete return trip to depot within time limit")
                                # Continue to next customer (if any)
                                break  # Move to next customer in the while loop
                        
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
                        
                        # Set flag to exit main while loop after overnight stop
                        customer_assigned = True  # Mark that we did something (overnight stop)
                        break
                    else:
                        # We CAN reach the customer - visit them first
                        current_time += total_stop_time
                        route_distance += travel_distance
                        total_demand += customer.get('demand', 0)
                        current_pos = customer_abs_idx
                        
                        # Update vehicle capacity
                        vehicle_state['remaining_capacity'] -= customer.get('demand', 0)
                        
                        # Mark customer as assigned globally
                        assigned_customers.add(customer_rel_idx)
                        # Update vehicle available customers
                        vehicle_available_customers = all_customer_indices - assigned_customers
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
                        
                        # Mark that we successfully assigned this customer
                        customer_assigned = True
                        
                        # Now check if we can return to depot or need overnight stop on the way back
                        if projected_time_with_return > self.daily_time_limit_minutes:
                            print(f"      🚨 CAN'T RETURN TO DEPOT! Will need overnight stop on return journey")
                            # Stop processing more customers - we need to start return journey
                            break
                        else:
                            print(f"      ✅ Can complete return trip to depot within time limit")
                            # Continue to next customer (if any)
                            break  # Move to next customer in the while loop
                    
                    # If we couldn't assign any customer, break out of the while loop
                    if not customer_assigned:
                        print(f"      ⚠️ No suitable customers remaining for {vehicle_id}")
                        break
                    
                    # If an overnight stop was created, break out of the main while loop
                    if vehicle_id in overnight_locations:
                        print(f"      🌙 Overnight stop created for {vehicle_id}, ending day")
                        break
            
            # Return to depot if no overnight stop
            if vehicle_id not in overnight_locations:
                try:
                    return_time = self.distance_calculator.time_matrix[current_pos][0] if current_pos < len(self.locations) else 30
                    return_distance = self.distance_calculator.distance_matrix[current_pos][0] if current_pos < len(self.locations) else 20
                except:
                    return_time = 30
                    return_distance = 20
                
                # Check if we can make it back to depot within time limit
                projected_time_with_return = current_time + return_time
                
                # On the final day, we MUST return to depot regardless of time constraints
                if force_return_to_depot:
                    print(f"      🏠 FINAL DAY: Forcing return to depot (time: {projected_time_with_return:.1f} min)")
                    # Force return to depot even if it exceeds time limit
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
                    
                    print(f"    {current_time:05.1f} - Return to depot completed (FINAL DAY)")
                    
                elif projected_time_with_return > self.daily_time_limit_minutes:
                    # Time limit would be exceeded on return trip - create overnight stop ON THE ROAD back to the depot
                    print(f"      🚨 TIME LIMIT EXCEEDED ON RETURN TRIP! Creating overnight stop ON THE WAY BACK")
                    
                    # Calculate how much time we can use to travel towards the depot
                    remaining_time = self.daily_time_limit_minutes - current_time
                    print(f"         Remaining time available: {remaining_time:.1f} min")
                    
                    # Get current position coordinates
                    if current_pos < len(self.locations):
                        current_coords = (self.locations[current_pos]['x'], self.locations[current_pos]['y'])
                    else:
                        current_coords = (start_location['x'], start_location['y'])
                    
                    # Get depot coordinates
                    depot_coords = (self.locations[0]['x'], self.locations[0]['y'])
                        
                    if remaining_time > 30:  # Only move if we have meaningful time left
                        # Calculate how far we can travel towards the depot
                        # Assume average speed for calculating position on route
                        max_travel_time = remaining_time - 30  # Leave 30 min buffer for setup
                        travel_ratio = min(max_travel_time / return_time, 0.8) if return_time > 0 else 0  # Max 80% of the way
                        
                        # Calculate intermediate position on the road to depot
                        dx = depot_coords[0] - current_coords[0]
                        dy = depot_coords[1] - current_coords[1]
                        
                        overnight_x = current_coords[0] + (dx * travel_ratio)
                        overnight_y = current_coords[1] + (dy * travel_ratio)
                        
                        print(f"         Moving {travel_ratio:.1%} of the way towards depot")
                        print(f"         From: {current_coords}")
                        print(f"         To overnight position: ({overnight_x:.1f}, {overnight_y:.1f})")
                        
                        # Update time and distance for partial travel
                        partial_travel_time = max_travel_time
                        partial_travel_distance = return_distance * travel_ratio
                        current_time += partial_travel_time
                        route_distance += partial_travel_distance
                        
                        # Create temporary location for overnight position
                        temp_overnight_idx = len(self.locations)  # Use next available index
                        overnight_location = {
                            'id': f'return_overnight_{vehicle_id}_day{day_num}',
                            'x': overnight_x,
                            'y': overnight_y,
                            'demand': 0,
                            'is_overnight_node': True,
                            'is_road_position': True,
                            'target_customer': 'depot',
                            'original_customer_idx': temp_overnight_idx
                        }
                        
                        # Add this position to locations temporarily for next day
                        self.locations.append(overnight_location)
                        overnight_locations[vehicle_id] = temp_overnight_idx
                        
                        route_stops.append({
                            'location_id': overnight_location['id'],
                            'coordinates': (overnight_location['x'], overnight_location['y']),
                            'is_overnight': True,
                            'demand': 0,
                            'original_customer_idx': overnight_locations[vehicle_id],
                            'arrival_time': current_time
                        })
                        
                        print(f"      🌙 {vehicle_id} stays overnight ON THE WAY BACK at ({overnight_location['x']:.1f}, {overnight_location['y']:.1f})")
                        print(f"         Final driving time for day: {current_time:.1f} min")
                        
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
                        
                        route_stops.append({
                            'location_id': overnight_location['id'],
                            'coordinates': (overnight_location['x'], overnight_location['y']),
                            'is_overnight': True,
                            'demand': 0,
                            'original_customer_idx': overnight_locations[vehicle_id],
                            'arrival_time': current_time
                        })
                        
                        print(f"      🌙 {vehicle_id} stays overnight at current position ({overnight_location['x']:.1f}, {overnight_location['y']:.1f})")
                        print(f"         Final driving time for day: {current_time:.1f} min")
                else:
                    # We can make it back to depot within time limit
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
                    
                    print(f"    {current_time:05.1f} - Return to depot completed successfully")
            
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
    
    def get_vehicle_travel_time(self, from_idx, to_idx, vehicle_id):
        """
        Get travel time between two locations for a specific vehicle type.
        Applies vehicle-specific speed ratios based on road composition.
        
        Args:
            from_idx: Source location index
            to_idx: Destination location index 
            vehicle_id: Vehicle ID to get speed ratios for
            
        Returns:
            Travel time in minutes
        """
        # Get base time from OSRM
        base_time = self.distance_calculator.time_matrix[from_idx][to_idx]
        
        # If no road composition data or not using truck speeds, return base time
        if not hasattr(self.distance_calculator, 'road_composition_matrix') or not self.use_truck_speeds:
            return base_time
        
        # Get vehicle speed ratios
        vehicle = next((v for v in self.vehicles if v['id'] == vehicle_id), None)
        if not vehicle or 'truck_speed_ratios' not in vehicle:
            return base_time
        
        # Apply vehicle-specific speed adjustment
        speed_ratio = self._calculate_vehicle_weighted_ratio(from_idx, to_idx, vehicle_id)
        adjusted_time = base_time / speed_ratio
        
        return adjusted_time
    
    def _calculate_vehicle_weighted_ratio(self, from_idx, to_idx, vehicle_id):
        """
        Calculate weighted speed ratio for a specific vehicle based on road composition.
        
        Args:
            from_idx: Source location index
            to_idx: Destination location index
            vehicle_id: Vehicle ID to get speed ratios for
            
        Returns:
            Weighted speed ratio (lower = slower)
        """
        # Get vehicle speed ratios
        vehicle = next((v for v in self.vehicles if v['id'] == vehicle_id), None)
        if not vehicle or 'truck_speed_ratios' not in vehicle:
            return 1.0
        
        vehicle_ratios = vehicle['truck_speed_ratios']
        
        # Get road composition
        if not hasattr(self.distance_calculator, 'road_composition_matrix'):
            return vehicle_ratios.get('default', 0.8)
        
        try:
            composition = self.distance_calculator.road_composition_matrix[from_idx][to_idx]
            if not composition:
                return vehicle_ratios.get('default', 0.8)
        except (IndexError, KeyError):
            return vehicle_ratios.get('default', 0.8)
        
        # Calculate weighted ratio
        total_weight = 0
        weighted_ratio = 0
        
        for road_type, percentage in composition.items():
            if percentage > 0:
                ratio = vehicle_ratios.get(road_type, vehicle_ratios.get('default', 0.8))
                weighted_ratio += ratio * percentage
                total_weight += percentage
        
        if total_weight > 0:
            return weighted_ratio / total_weight
        else:
            return vehicle_ratios.get('default', 0.8)
    
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
            
            # Check if we still have work to do (customers to visit OR vehicles to return to depot)
            active_vehicles_count = sum(1 for v in self.vehicle_states.values() if v['is_active'])
            
            if not unvisited_customers and active_vehicles_count == 0:
                print(f"  ✅ All customers visited and all vehicles at depot! Complete at day {day_num - 1}")
                break
            elif not unvisited_customers and active_vehicles_count > 0:
                print(f"  🚛 All customers visited, but {active_vehicles_count} vehicles need to return to depot")
            elif unvisited_customers:
                print(f"  📦 {len(unvisited_customers)} customers remaining to visit")
            
            # Get active vehicles for this day
            active_vehicles = []
            for vehicle in self.vehicles:
                if self.vehicle_states[vehicle['id']]['is_active']:
                    active_vehicles.append(vehicle)
            
            if not active_vehicles:
                print(f"  ⚠️ No active vehicles remaining for day {day_num}")
                break

            # --- Check if all unvisited customers can be served in one day (no overnight needed) ---
            if unvisited_customers:
                # Build day_locations with just depot + unvisited customers
                test_day_locations = [self.locations[0]] + [self.locations[idx] for idx in unvisited_customers]
                # Estimate total demand and total time for all customers
                total_demand = sum(self.locations[idx].get('demand', 0) for idx in unvisited_customers)
                total_capacity = sum(self.vehicle_states[v['id']]['remaining_capacity'] for v in active_vehicles)
                
                # Estimate total time needed (more realistic approximation)
                total_time_estimate = 0
                for customer_idx in unvisited_customers:
                    # Time from depot to customer + service time + return to depot
                    depot_to_customer_time = self.distance_calculator.time_matrix[0][customer_idx]
                    customer_to_depot_time = self.distance_calculator.time_matrix[customer_idx][0]
                    service_time = self.locations[customer_idx].get('service_time', 0)
                    # Each customer requires round trip + service time
                    total_time_estimate += depot_to_customer_time + service_time + customer_to_depot_time
                
                # More conservative time feasibility: assume less than perfect parallelization
                # Divide by vehicles but add routing overhead factor
                estimated_time_per_vehicle = total_time_estimate / len(active_vehicles) * 1.3  # 30% routing overhead
                
                # Check if can be solved in one day: capacity OK AND time feasible
                capacity_ok = total_demand <= total_capacity
                time_feasible = estimated_time_per_vehicle <= self.daily_time_limit_minutes * 0.9  # 90% utilization
                
                print(f"  📊 Feasibility check:")
                print(f"    Total demand: {total_demand} / {total_capacity} (capacity)")
                print(f"    Est. time per vehicle: {estimated_time_per_vehicle:.1f} / {self.daily_time_limit_minutes * 0.9:.1f} min")
                print(f"    Capacity OK: {capacity_ok}, Time feasible: {time_feasible}")
                
                # Determine if this is the final day (all customers can be served)
                is_final_day = capacity_ok and time_feasible
            else:
                # No customers left - this is a return-to-depot day
                print(f"  📊 Return-to-depot day (no customers remaining)")
                is_final_day = True
                capacity_ok = True
                time_feasible = True
            
            if is_final_day and unvisited_customers:
                print("  🟢 Final day: All customers can be served. Ensuring all vehicles return to depot.")
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
            elif is_final_day and not unvisited_customers:
                print("  🏠 Return-to-depot day: Bringing all vehicles back to depot.")
                # Create minimal day_locations with just depot for return
                day_locations = []
                depot_copy = copy.deepcopy(self.locations[0])
                depot_copy['original_customer_idx'] = 0
                day_locations.append(depot_copy)
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
            day_solution = self.solve_single_day(day_num, day_locations, active_vehicles, force_return_to_depot=is_final_day)
            
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
            
            # Handle depot bay consumption when delivery destinations are visited
            # When a delivery destination is visited, remove its corresponding depot bay
            depot_bays_to_remove = set()
            for visited_idx in visited_today:
                if visited_idx < len(self.locations):
                    visited_location = self.locations[visited_idx]
                    visited_id = visited_location['id']
                    
                    # Check if this is a delivery destination that has a corresponding depot bay
                    if visited_id == 'cormano_mi':
                        # Find depot_bay_1
                        for idx, loc in enumerate(self.locations):
                            if loc['id'] == 'depot_bay_1':
                                depot_bays_to_remove.add(idx)
                                print(f"    📦 Consumed depot_bay_1 after visiting {visited_id}")
                                break
                    elif visited_id == 'malmo_sweden':
                        # Find depot_bay_2  
                        for idx, loc in enumerate(self.locations):
                            if loc['id'] == 'depot_bay_2':
                                depot_bays_to_remove.add(idx)
                                print(f"    📦 Consumed depot_bay_2 after visiting {visited_id}")
                                break
                    elif visited_id.startswith('depot_bay_'):
                        # If visiting depot bay directly, find corresponding delivery destination
                        bay_number = visited_id.split('_')[-1]
                        if bay_number == '1':
                            # This depot bay is for cormano_mi delivery
                            for idx, loc in enumerate(self.locations):
                                if loc['id'] == 'cormano_mi':
                                    depot_bays_to_remove.add(visited_idx)  # Remove the bay itself
                                    print(f"    📦 Picked up cargo from {visited_id} for cormano_mi delivery")
                                    break
                        elif bay_number == '2':
                            # This depot bay is for malmo_sweden delivery
                            for idx, loc in enumerate(self.locations):
                                if loc['id'] == 'malmo_sweden':
                                    depot_bays_to_remove.add(visited_idx)  # Remove the bay itself
                                    print(f"    📦 Picked up cargo from {visited_id} for malmo_sweden delivery")
                                    break
            
            # Remove consumed depot bays from unvisited customers
            unvisited_customers -= depot_bays_to_remove
            all_visited_customers |= depot_bays_to_remove
            
            # Update vehicle positions for next day
            # Special handling for final day: all vehicles should return to depot
            if is_final_day:
                print(f"  🏠 FINAL DAY: Ensuring all vehicles return to depot")
                for vehicle in active_vehicles:
                    vehicle_id = vehicle['id']
                    self.vehicle_states[vehicle_id]['is_active'] = False
                    self.vehicle_states[vehicle_id]['current_location_idx'] = 0  # Return to depot
                    self.vehicle_states[vehicle_id]['overnight_position'] = None
                    print(f"    🏠 {vehicle_id} returned to depot (final day complete)")
            else:
                # Normal multi-day logic
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
            
            # Check if we need a final day to return vehicles to depot
            vehicles_still_away = sum(1 for v in self.vehicle_states.values() if v['is_active'])
            
            if not unvisited_customers and vehicles_still_away > 0:
                print(f"\n  🚛 Need final day to return {vehicles_still_away} vehicles to depot")
        
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
        """Plot the sequential multi-day solution with clean visualization."""
        if not solution or solution.get('status') != 'SUCCESS':
            print("❌ Cannot plot: No valid solution provided")
            return
        
        print(f"\n🎨 Plotting sequential multi-day solution...")
        
        plt.figure(figsize=(12, 8))
        plt.title(title, fontsize=14, pad=10)
        
        # Colors for different vehicles/days
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        # Line styles for different days
        day_styles = ['-', '--', '-.', ':']
        
        # Plot all locations
        for i, loc in enumerate(self.locations):
            x, y = loc['x'], loc['y']
            if i == 0:  # Depot
                plt.plot(x, y, 's', color='black', markersize=12, 
                        label='Depot' if 'Depot' not in plt.gca().get_legend_handles_labels()[1] else "")
            else:  # Customer
                plt.plot(x, y, 'o', color='gray', markersize=6, alpha=0.6)
            
            # Add location ID as text
            plt.annotate(loc['id'], (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
        
        # Plot vehicle routes by day
        for vehicle_idx, (vehicle_id, vehicle_data) in enumerate(solution['vehicle_routes'].items()):
            full_route = vehicle_data['full_route']
            
            if len(full_route) < 2:
                continue
            
            vehicle_color = colors[vehicle_idx % len(colors)]
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
                
                # Apply day-specific line style
                day_style = day_styles[(day_num - 1) % len(day_styles)]
                
                # Get coordinates
                day_coords = [stop['coordinates'] for stop in day_route]
                
                if len(day_coords) >= 2:
                    x_coords = [coord[0] for coord in day_coords]
                    y_coords = [coord[1] for coord in day_coords]
                    
                    # Plot route with clean styling
                    plt.plot(x_coords, y_coords, linestyle=day_style, color=vehicle_color, 
                            linewidth=2, alpha=0.8, marker='o', markersize=4,
                            label=f'{vehicle_id} Day {day_num}')
                    
                    # Add arrows to show direction
                    for i in range(len(x_coords) - 1):
                        dx = x_coords[i + 1] - x_coords[i]
                        dy = y_coords[i + 1] - y_coords[i]
                        if abs(dx) > 0.01 or abs(dy) > 0.01:
                            plt.annotate('', xy=(x_coords[i+1], y_coords[i+1]), 
                                       xytext=(x_coords[i], y_coords[i]),
                                       arrowprops=dict(arrowstyle='->', color=vehicle_color, alpha=0.6))
                
                # Mark overnight locations
                overnight_coords = [stop['coordinates'] for stop in day_route if stop['is_overnight']]
                for coord in overnight_coords:
                    plt.plot(coord[0], coord[1], '*', color='red', markersize=15, 
                           label='Overnight' if 'Overnight' not in plt.gca().get_legend_handles_labels()[1] else "")
                    plt.annotate(f'Overnight D{day_num}', (coord[0], coord[1]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Customize plot with clean styling
        plt.xlabel('Longitude', fontsize=12)
        plt.ylabel('Latitude', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Clean up legend using automatic legend creation
        # Handle legend duplicates by using the labels already created
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), 
                 bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
                
        # Add simple summary title
        total_distance = sum(route['total_distance'] for route in solution['vehicle_routes'].values())
        total_overnight_stays = sum(route['total_overnight_stays'] for route in solution['vehicle_routes'].values())
        
        # Calculate cost if available
        total_cost = 0
        for vehicle_id, route_data in solution['vehicle_routes'].items():
            vehicle = next((v for v in self.vehicles if v['id'] == vehicle_id), None)
            if vehicle and 'cost_per_km' in vehicle:
                total_cost += route_data['total_distance'] * vehicle['cost_per_km']
        
        cost_text = f", Cost: {total_cost:.2f} CHF" if total_cost > 0 else ""
        
        plt.title(f"{title}\nTotal Distance: {total_distance:.1f} km, Days: {solution['total_days']}, Overnights: {total_overnight_stays}{cost_text}", 
                 fontsize=12)
        
        plt.tight_layout()
        
        # Save plot with clean filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"vrp_solution_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  📊 Plot saved as: {filename}")
        
        plt.axis('equal')  # Ensure map has correct proportions
        plt.show()
        return filename


def test_sequential_multiday():
    """Test the sequential multi-day VRP implementation with large scenario."""
    print("🧪 TESTING SEQUENTIAL MULTI-DAY VRP IMPLEMENTATION")
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
        
        # Additional challenging locations
        {'id': 'zermatt', 'x': 46.0207, 'y': 7.7491, 'demand': 60, 'volume_demand': 1.5, 'service_time': 15, 'address': 'Zermatt'},
        {'id': 'appenzell', 'x': 47.3319, 'y': 9.4108, 'demand': 70, 'volume_demand': 1.8, 'service_time': 18, 'address': 'Appenzell'},
        {'id': 'aarau', 'x': 47.3911, 'y': 8.0431, 'demand': 135, 'volume_demand': 3.5, 'service_time': 29, 'address': 'Aarau'},
    ]
    
    vehicles = [
        {'id': 'truck_40t_alpha', 'capacity': 800, 'volume_capacity': 18.0, 'cost_per_km': 2.20, 'max_daily_km': 600, 'truck_speed_ratios': DEFAULT_TRUCK_SPEED_RATIOS['heavy']},
        {'id': 'truck_24t_beta', 'capacity': 500, 'volume_capacity': 12.0, 'cost_per_km': 1.80, 'max_daily_km': 650, 'truck_speed_ratios': DEFAULT_TRUCK_SPEED_RATIOS['heavy']},
        {'id': 'van_7t_gamma', 'capacity': 350, 'volume_capacity': 10.0, 'cost_per_km': 1.20, 'max_daily_km': 700, 'truck_speed_ratios': DEFAULT_TRUCK_SPEED_RATIOS['standard']},
        {'id': 'van_4t_delta', 'capacity': 250, 'volume_capacity': 8.0, 'cost_per_km': 0.95, 'max_daily_km': 750, 'truck_speed_ratios': DEFAULT_TRUCK_SPEED_RATIOS['standard']},
        {'id': 'sprinter_3t_epsilon', 'capacity': 180, 'volume_capacity': 6.0, 'cost_per_km': 0.75, 'max_daily_km': 800, 'truck_speed_ratios': DEFAULT_TRUCK_SPEED_RATIOS['standard']},
    ]
    
    print(f"📍 Large test scenario: {len(locations)} locations, {len(vehicles)} vehicles")
    print(f"📦 Total demand: {sum(loc.get('demand', 0) for loc in locations)} units")
    print(f"📊 Total volume: {sum(loc.get('volume_demand', 0) for loc in locations):.1f} m³")
    print(f"🕒 Total service time: {sum(loc.get('service_time', 0) for loc in locations)} minutes")
    
    # Test the sequential implementation with cached OSRM routing
    print(f"\n🚛 Initializing Sequential Multi-Day VRP with cached OSRM routing...")
    sequential_vrp = SequentialMultiDayVRP(vehicles, locations, use_truck_speeds=True, db_path="switzerland_routes.db")
    solution = sequential_vrp.solve_sequential_multiday(max_days=7)
    
    if solution:
        # Plot the solution with enhanced visualization
        plot_filename = sequential_vrp.plot_sequential_solution(solution, 
                                                               "Enhanced Sequential Multi-Day VRP - Large Switzerland Scenario")
        print(f"\n� Solution plotted and saved as: {plot_filename}")
        
        return solution
    else:
        print("❌ Failed to solve sequential multi-day VRP")
        return None


if __name__ == "__main__":
    test_sequential_multiday()
