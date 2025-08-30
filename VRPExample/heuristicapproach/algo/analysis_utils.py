"""
Analysis and tracking utilities for the EPDT heuristic solver.

This module contains tracker classes for violations, financial metrics, and order assignments.
"""

class ViolationTracker:
    """Class to track various types of violations across all routes."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all violation counters."""
        # Constraint violations
        self.capability_violations = {
            'loader': 0,
            'low_temp': 0,
            'hangers': 0,
            'total': 0
        }
        
        # Route violations (from HoS validation)
        self.route_violations = {
            'lifo': 0,
            'pallet_constraint': 0,
            'weight_constraint': 0,
            'volume_constraint': 0,
            'time_window': 0,
            'hos_violations': 0,
            'total': 0
        }
        
        # Routes tracking
        self.routes_processed = 0
        self.routes_with_violations = 0
        
        # Store detailed violations for reporting
        self.violation_details = []
    
    def add_capability_violations(self, vehicle_id, violations):
        """Add capability violations for a vehicle."""
        if violations:
            self.routes_with_violations += 1
            for violation in violations:
                if 'LOADER' in violation:
                    self.capability_violations['loader'] += 1
                elif 'LOW_TEMP' in violation:
                    self.capability_violations['low_temp'] += 1
                elif 'HANGERS' in violation:
                    self.capability_violations['hangers'] += 1
                
                self.capability_violations['total'] += 1
                self.violation_details.append({
                    'vehicle_id': vehicle_id,
                    'type': 'capability',
                    'violation': violation
                })
    
    def add_route_violation(self, vehicle_id, violation_msg):
        """Add route violation from HoS validation messages."""
        if 'LIFO' in violation_msg.upper():
            self.route_violations['lifo'] += 1
        elif 'PALLET' in violation_msg.upper():
            self.route_violations['pallet_constraint'] += 1
        elif 'WEIGHT' in violation_msg.upper():
            self.route_violations['weight_constraint'] += 1
        elif 'VOLUME' in violation_msg.upper():
            self.route_violations['volume_constraint'] += 1
        elif 'TIME WINDOW' in violation_msg.upper():
            self.route_violations['time_window'] += 1
        elif 'HOS' in violation_msg.upper():
            self.route_violations['hos_violations'] += 1
        
        self.route_violations['total'] += 1
        self.violation_details.append({
            'vehicle_id': vehicle_id,
            'type': 'route',
            'violation': violation_msg
        })
    
    def increment_routes_processed(self):
        """Increment the counter of processed routes."""
        self.routes_processed += 1
    
    def get_summary(self):
        """Get a summary of all violations."""
        total_violations = self.capability_violations['total'] + self.route_violations['total']
        
        return {
            'routes_processed': self.routes_processed,
            'routes_with_violations': self.routes_with_violations,
            'routes_clean': self.routes_processed - self.routes_with_violations,
            'total_violations': total_violations,
            'capability_violations': self.capability_violations,
            'route_violations': self.route_violations,
            'violation_details': self.violation_details
        }


class ProfitTracker:
    """Class to track profit and financial metrics across all routes."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all profit tracking."""
        self.total_cost = 0.0
        self.total_revenue = 0.0
        self.total_profit = 0.0
        
        self.driver_costs = 0.0
        self.vehicle_costs = 0.0
        
        self.profitable_routes = 0
        self.unprofitable_routes = 0
        self.routes_processed = 0
        
        # Store individual route profits for analysis
        self.route_profits = []
        
        # Track best and worst performing routes
        self.best_route = None
        self.worst_route = None
        self.best_profit = float('-inf')
        self.worst_profit = float('inf')
    
    def add_route_profit(self, vehicle_id, breakdown):
        """Add profit data from a route breakdown."""
        profit = breakdown.get('net_profit', 0.0)
        cost = breakdown.get('total_cost', 0.0)
        revenue = breakdown.get('total_profit', 0.0)  # total_profit is actually revenue
        driver_cost = breakdown.get('driver_cost', 0.0)
        vehicle_cost = breakdown.get('vehicle_cost', 0.0)
        
        # Update totals
        self.total_cost += cost
        self.total_revenue += revenue
        self.total_profit += profit
        self.driver_costs += driver_cost
        self.vehicle_costs += vehicle_cost
        
        # Track profitable vs unprofitable
        if profit >= 0:
            self.profitable_routes += 1
        else:
            self.unprofitable_routes += 1
        
        self.routes_processed += 1
        
        # Store route profit data
        route_data = {
            'vehicle_id': vehicle_id,
            'profit': profit,
            'cost': cost,
            'revenue': revenue,
            'margin': (profit / revenue * 100) if revenue > 0 else 0
        }
        self.route_profits.append(route_data)
        
        # Track best and worst routes
        if profit > self.best_profit:
            self.best_profit = profit
            self.best_route = route_data
        
        if profit < self.worst_profit:
            self.worst_profit = profit
            self.worst_route = route_data
    
    def get_summary(self):
        """Get financial summary."""
        overall_margin = (self.total_profit / self.total_revenue * 100) if self.total_revenue > 0 else 0
        avg_profit_per_route = self.total_profit / self.routes_processed if self.routes_processed > 0 else 0
        
        return {
            'routes_processed': self.routes_processed,
            'total_cost': self.total_cost,
            'total_revenue': self.total_revenue,
            'total_profit': self.total_profit,
            'overall_margin': overall_margin,
            'driver_costs': self.driver_costs,
            'vehicle_costs': self.vehicle_costs,
            'profitable_routes': self.profitable_routes,
            'unprofitable_routes': self.unprofitable_routes,
            'avg_profit_per_route': avg_profit_per_route,
            'best_route': self.best_route,
            'worst_route': self.worst_route
        }


class OrderTracker:
    """Comprehensive order tracking system for debugging assignment pipeline."""
    
    def __init__(self, orders_to_track=None):
        self.orders_to_track = orders_to_track or []
        self.assignment_log = {}
        self.phase_assignments = {
            'initialization': {},
            'optimization': {},
            'force_assignment': {}
        }
        
    def log_assignment(self, order_id, vehicle_id, phase, details=""):
        """Log when an order gets assigned to a vehicle."""
        if order_id not in self.assignment_log:
            self.assignment_log[order_id] = []
        
        entry = {
            'phase': phase,
            'vehicle_id': vehicle_id,
            'details': details,
            'timestamp': len(self.assignment_log[order_id])
        }
        
        self.assignment_log[order_id].append(entry)
        self.phase_assignments[phase][order_id] = vehicle_id
        
        if order_id in self.orders_to_track:
            print(f"     TRACKING Order {order_id}: ASSIGNED to {vehicle_id} (by {phase}) - {details}")
    
    def log_attempt(self, order_id, vehicle_id, phase, reason=""):
        """Log when an assignment is attempted but fails."""
        if order_id in self.orders_to_track:
            print(f"     TRACKING Order {order_id}: FAILED attempt on {vehicle_id} (in {phase}) - {reason}")
    
    def check_assignment_status(self, solution, orders):
        """Check current assignment status and log changes."""
        print(f"\nORDER TRACKING: ASSIGNMENT STATUS CHECK")
        for order in orders:
            if order.id in self.orders_to_track:
                assigned_vehicle = None
                for vehicle_id, route in solution.routes.items():
                    if route and route.tasks:
                        for task in route.tasks:
                            if hasattr(task, 'order_id') and task.order_id == order.id:
                                assigned_vehicle = vehicle_id
                                break
                        if assigned_vehicle:
                            break
                
                if assigned_vehicle:
                    print(f"     TRACKING Order {order.id}: ASSIGNED to {assigned_vehicle}")
                    if order.id not in self.assignment_log:
                        self.log_assignment(order.id, assigned_vehicle, "unknown", "Found in solution")
                else:
                    print(f"     TRACKING Order {order.id}: UNASSIGNED")
    
    def print_summary(self):
        """Print final tracking summary."""
        print(f"\n" + "="*80)
        print(f"ORDER TRACKING SUMMARY")
        print(f"="*80)
        
        for phase in ['initialization', 'optimization', 'force_assignment']:
            assigned_in_phase = len(self.phase_assignments[phase])
            print(f"\n{phase.upper()}:")
            if assigned_in_phase > 0:
                for order_id, vehicle_id in self.phase_assignments[phase].items():
                    details = ""
                    for entry in self.assignment_log.get(order_id, []):
                        if entry['phase'] == phase:
                            details = entry['details']
                            break
                    print(f"   Order {order_id} -> {vehicle_id} ({details})")
            else:
                print(f"   No orders assigned in this phase")


def analyze_vehicle_utilization_detailed(solution, vehicles, orders=None):
    """Analyze vehicle utilization and provide detailed capacity analysis."""
    if not solution or not vehicles:
        print("ERROR: Missing solution or vehicles for utilization analysis")
        return
    
    active_vehicles = []
    idle_vehicles = []
    underutilized_vehicles = []
    
    print("\nVehicles with Significant Remaining Capacity (>1000kg):")
    print("Vehicle ID   Peak Load (Max During Route)  Remaining Capacity        Utilization          Status")
    print("-" * 100)
    
    for vehicle in vehicles:
        vehicle_id = vehicle.id
        route = solution.routes.get(vehicle_id)
        
        if route and hasattr(route, 'tasks') and route.tasks and len(route.tasks) > 2:
            # Import utilization function
            try:
                from algo.route_utils import calculate_peak_route_utilization
            except ImportError:
                # Fallback simple utilization calculation
                utilization = {
                    'peak_weight': 0, 'peak_volume': 0, 'peak_pallets': 0,
                    'utilization_weight': 0.0, 'utilization_volume': 0.0, 'utilization_pallets': 0.0
                }
            
            utilization = calculate_peak_route_utilization(route)
            
            weight_capacity = getattr(vehicle, 'weight_capacity', 1) or 1
            remaining_weight = weight_capacity - utilization['peak_weight']
            
            # Check for significant remaining capacity
            if remaining_weight > 1000:
                underutilized_vehicles.append({
                    'vehicle': vehicle,
                    'utilization': utilization,
                    'remaining_weight': remaining_weight
                })
                
                print(f"{vehicle_id:<11} Weight: {utilization['peak_weight']:.0f}kg/{weight_capacity}kg    "
                      f"Unused: {remaining_weight:.0f}kg          "
                      f"Util: {utilization['utilization_weight']:.1f}%           "
                      f"UNDERUTILIZED")
            
            active_vehicles.append(vehicle)
        else:
            idle_vehicles.append(vehicle)
            weight_capacity = getattr(vehicle, 'weight_capacity', 1) or 1
            print(f"{vehicle_id:<11} Weight: 0kg/{weight_capacity}kg           "
                  f"Unused: {weight_capacity}kg          "
                  f"Util: 0.0%              "
                  f"IDLE")
    
    print(f"\nVehicles with >1000kg unused capacity: {len(underutilized_vehicles)}")
    print(f"Completely idle vehicles: {len(idle_vehicles)}")
    
    # Summary statistics
    total_vehicles = len(vehicles)
    active_count = len(active_vehicles)
    idle_count = len(idle_vehicles)
    underutilized_count = len(underutilized_vehicles)
    
    print(f"\nUTILIZATION SUMMARY:")
    print(f"   • Total Vehicles: {total_vehicles}")
    print(f"   • Active Vehicles: {active_count} ({(active_count/total_vehicles*100):.1f}%)")
    print(f"   • Idle Vehicles: {idle_count} ({(idle_count/total_vehicles*100):.1f}%)")
    print(f"   • Underutilized Vehicles (>1000kg free): {underutilized_count} ({(underutilized_count/total_vehicles*100):.1f}%)")
