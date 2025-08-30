"""
Reporting and output utilities for the EPDT heuristic solver.

This module contains functions for printing reports, summaries, and detailed breakdowns.
"""

def print_route_cost_breakdown(vehicle_id: str, route, vehicle=None, orders=None, 
                              violation_tracker=None, profit_tracker=None):
    """Print detailed cost and profit breakdown for a route."""
    # Import required functions
    try:
        from algo.financial_utils import calculate_route_cost_and_profit
        from algo.vehicle_utils import get_vehicle_capabilities, get_route_order_requirements, validate_constraints
    except ImportError:
        print(f"Warning: Could not import required utilities for cost breakdown")
        return
    
    breakdown = calculate_route_cost_and_profit(vehicle_id, route, vehicle, orders)
    
    print(f"\n       COST & PROFIT BREAKDOWN:")
    print(f"      ============================")
    print(f"       Operations:")
    print(f"         • Total Hours: {breakdown['total_hours']:.1f}h")
    print(f"         • Total Distance: {breakdown['total_distance']:.1f}km")
    print(f"         • Vehicle Type: {'Camion (Heavy)' if breakdown['price_per_km'] == 1.25 else 'Furgone (Standard)'}")
    
    # Add constraint validation
    vehicle_caps = get_vehicle_capabilities(vehicle)
    order_reqs = get_route_order_requirements(orders, route)
    violations = validate_constraints(vehicle_caps, order_reqs)
    
    print(f"       Vehicle Capabilities:")
    print(f"         • Loader: {'YES' if vehicle_caps['loader'] else 'NO'}")
    print(f"         • Low Temp: {'YES' if vehicle_caps['low_temp'] else 'NO'}")
    print(f"         • Hangers: {'YES' if vehicle_caps['hangers'] else 'NO'}")
    print(f"         • LIFO Required: {'YES' if vehicle_caps['lifo_required'] else 'NO'}")
    if vehicle_caps['regulations']:
        print(f"         • Regulations: {', '.join(vehicle_caps['regulations'])}")
    
    print(f"       Order Requirements:")
    print(f"         • Needs Loader: {'YES' if order_reqs['loader'] else 'NO'}")
    print(f"         • Needs Low Temp: {'YES' if order_reqs['low_temp'] else 'NO'}")
    print(f"         • Needs Hangers: {'YES' if order_reqs['hangers'] else 'NO'}")
    if order_reqs['special_requirements']:
        print(f"         • Special: {', '.join(order_reqs['special_requirements'])}")
    
    print(f"       Constraint Validation:")
    if violations:
        print(f"         X VIOLATIONS: {', '.join(violations)}")
        # Track capability violations if tracker is provided
        if violation_tracker:
            violation_tracker.add_capability_violations(vehicle_id, violations)
    else:
        print(f"         OK All constraints satisfied")
    
    # Track route processing and profit if trackers are provided
    if violation_tracker:
        violation_tracker.increment_routes_processed()
    if profit_tracker:
        profit_tracker.add_route_profit(vehicle_id, breakdown)
    
    print(f"       Costs:")
    print(f"         • Driver Cost: €{breakdown['driver_cost']:.2f} ({breakdown['driver_cost_per_hour']:.1f}€/h × {breakdown['driving_hours']:.1f}h driving)")
    print(f"         • Vehicle Cost: €{breakdown['vehicle_cost']:.2f} ({breakdown['vehicle_cost_per_km']:.2f}€/km × {breakdown['total_distance']:.1f}km)")
    print(f"         • Total Cost: €{breakdown['total_cost']:.2f}")
    
    print(f"       Revenue:")
    print(f"         • Rate: {breakdown['price_per_km']:.2f}€/km")
    print(f"         • Total Revenue: €{breakdown['total_profit']:.2f} (sum of dedicated route revenues)")
    print(f"         • Revenue Model: Each order valued at dedicated route distance × rate")
    
    print(f"       Net Result:")
    net_color = "Y" if breakdown['net_profit'] >= 0 else "N"
    print(f"         {net_color} Net Profit: €{breakdown['net_profit']:.2f}")
    
    if breakdown['total_profit'] > 0:
        margin = (breakdown['net_profit'] / breakdown['total_profit']) * 100
        print(f"          Profit Margin: {margin:.1f}%")


def print_comprehensive_final_report(solution=None, orders=None, vehicles=None,
                                   violation_tracker=None, profit_tracker=None):
    """Print the final comprehensive report with violations and financial summary."""
    print("\n" + "="*80)
    print("COMPREHENSIVE FINAL REPORT")
    print("="*80)
    
    if not solution or not orders or not vehicles:
        print("ERROR: Missing required data for final report")
        return
    
    # Get tracker summaries
    violation_summary = violation_tracker.get_summary() if violation_tracker else {}
    profit_summary = profit_tracker.get_summary() if profit_tracker else {}
    
    # Print violation summary
    print(f"\nVIOLATION SUMMARY:")
    if violation_summary:
        print(f"   Routes Processed: {violation_summary.get('routes_processed', 0)}")
        print(f"   Routes Clean: {violation_summary.get('routes_clean', 0)}")
        print(f"   Routes with Violations: {violation_summary.get('routes_with_violations', 0)}")
        print(f"   Total Violations: {violation_summary.get('total_violations', 0)}")
        
        cap_violations = violation_summary.get('capability_violations', {})
        if cap_violations.get('total', 0) > 0:
            print(f"\n   Capability Violations:")
            print(f"      • Loader Missing: {cap_violations.get('loader', 0)}")
            print(f"      • Low Temp Missing: {cap_violations.get('low_temp', 0)}")
            print(f"      • Hangers Missing: {cap_violations.get('hangers', 0)}")
        
        route_violations = violation_summary.get('route_violations', {})
        if route_violations.get('total', 0) > 0:
            print(f"\n   Route Violations:")
            print(f"      • LIFO Issues: {route_violations.get('lifo', 0)}")
            print(f"      • Pallet Constraints: {route_violations.get('pallet_constraint', 0)}")
            print(f"      • Weight Constraints: {route_violations.get('weight_constraint', 0)}")
            print(f"      • Volume Constraints: {route_violations.get('volume_constraint', 0)}")
            print(f"      • Time Window Issues: {route_violations.get('time_window', 0)}")
            print(f"      • HoS Violations: {route_violations.get('hos_violations', 0)}")
    else:
        print("   No violation data available")
    
    # Print financial summary
    print(f"\nFINANCIAL SUMMARY:")
    if profit_summary:
        print(f"   Routes Analyzed: {profit_summary.get('routes_processed', 0)}")
        print(f"   Total Cost: €{profit_summary.get('total_cost', 0):.2f}")
        print(f"      • Driver Costs: €{profit_summary.get('driver_costs', 0):.2f}")
        print(f"      • Vehicle Costs: €{profit_summary.get('vehicle_costs', 0):.2f}")
        print(f"   Total Revenue: €{profit_summary.get('total_revenue', 0):.2f}")
        print(f"   Net Profit: €{profit_summary.get('total_profit', 0):.2f}")
        print(f"   Overall Margin: {profit_summary.get('overall_margin', 0):.1f}%")
        print(f"   Avg Profit/Route: €{profit_summary.get('avg_profit_per_route', 0):.2f}")
        
        profitable = profit_summary.get('profitable_routes', 0)
        unprofitable = profit_summary.get('unprofitable_routes', 0)
        total_routes = profitable + unprofitable
        if total_routes > 0:
            print(f"   Profitable Routes: {profitable}/{total_routes} ({(profitable/total_routes*100):.1f}%)")
        
        best_route = profit_summary.get('best_route')
        worst_route = profit_summary.get('worst_route')
        if best_route:
            print(f"   Best Route: {best_route['vehicle_id']} (€{best_route['profit']:.2f})")
        if worst_route:
            print(f"   Worst Route: {worst_route['vehicle_id']} (€{worst_route['profit']:.2f})")
    else:
        print("   No financial data available")
    
    # Get solution statistics
    try:
        total_orders = len(orders)
        assigned_orders = 0
        unassigned_orders = []
        
        # Count assigned orders
        for vehicle_id, route in solution.routes.items():
            if route and route.tasks:
                for task in route.tasks:
                    if hasattr(task, 'order_id') and task.order_id:
                        assigned_orders += 1
        
        # Find unassigned orders
        for order in orders:
            assigned = False
            for vehicle_id, route in solution.routes.items():
                if route and route.tasks:
                    for task in route.tasks:
                        if hasattr(task, 'order_id') and task.order_id == order.id:
                            assigned = True
                            break
                    if assigned:
                        break
            if not assigned:
                unassigned_orders.append(order.id)
        
        print(f"\nSOLUTION SUMMARY:")
        print(f"   Total Orders: {total_orders}")
        print(f"   Assigned Tasks: {assigned_orders}")
        print(f"   Unassigned Orders: {len(unassigned_orders)}")
        if unassigned_orders:
            print(f"   Unassigned Order IDs: {unassigned_orders[:10]}{'...' if len(unassigned_orders) > 10 else ''}")
        
        # Vehicle utilization
        active_vehicles = sum(1 for route in solution.routes.values() if route and route.tasks and len(route.tasks) > 2)
        total_vehicles = len(vehicles)
        print(f"   Active Vehicles: {active_vehicles}/{total_vehicles} ({(active_vehicles/total_vehicles*100):.1f}%)")
        
    except Exception as e:
        print(f"   Could not calculate solution statistics: {e}")
    
    print("\n" + "="*80)


def flush_print(*args, **kwargs):
    """Helper function to print and flush output immediately for redirection compatibility."""
    import sys
    print(*args, **kwargs)
    sys.stdout.flush()


def export_routes_to_text(solution, file_path: str):
    """Export route solution to text file for external analysis."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("ROUTE EXPORT\n")
            f.write("=" * 50 + "\n\n")
            
            for vehicle_id, route in solution.routes.items():
                if route and route.tasks and len(route.tasks) > 2:
                    f.write(f"Vehicle: {vehicle_id}\n")
                    f.write("-" * 20 + "\n")
                    
                    for i, task in enumerate(route.tasks):
                        task_type = "DEPOT" if hasattr(task, 'is_depot_start') and task.is_depot_start() else \
                                   "DEPOT" if hasattr(task, 'is_depot_return') and task.is_depot_return() else \
                                   "PICKUP" if hasattr(task, 'is_pickup') and task.is_pickup() else \
                                   "DELIVERY" if hasattr(task, 'is_delivery') and task.is_delivery() else "UNKNOWN"
                        
                        order_id = getattr(task, 'order_id', 'N/A')
                        address = getattr(task, 'address', 'Unknown Address')
                        
                        f.write(f"  {i+1:2d}. {task_type:<8} Order:{order_id:<4} {address}\n")
                    
                    f.write("\n")
            
        print(f"Routes exported to: {file_path}")
        
    except Exception as e:
        print(f"Error exporting routes: {e}")
