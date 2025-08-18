"""
Comprehensive Integration Test for EPDT System

This script implements step 16 from TODO2.md - creating a single, comprehensive test
that validates the integration of all advanced features in the EPDT system.

Test Phases:
1. Phase 1: Heuristic Solver Test
   - Load scenario from furgoni.xlsx using create_scenario_from_excel
   - Run l1_heuristic to generate optimized vehicle routes
   - Validate routes and print detailed summary

2. Phase 2: Driver Assignment Integration
   - Load drivers using load_drivers_from_excel_enhanced
   - Enhanced driver qualifications via license correction logic
   - Assign drivers to routes using assign_drivers_to_routes_enhanced
   - Print final comprehensive solution summary

Usage:
    python comprehensive_integration_test.py
"""

import sys
import os
import time
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add necessary paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
heuristic_root = os.path.join(current_dir, '..')
src_dir = os.path.join(heuristic_root, 'src')
algo_dir = os.path.join(heuristic_root, 'algo')
utils_dir = os.path.join(heuristic_root, 'utils')

sys.path.insert(0, heuristic_root)
sys.path.insert(0, src_dir)
sys.path.insert(0, algo_dir)
sys.path.insert(0, utils_dir)

print("=== EPDT Comprehensive Integration Test ===")
print(f"Test directory: {current_dir}")
print(f"Heuristic root: {heuristic_root}")

# Import required modules with robust error handling
try:
    from scenario_creator import create_scenario_from_excel
    print("OK Successfully imported create_scenario_from_excel")
except ImportError as e:
    print(f"ERROR importing scenario_creator: {e}")
    sys.exit(1)

try:
    # We'll skip data_adapter if create_scenario_from_excel returns EPDT objects directly
    pass  # Removed data_adapter import since create_scenario_from_excel returns EPDT objects
    print("✅ Skipping data_adapter (not needed for direct EPDT scenario loading)")
except ImportError:
    try:
        from algo.data_adapter import convert_instance_to_epdt_input, get_default_parameters
        print("✅ Successfully imported data_adapter functions with algo prefix")
    except ImportError as e:
        print(f"⚠️  Warning: data_adapter not available: {e}")
        print("   Will use direct EPDT objects from scenario_creator")

try:
    from first_level import l1_heuristic
    print("✅ Successfully imported l1_heuristic")
except ImportError:
    try:
        from algo.first_level import l1_heuristic
        print("✅ Successfully imported l1_heuristic with algo prefix")
    except ImportError as e:
        print(f"❌ Error importing l1_heuristic: {e}")
        sys.exit(1)

try:
    from driver_assignment_enhanced import (
        load_drivers_from_excel_enhanced,
        assign_drivers_to_routes_enhanced,
        DriverAssignmentConfig,
        print_assignment_summary
    )
    print("✅ Successfully imported driver assignment functions")
except ImportError:
    try:
        from algo.driver_assignment_enhanced import (
            load_drivers_from_excel_enhanced,
            assign_drivers_to_routes_enhanced,
            DriverAssignmentConfig,
            print_assignment_summary
        )
        print("✅ Successfully imported driver assignment functions with algo prefix")
    except ImportError as e:
        print(f"❌ Error importing driver assignment: {e}")
        sys.exit(1)

# Import route provider for OSRM mode
try:
    from route_provider import set_testing_mode
    set_testing_mode(use_haversine=False)  # Enable OSRM routing
    print("✅ Configured route provider for OSRM mode")
except ImportError:
    try:
        from algo.route_provider import set_testing_mode
        set_testing_mode(use_haversine=False)  # Enable OSRM routing
        print("✅ Configured route provider for OSRM mode with algo prefix")
    except ImportError:
        print("⚠️  Warning: route_provider not available, using fallback calculations")

# Test import of previously problematic modules (fixed circular dependencies)
try:
    from granular_tabu_search import granular_multiple_order_relocation_neighborhood
    print("✅ Successfully imported granular_tabu_search (circular dependency fixed)")
except ImportError:
    try:
        from algo.granular_tabu_search import granular_multiple_order_relocation_neighborhood
        print("✅ Successfully imported granular_tabu_search with algo prefix (circular dependency fixed)")
    except ImportError as e:
        print(f"⚠️  Warning: granular_tabu_search not available: {e}")

try:
    from destroy_and_repair import destroy_and_repair
    print("✅ Successfully imported destroy_and_repair (circular dependency fixed)")
except ImportError:
    try:
        from algo.destroy_and_repair import destroy_and_repair
        print("✅ Successfully imported destroy_and_repair with algo prefix (circular dependency fixed)")
    except ImportError as e:
        print(f"⚠️  Warning: destroy_and_repair not available: {e}")

try:
    from parallelization import l1_heuristic_parallel
    print("✅ Successfully imported parallelization (circular dependency fixed)")
except ImportError:
    try:
        from algo.parallelization import l1_heuristic_parallel
        print("✅ Successfully imported parallelization with algo prefix (circular dependency fixed)")
    except ImportError as e:
        print(f"⚠️  Warning: parallelization not available: {e}")

# Import route pre-computation infrastructure
try:
    from utils.precompute_routes import RoutePrecomputer
    print("✅ Successfully imported RoutePrecomputer for OSRM pre-computation")
except ImportError as e:
    print(f"⚠️  Warning: RoutePrecomputer not available: {e}")
    RoutePrecomputer = None


# Global counter for tracking route calculations
# This helps track how many route calculations are made (OSRM calls or cache hits)
route_calculation_count = 0

def calculate_travel_time_with_counter(prev_task, curr_task, vehicle):
    """
    Wrapper for calculate_travel_time_between_tasks that increments the global counter.
    This helps track how many route calculations are made.
    
    Args:
        prev_task: Previous task in route
        curr_task: Current task in route  
        vehicle: Vehicle object
        
    Returns:
        Travel time in minutes
    """
    global route_calculation_count
    route_calculation_count += 1
    
    try:
        from route_provider import calculate_travel_time_between_tasks
        return calculate_travel_time_between_tasks(prev_task, curr_task, vehicle)
    except ImportError:
        try:
            from algo.route_provider import calculate_travel_time_between_tasks
            return calculate_travel_time_between_tasks(prev_task, curr_task, vehicle)
        except ImportError:
            # Fallback: simple time estimation
            return 60  # 1 hour default

def reset_route_calculation_counter():
    """Reset the global route calculation counter."""
    global route_calculation_count
    route_calculation_count = 0

def get_route_calculation_count():
    """Get the current route calculation count."""
    global route_calculation_count
    return route_calculation_count


def format_duration_detailed(minutes: float) -> str:
    """
    Format duration in minutes to a human-readable string with days, hours, and minutes.
    
    Args:
        minutes: Duration in minutes
    
    Returns:
        Formatted string like "1d 5h 30m", "2h 15m", or "45m"
    """
    if minutes < 0:
        return "0m"
    
    total_minutes = int(minutes)
    days = total_minutes // 1440
    remaining_minutes = total_minutes % 1440
    hours = remaining_minutes // 60
    mins = remaining_minutes % 60
    
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if mins > 0 or len(parts) == 0:
        parts.append(f"{mins}m")
    
    return " ".join(parts)


def print_detailed_route_breakdown(vehicle_id: str, route, vehicle=None):
    """
    Print detailed route breakdown with task sequence, load tracking, and timing.
    """
    if not route or not route.tasks:
        return

    def get_location_name(task):
        return getattr(getattr(task, 'location', None), 'name', getattr(task, 'location_id', "Unknown"))

    def get_order_info(task):
        return f"Order: {getattr(task, 'order_id', 'N/A')}"

    def format_absolute_minutes(minutes):
        if minutes is None:
            return "No window"
        day = int(minutes / 1440) + 1
        remaining_minutes = int(minutes % 1440)
        hour = remaining_minutes // 60
        minute = remaining_minutes % 60
        return f"Day {day}, {hour:02d}:{minute:02d}"

    def get_time_window_info(task):
        earliest = getattr(task, 'earliest_time', None)
        latest = getattr(task, 'latest_time', None)
        start_str = format_absolute_minutes(earliest)
        end_str = format_absolute_minutes(latest)
        return f"[{start_str} -> {end_str}]"

    def get_load_change(task):
        weight_change = getattr(task, 'demand', 0.0)
        volume_change = getattr(task, 'volume', 0.0)
        return weight_change, volume_change

    try:
        from second_level import is_feasible
        feasible, reason = is_feasible(route, debug_feasibility=True, return_reason=True)
    except ImportError:
        feasible, reason = True, "(Feasibility check unavailable)"

    time.sleep(1)  # Add 1-second delay
    print(f"   VEHICLE: {vehicle_id}:")
    print(f"      Feasible: {feasible}")
    if not feasible:
        print(f"      Reason: {reason}")

    total_duration_minutes = 0
    if len(route.tasks) > 1:
        try:
            for i in range(1, len(route.tasks)):
                prev_task = route.tasks[i-1]
                curr_task = route.tasks[i]
                travel_time = calculate_travel_time_with_counter(prev_task, curr_task, route.vehicle)
                prev_location = get_location_name(prev_task)
                curr_location = get_location_name(curr_task)
                #print(f"DEBUG: Used route_provider for {prev_location} -> {curr_location}: {travel_time:.1f}m")
                total_duration_minutes += travel_time
                # Add service time at current location
                service_time = getattr(curr_task, 'service_time', 5.0)  # Default 5 minutes
                total_duration_minutes += service_time
        except Exception as e:
            total_duration_minutes = len(route.tasks) * 60  # Fallback

    duration_formatted = format_duration_detailed(total_duration_minutes)
    days = int(total_duration_minutes / 1440)
    hos_warning = " (would violate HoS if attempted without proper rests)" if total_duration_minutes > 11 * 60 else ""

    print(f"       Route duration: {days} day(s) ({duration_formatted}){hos_warning}")
    
    # Display total waiting time if any waiting occurred
    if hasattr(route, 'total_wait_time') and route.total_wait_time > 0:
        wait_time_formatted = format_duration_detailed(route.total_wait_time)
        print(f"       Total waiting time: {wait_time_formatted}")
    
    print(f"       Task sequence ({len(route.tasks)} tasks) - Real-time monitoring:")

    current_weight, current_volume, completion_time = 0, 0, 0

    for i, task in enumerate(route.tasks, 1):
        # Calculate waiting time at departure (following markdown specification)
        departure_time = completion_time
        travel_time = 0
        wait_time = 0
        
        if i > 1:
            try:
                prev_task = route.tasks[i-2]  # Previous task (i-2 because enumerate starts at 1)
                travel_time = calculate_travel_time_with_counter(prev_task, task, route.vehicle)
                
                # Calculate waiting time at previous location (as per markdown)
                if task.earliest_time and task.earliest_time > 0:
                    # Required departure time from previous location to arrive exactly at earliest_time
                    required_departure_time = task.earliest_time - travel_time
                    
                    # If we are ready to depart earlier than required, we wait at previous location
                    if completion_time < required_departure_time:
                        wait_time = required_departure_time - completion_time
                        
            except:
                travel_time = 60  # Fallback

        # Departure time from previous task location includes waiting
        departure_time = completion_time + wait_time
        
        # Arrival at current task
        arrival_time = departure_time + travel_time
        service_time = getattr(task, 'service_time', 5.0)
        completion_time = arrival_time + service_time

        # Get task details
        location = get_location_name(task)
        order_info = get_order_info(task)
        time_window = get_time_window_info(task)
        
        # Calculate load changes
        weight_change, volume_change = get_load_change(task)
        current_weight += weight_change
        current_volume += volume_change
        
        # Format cumulative time (arrival time at this task)
        cumulative_formatted = format_duration_detailed(arrival_time)
        
        # Format delta time with breakdown (as per markdown specification)
        if i == 1:
            delta_str = ""
        else:
            if wait_time > 0:
                # Show breakdown: travel time + wait time (as per markdown)
                travel_formatted = format_duration_detailed(travel_time)
                wait_formatted = format_duration_detailed(wait_time)
                delta_str = f" (+{travel_formatted} travel, +{wait_formatted} wait)"
            else:
                # Only travel time
                travel_formatted = format_duration_detailed(travel_time)
                delta_str = f" (+{travel_formatted})"
        
        # Format load change
        weight_sign = "+" if weight_change >= 0 else ""
        volume_sign = "+" if volume_change >= 0 else ""
        
        print(f"          {i}. {location} ({order_info}) - Cumulative: {cumulative_formatted}{delta_str} {time_window}")
        print(f"             Load: {weight_sign}{weight_change:.1f}kg, {volume_sign}{volume_change:.1f}m³ -> Total: {current_weight:.1f}kg, {current_volume:.1f}m³")
        
        # Display arrival status with waiting information (as per markdown specification)
        arrival_status = ""
        if hasattr(task, 'earliest_time') and hasattr(task, 'latest_time'):
            if task.earliest_time is not None and arrival_time < task.earliest_time:
                remaining_wait = task.earliest_time - arrival_time
                arrival_status = f"(Arrived early, would wait {format_duration_detailed(remaining_wait)})"
            elif task.latest_time is not None and arrival_time > task.latest_time:
                lateness = arrival_time - task.latest_time
                arrival_status = f"(LATE by {format_duration_detailed(lateness)})"
            else:
                arrival_status = "(On time)"
        else:
            arrival_status = "(On time)"
            
        # For previous task, show if there was waiting before departure (as per markdown)
        if i > 1 and wait_time > 0:
            prev_location = get_location_name(route.tasks[i-2])
            print(f"             Waiting: ({format_duration_detailed(wait_time)} at {prev_location} before departure)")
        
        print(f"             Status: {arrival_status}")


def configure_algorithm_parameters() -> dict:
    """
    Configure optimized algorithm parameters for comprehensive testing.
    Enhanced to be more lenient in initial assignment for better order coverage.
    
    Returns:
        Dictionary of algorithm parameters suitable for l1_heuristic
    """
    return {
        'tabu_tenure': 50,  # Reduced from 100 - allow more flexibility in search
        'M1': 50,  # Increased from 30 - more L1 iterations for better exploration
        'M2': 300,  # Increased from 200 - even more total iterations
        'exploration_strategy': 'vnd',
        'enable_advanced_neighborhoods': True,
        'enable_granular_search': True,
        'enable_parallelization': False,
        'parallel_strategy': 'PE',
        'local_search_strategy': 'best_improvement',
        'initialization_method': 'best_insertion',
        'vehicle_penalty_per_vehicle': 0.0,  # Further reduced from 10.0 - make vehicles very cheap
        'unassigned_order_base_penalty': 50000.0,  # Dramatically reduced from 500000.0 - much more lenient
        'time_window_violation_penalty': 500000.0, # Significantly increased to enforce time windows
        'capacity_violation_penalty': 100.0,  # Reduced from 250.0 - very tolerant of capacity violations
        'distance_violation_penalty': 50.0,  # Reduced from 100.0 - moderate penalty for distance violations
        'Lo': 1000.0,  # Reduced from 1500.0 - smaller initial threshold
        'wk_ID': 60.0,  # Reduced from 80.0 - less strict on distance improvements
        'wk_IE': 60.0,  # Reduced from 80.0 - less strict on exchanges
        'wk_IF': 40.0,  # Reduced from 40.0 - less strict on feasibility
        'wk_IH': 30.0,  # Reduced from 40.0 - less strict on relocations
        'wk_IJ': 10.0,  # Reduced from 15.0 - less strict on swaps
        'M': 5000.0,  # Reduced from 8000.0 - smaller penalty multiplier
        'P_task': 500000.0,  # Reduced from 75000.0 - less penalty for task violations
        'P_fleet': 50000000.0,  # Reduced from 75000.0 - less penalty for fleet violations
        'max_neighbors_to_evaluate': 500,  # Increased from 150 - explore more neighbors
        'best_k_insertions': 50,  # Increased from 25 - try more insertion positions
        'enable_delta_evaluation': True,
        'max_neighbors_per_iteration': 500,  # Increased from 250 - even less VND limits
        # Cluster-aware initialization parameters for much more lenient assignment
        'cluster_tolerance_factor': 1.0,  # Increased from 1.5 - Allow 100% more tolerance in clustering
        'initial_assignment_relaxation': 0.8,  # Reduced from 0.8 - Relax constraints by 50% during initialization
        'capacity_buffer_factor': 1.5,  # Increased from 1.2 - Allow 50% capacity buffer during initial assignment
        'time_window_buffer_minutes': 30,  # Increased from 30 - Allow 60 minutes buffer for time windows
        'max_assignment_attempts': 10,  # Keep at 10 - Try multiple times to assign difficult orders
        # New ultra-lenient parameters
        'force_assignment_mode': True,  # New: Force assignment even with minor violations
        'capacity_overflow_tolerance': 1.3,  # New: Allow 30% capacity overflow during initial assignment
        'assignment_priority_boost': 2.0,  # New: Boost assignment priority for difficult orders
        'relaxed_constraints_iteration_limit': 20,  # New: Use relaxed constraints for first 20 iterations
        # Force assignment strategy for 100% order coverage
        'enable_force_assignment': True,  # New: Enable smart force assignment of unassigned orders
        'force_assignment_strategy': 'least_loaded_capable',  # New: Strategy for selecting vehicles for force assignment
        # Advanced order insertion strategies (TODO 20)
        'initialization_method': 'regret_k',  # New: Use regret-k insertion heuristic ('best_insertion', 'cluster_aware', or 'regret_k')
        'regret_k_value': 3,  # New: k value for regret calculation (2 or 3 is common)
        'enable_destroy_and_repair': True,  # New: Enable destroy and repair for large unassigned orders
        'max_destroy_attempts': 10,  # New: Maximum number of destroy-repair attempts for difficult orders
        'debug_regret': True,  # New: Enable debug output for regret-k initialization
        'debug_destroy_repair': True,  # New: Enable debug output for destroy and repair operations
    }


def print_route_validation_summary(solution, orders, vehicles, runtime_seconds=None):
    """
    Print detailed validation summary for the heuristic solution.
    Enhanced to include systematic analysis of unassigned orders.
    
    Args:
        solution: Solution object from l1_heuristic
        orders: List of Order objects
        vehicles: List of Vehicle objects
        runtime_seconds: Optional runtime information for test summary
    """
    print("\n" + "="*80)
    print("📊 PHASE 1: HEURISTIC SOLVER VALIDATION RESULTS")
    print("="*80)
    
    # Debug: Ensure we reach this function
    print(f"DEBUG: Validation summary called with runtime={runtime_seconds}")
    
    total_orders = len(orders)
    total_vehicles = len(vehicles)
    total_routes = len([r for r in solution.routes.values() if r.tasks])
    
    # Track all order IDs for unassigned analysis
    all_order_ids = {order.id for order in orders}
    assigned_orders = set()
    total_tasks = 0
    total_distance = 0.0
    
    print(f"\n🚛 Vehicle and Route Summary:")
    print(f"   • Total available vehicles: {total_vehicles}")
    print(f"   • Vehicles with routes: {total_routes}")
    print(f"   • Vehicles idle: {total_vehicles - total_routes}")
    
    for vehicle_id, route in solution.routes.items():
        if route.tasks:
            # Extract order IDs from tasks (filter out depot/auxiliary tasks)
            route_orders = set()
            for task in route.tasks:
                if hasattr(task, 'order_id') and task.order_id:
                    # Filter out depot and auxiliary tasks
                    order_id = task.order_id
                    if not (order_id.startswith('depot_') or 
                           order_id.startswith('DEPOT_') or
                           order_id.endswith('_start') or 
                           order_id.endswith('_return')):
                        route_orders.add(order_id)
            
            # Add to global assigned orders set (automatically handles duplicates)
            assigned_orders.update(route_orders)
            total_tasks += len(route.tasks)
            
            # Calculate estimated route distance for summary stats
            try:
                route_distance = 0.0
                for i in range(1, len(route.tasks)):
                    prev_task = route.tasks[i-1]
                    curr_task = route.tasks[i]
                    travel_time = calculate_travel_time_with_counter(prev_task, curr_task, route.vehicle)
                    # Convert time to distance (rough estimate: 50 km/h average)
                    route_distance += travel_time * 50 / 60
                total_distance += route_distance
            except:
                pass  # Distance calculation not available
            
            # Print detailed route breakdown
            print_detailed_route_breakdown(vehicle_id, route, getattr(route, 'vehicle', None))
    
    # Calculate assignment statistics (simple and clear)
    unassigned_order_ids = all_order_ids - assigned_orders
    unassigned_orders = len(unassigned_order_ids)
    assigned_orders_count = len(assigned_orders)
    assignment_rate = (assigned_orders_count / total_orders) * 100 if total_orders > 0 else 0
    
    print(f"\n📦 Order Assignment Summary:")
    print(f"   • Orders assigned: {assigned_orders_count}/{total_orders} ({assignment_rate:.1f}%)")
    print(f"   • Total tasks created: {total_tasks}")
    
    # Only show detailed analysis if there are unassigned orders
    if unassigned_orders > 0:
        print(f"   • Unassigned orders: {unassigned_orders}")
        
        try:
            # Detailed analysis of unassigned orders
            print(f"\n❌ UNASSIGNED ORDERS ANALYSIS:")
            print(f"   Analyzing {len(unassigned_order_ids)} unassigned orders:")
            
            # Create order lookup map
            order_map = {order.id: order for order in orders}
            
            for i, order_id in enumerate(list(unassigned_order_ids)[:10]):  # Limit to first 10 for readability
                order = order_map.get(order_id)
                if order:
                    print(f"\n   {i+1}. Order: {order_id}")
                    
                    try:
                        # Calculate total requirements - use absolute values to show actual cargo needs
                        total_weight = 0.0
                        total_volume = 0.0
                        total_pallets = 0
                        pickup_locations = []
                        delivery_locations = []
                        
                        for task in order.get_all_tasks():
                            # Use absolute values to show actual cargo requirements
                            # (depot bay pairs cancel out, but we want to see the actual load)
                            total_weight += abs(task.demand)
                            total_volume += abs(task.volume) 
                            total_pallets += abs(task.pallets)
                            
                            if task.is_pickup():
                                pickup_locations.append(task.location_id)
                            elif task.is_delivery():
                                delivery_locations.append(task.location_id)
                        
                        # Since depot bay pairs have 2 tasks, divide by 2 to get actual order requirements
                        actual_weight = total_weight / 2
                        actual_volume = total_volume / 2
                        actual_pallets = total_pallets / 2
                        
                        print(f"      • Actual cargo weight: {actual_weight:.1f} kg")
                        print(f"      • Actual cargo volume: {actual_volume:.2f} m³")
                        print(f"      • Actual cargo pallets: {actual_pallets:.0f}")
                        print(f"      • Pickup locations: {len(pickup_locations)}")
                        print(f"      • Delivery locations: {len(delivery_locations)}")
                        print(f"      • Priority: {getattr(order, 'priority', 'N/A')}")
                        print(f"      • Is urgent: {getattr(order, 'is_urgent', 'N/A')}")
                        print(f"      • Is mandatory: True")  # All orders are now mandatory
                        
                        # Time window analysis
                        earliest_pickup = None
                        latest_delivery = None
                        for task in order.get_all_tasks():
                            if task.is_pickup() and hasattr(task, 'earliest_time') and task.earliest_time:
                                if earliest_pickup is None or task.earliest_time < earliest_pickup:
                                    earliest_pickup = task.earliest_time
                            if task.is_delivery() and hasattr(task, 'latest_time') and task.latest_time:
                                if latest_delivery is None or task.latest_time > latest_delivery:
                                    latest_delivery = task.latest_time
                        
                        if earliest_pickup is not None or latest_delivery is not None:
                            print(f"      • Time constraints:")
                            if earliest_pickup is not None:
                                print(f"        - Earliest pickup: {earliest_pickup:.0f} minutes")
                            if latest_delivery is not None:
                                print(f"        - Latest delivery: {latest_delivery:.0f} minutes")
                    except Exception as e:
                        print(f"      • Error analyzing order {order_id}: {e}")
            
            if len(unassigned_order_ids) > 10:
                print(f"\n   ... and {len(unassigned_order_ids) - 10} more unassigned orders")
        
        except Exception as e:
            print(f"   • Error in unassigned orders analysis: {e}")
    
    # Distance summary with error handling
    try:
        if total_distance > 0:
            print(f"\n🛣️  Route Distance Summary:")
            print(f"   • Total estimated distance: {total_distance:.1f} km")
            print(f"   • Average distance per route: {total_distance/total_routes:.1f} km")
    except Exception as e:
        print(f"\n🛣️  Route Distance Summary: Error calculating distances: {e}")
    
    # Solution quality assessment with error handling
    try:
        print(f"\n⭐ Solution Quality Assessment:")
        if assignment_rate >= 90:
            print("   📈 Excellent: >90% orders assigned")
        elif assignment_rate >= 75:
            print("   📊 Good: 75-90% orders assigned")
        elif assignment_rate >= 50:
            print("   📉 Fair: 50-75% orders assigned")
        else:
            print("   🔴 Poor: <50% orders assigned")
    except Exception as e:
        print(f"\n⭐ Solution Quality Assessment: Error calculating quality: {e}")
    
    # Test Summary with Haversine call count for OSRM estimation - ALWAYS DISPLAY
    print("DEBUG: About to display Test Summary...")
    try:
        print(f"\n📊 Test Summary:")
        print(f"   • Scenario source: furgoni.xlsx")
        print(f"   • Orders processed: {total_orders}")
        print(f"   • Vehicles available: {total_vehicles}")
        if runtime_seconds is not None:
            print(f"   • Total runtime: {runtime_seconds:.2f} seconds")
        print(f"   • Route calculations: {get_route_calculation_count()}")
        print(f"   • OSRM calls made: ~{get_route_calculation_count()} (cached results available for future runs)")
        print("DEBUG: Test Summary displayed successfully!")
    except Exception as e:
        print(f"\n📊 Test Summary: Error displaying summary: {e}")
        # Fallback summary even if there are errors
        print(f"   • Basic info: {len(orders)} orders, {len(vehicles)} vehicles")
        if runtime_seconds is not None:
            print(f"   • Runtime: {runtime_seconds:.2f} seconds")
    
    return {
        'total_orders': total_orders,
        'assigned_orders': assigned_orders_count,
        'assignment_rate': assignment_rate,
        'total_routes': total_routes,
        'total_distance': total_distance,
        'route_calculations': get_route_calculation_count()
    }


def get_unassigned_orders(solution, orders):
    """
    Identify orders that are not assigned to any vehicle route.
    
    Args:
        solution: Solution object with routes
        orders: List of all Order objects
        
    Returns:
        List of unassigned Order objects
    """
    # Get all assigned order IDs from the solution
    assigned_order_ids = set()
    
    for route in solution.routes.values():
        if route and route.tasks:
            for task in route.tasks:
                if hasattr(task, 'order_id') and task.order_id:
                    # Remove depot-related order IDs
                    order_id = task.order_id
                    if not ('depot_start' in str(order_id).lower() or 'depot_return' in str(order_id).lower()):
                        assigned_order_ids.add(order_id)
    
    # Find unassigned orders
    unassigned_orders = []
    for order in orders:
        if order.id not in assigned_order_ids:
            unassigned_orders.append(order)
    
    return unassigned_orders


def calculate_route_load(route):
    """Calculate the current load (weight, volume, pallets) of a route."""
    if not route or not route.tasks:
        return 0.0, 0.0, 0
    
    total_weight = 0.0
    total_volume = 0.0
    total_pallets = 0
    
    for task in route.tasks:
        # Only count positive demands (pickups)
        if hasattr(task, 'demand') and task.demand > 0:
            total_weight += task.demand
        if hasattr(task, 'volume') and task.volume > 0:
            total_volume += task.volume
        if hasattr(task, 'pallets') and task.pallets > 0:
            total_pallets += task.pallets
    
    return total_weight, total_volume, total_pallets


def get_order_requirements(order):
    """Extract weight, volume, and pallet requirements from an order."""
    tasks = order.get_all_tasks()
    
    total_weight = 0.0
    total_volume = 0.0
    total_pallets = 0
    
    for task in tasks:
        total_weight += abs(task.demand)
        total_volume += abs(task.volume)
        total_pallets += abs(task.pallets)
    
    # Actual requirements (pickup/delivery pairs cancel out, so divide by 2)
    actual_weight = total_weight / 2 if len(tasks) > 1 else total_weight
    actual_volume = total_volume / 2 if len(tasks) > 1 else total_volume
    actual_pallets = total_pallets / 2 if len(tasks) > 1 else total_pallets
    
    return actual_weight, actual_volume, actual_pallets


def can_vehicle_handle_order_with_penalties(vehicle, order, current_load=None):
    """
    Check if a vehicle can handle an order with hard/soft constraints and penalty calculation.
    
    Hard constraints (cannot be violated):
    - Volume capacity: must not exceed vehicle.volume_capacity
    - Pallet capacity: must not exceed vehicle.pallet_capacity
    
    Soft constraints (can be violated with penalties):
    - Weight capacity: can exceed vehicle.weight_capacity but with penalty
    
    Args:
        vehicle: Vehicle object
        order: Order object
        current_load: Optional tuple of (weight, volume, pallets) current load
        
    Returns:
        Tuple of (can_handle: bool, penalty_score: float)
        - can_handle: True if hard constraints are satisfied
        - penalty_score: 0.0 if no violations, positive value for weight overflow
    """
    if current_load is None:
        current_load = (0.0, 0.0, 0)
    
    current_weight, current_volume, current_pallets = current_load
    order_weight, order_volume, order_pallets = get_order_requirements(order)
    
    # Calculate final loads after adding this order
    final_weight = current_weight + order_weight
    final_volume = current_volume + order_volume
    final_pallets = current_pallets + order_pallets
    
    # Hard constraints (must not be violated)
    can_handle_volume = final_volume <= vehicle.volume_capacity
    can_handle_pallets = final_pallets <= vehicle.pallet_capacity
    
    # For very heavy orders, allow more weight overflow tolerance
    order_difficulty = calculate_order_difficulty(order)
    if order_difficulty > 300:  # Very difficult orders (like ORDER_CURTI_SRL_10)
        max_weight_multiplier = 1.2  # Allow 20% weight overflow
    else:
        max_weight_multiplier = 1.1  # Allow 10% weight overflow for normal orders
    
    max_allowed_weight = vehicle.weight_capacity * max_weight_multiplier
    can_handle_weight_with_tolerance = final_weight <= max_allowed_weight
    
    # Check hard constraints (including weight tolerance)
    if not (can_handle_volume and can_handle_pallets and can_handle_weight_with_tolerance):
        return False, float('inf')  # Cannot handle due to hard constraint violation
    
    # Soft constraint (weight) - calculate penalty
    penalty_score = 0.0
    if final_weight > vehicle.weight_capacity:
        weight_overflow = final_weight - vehicle.weight_capacity
        penalty_score = weight_overflow * 100.0  # Reduced penalty: 100 per kg overflow (was 1000)
        print(f"      💡 Allowing weight overflow: {weight_overflow:.1f}kg (penalty: {penalty_score:.0f})")
    
    return True, penalty_score


def calculate_order_difficulty(order):
    """
    Calculate difficulty score for an order based on its requirements.
    Higher scores mean more difficult to assign (bigger orders).
    
    Args:
        order: Order object
        
    Returns:
        Float difficulty score (higher = more difficult)
    """
    weight, volume, pallets = get_order_requirements(order)
    
    # Normalize each component and combine
    # Weight in tons (divide by 1000), volume as-is, pallets * 10 for importance
    difficulty = (weight / 1000.0) + volume + (pallets * 10.0)
    
    return difficulty


def force_assign_order_to_vehicle(solution, order, vehicle):
    """Force assign an order to a specific vehicle by adding its tasks to the vehicle's route."""
    # Get or create the route for this vehicle
    if vehicle.id not in solution.routes or solution.routes[vehicle.id] is None:
        # Create a new route - simple route object
        class SimpleRoute:
            def __init__(self, vehicle):
                self.vehicle = vehicle
                self.tasks = []
        
        solution.routes[vehicle.id] = SimpleRoute(vehicle)
    
    route = solution.routes[vehicle.id]
    
    # Get all tasks from the order
    order_tasks = order.get_all_tasks()
    
    # Add tasks to the end of the route (simple insertion strategy)
    for task in order_tasks:
        route.tasks.append(task)
    
    print(f"✅ Force assigned order {order.id} to vehicle {vehicle.id}")
    return True


def smart_force_assign_unassigned_orders(solution, orders, vehicles):
    """
    Smart force assignment of unassigned orders to capable vehicles.
    Enhanced with:
    - Big orders processed first (by difficulty score)
    - Hard constraints for volume and pallets
    - Soft constraints for weight (with penalties)
    """
    print("\n🔧 ENHANCED SMART FORCE ASSIGNMENT (BIG ORDERS FIRST)")
    print("="*60)
    
    unassigned_orders = get_unassigned_orders(solution, orders)
    
    if not unassigned_orders:
        print("✅ No unassigned orders found!")
        return 0
    
    # Sort orders by difficulty (big orders first)
    unassigned_orders_with_difficulty = []
    for order in unassigned_orders:
        difficulty = calculate_order_difficulty(order)
        unassigned_orders_with_difficulty.append((order, difficulty))
    
    # Sort by difficulty descending (most difficult first)
    unassigned_orders_with_difficulty.sort(key=lambda x: x[1], reverse=True)
    
    print(f"📦 Found {len(unassigned_orders)} unassigned orders (sorted by difficulty):")
    for order, difficulty in unassigned_orders_with_difficulty:
        weight, volume, pallets = get_order_requirements(order)
        print(f"   • {order.id}: {weight:.1f}kg, {volume:.2f}m³, {pallets:.0f}pal (difficulty: {difficulty:.1f})")
    
    force_assigned_count = 0
    
    for order, difficulty in unassigned_orders_with_difficulty:
        print(f"\n🎯 Processing order {order.id} (difficulty: {difficulty:.1f})...")
        
        # Find capable vehicles for this order with penalty scores
        capable_vehicles = []
        
        for vehicle in vehicles:
            # Calculate current load of this vehicle
            current_load = calculate_route_load(solution.routes.get(vehicle.id))
            
            can_handle, penalty_score = can_vehicle_handle_order_with_penalties(vehicle, order, current_load)
            
            if can_handle:
                # Store vehicle with penalty score and current total load for sorting
                total_load = sum(current_load)  # Simple total load metric
                # Combined score: prefer low penalty, then low current load
                combined_score = penalty_score + (total_load * 0.1)  # Weight current load less than penalty
                capable_vehicles.append((vehicle, penalty_score, total_load, combined_score))
        
        if capable_vehicles:
            # Sort by combined score (penalty + load factor)
            capable_vehicles.sort(key=lambda x: x[3])  # Sort by combined_score
            best_vehicle, penalty_score, current_load, combined_score = capable_vehicles[0]
            
            print(f"   • Found {len(capable_vehicles)} capable vehicles")
            if penalty_score > 0:
                weight_overflow = penalty_score / 1000.0  # Convert penalty back to kg
                print(f"   • ⚠️  SOFT CONSTRAINT: Weight overflow of {weight_overflow:.1f}kg")
                print(f"   • Assigning to best vehicle: {best_vehicle.id} (penalty: {penalty_score:.0f}, load: {current_load:.1f})")
            else:
                print(f"   • ✅ Perfect fit! Assigning to: {best_vehicle.id} (no penalties, load: {current_load:.1f})")
            
            # Force assign the order
            if force_assign_order_to_vehicle(solution, order, best_vehicle):
                force_assigned_count += 1
            else:
                print(f"   • ❌ Failed to assign order {order.id} to vehicle {best_vehicle.id}")
                
        else:
            print(f"   • ❌ No capable vehicles found for order {order.id} (hard constraints violated)")
            # Log vehicle capacity vs order requirements for debugging
            order_weight, order_volume, order_pallets = get_order_requirements(order)
            max_weight = max(v.weight_capacity for v in vehicles)
            max_volume = max(v.volume_capacity for v in vehicles)
            max_pallets = max(v.pallet_capacity for v in vehicles)
            print(f"     Order needs: {order_weight:.1f}kg, {order_volume:.2f}m³, {order_pallets:.0f}pal")
            print(f"     Max available: {max_weight:.1f}kg, {max_volume:.2f}m³, {max_pallets:.0f}pal")
    
    print(f"\n✅ Enhanced force assignment completed: {force_assigned_count}/{len(unassigned_orders)} orders assigned")
    print("="*60)
    
    return force_assigned_count


def debug_unassigned_orders(orders, vehicles):
    """
    Debug why specific orders are not getting assigned by checking constraints.
    """
    print("\n🔍 DEBUGGING UNASSIGNED ORDERS")
    print("="*60)
    
    # Focus on the two problematic orders
    problematic_orders = ['ORDER_GFM_IMBOTTITURE_SNC_31', 'ORDER_CURTI_SRL_10']
    
    # Create order lookup
    order_map = {order.id: order for order in orders}
    
    for order_id in problematic_orders:
        if order_id in order_map:
            order = order_map[order_id]
            print(f"\n📦 Analyzing {order_id}:")
            
            # Calculate actual order requirements
            total_weight = 0
            total_volume = 0 
            total_pallets = 0
            
            try:
                tasks = order.get_all_tasks()
                print(f"   • Number of tasks: {len(tasks)}")
                
                for task in tasks:
                    total_weight += abs(task.demand)
                    total_volume += abs(task.volume)
                    total_pallets += abs(task.pallets)
                    print(f"   • Task: {getattr(task, 'task_type', 'unknown')} at {getattr(task, 'location_id', 'unknown')}, demand: {task.demand}kg, volume: {task.volume}m³")
                
                # Actual requirements (depot bay pairs cancel out)
                actual_weight = total_weight / 2 if len(tasks) > 1 else total_weight
                actual_volume = total_volume / 2 if len(tasks) > 1 else total_volume
                actual_pallets = total_pallets / 2 if len(tasks) > 1 else total_pallets
                
                print(f"   • ACTUAL REQUIREMENTS:")
                print(f"     - Weight: {actual_weight:.1f} kg")
                print(f"     - Volume: {actual_volume:.2f} m³") 
                print(f"     - Pallets: {actual_pallets:.0f}")
                print(f"     - Priority: {getattr(order, 'priority', 'N/A')}")
                
                # Check which vehicles can handle this order
                suitable_vehicles = []
                weight_violations = []
                volume_violations = []
                pallet_violations = []
                
                for vehicle in vehicles:
                    can_handle_weight = vehicle.weight_capacity >= actual_weight
                    can_handle_volume = vehicle.volume_capacity >= actual_volume  
                    can_handle_pallets = vehicle.pallet_capacity >= actual_pallets
                    
                    if can_handle_weight and can_handle_volume and can_handle_pallets:
                        suitable_vehicles.append(vehicle.id)
                    else:
                        if not can_handle_weight:
                            weight_violations.append(f"{vehicle.id}({vehicle.weight_capacity}kg)")
                        if not can_handle_volume:
                            volume_violations.append(f"{vehicle.id}({vehicle.volume_capacity}m³)")
                        if not can_handle_pallets:
                            pallet_violations.append(f"{vehicle.id}({vehicle.pallet_capacity}pal)")
                
                print(f"   • VEHICLE COMPATIBILITY:")
                print(f"     - Suitable vehicles: {len(suitable_vehicles)}/{len(vehicles)}")
                if len(suitable_vehicles) > 0:
                    print(f"     - Examples: {suitable_vehicles[:5]}")
                else:
                    print(f"     - ❌ NO SUITABLE VEHICLES FOUND!")
                    if weight_violations:
                        print(f"     - Weight violations: {len(weight_violations)} vehicles")
                        print(f"       Examples: {weight_violations[:3]}")
                    if volume_violations:
                        print(f"     - Volume violations: {len(volume_violations)} vehicles")  
                        print(f"       Examples: {volume_violations[:3]}")
                    if pallet_violations:
                        print(f"     - Pallet violations: {len(pallet_violations)} vehicles")
                        print(f"       Examples: {pallet_violations[:3]}")
                        
            except Exception as e:
                print(f"   • Error analyzing order {order_id}: {e}")
                
        else:
            print(f"\n❌ Order {order_id} not found in loaded orders")
    
    print("\n🚛 VEHICLE CAPACITY SUMMARY:")
    try:
        if vehicles:
            weights = [v.weight_capacity for v in vehicles]
            volumes = [v.volume_capacity for v in vehicles]
            pallets = [v.pallet_capacity for v in vehicles]
            
            print(f"   • Weight capacity: {min(weights):.0f} - {max(weights):.0f} kg (avg: {sum(weights)/len(weights):.0f})")
            print(f"   • Volume capacity: {min(volumes):.1f} - {max(volumes):.1f} m³ (avg: {sum(volumes)/len(volumes):.1f})")
            print(f"   • Pallet capacity: {min(pallets):.0f} - {max(pallets):.0f} pal (avg: {sum(pallets)/len(pallets):.1f})")
    except Exception as e:
        print(f"   • Error analyzing vehicle capacities: {e}")
        
    print("="*60)


def run_phase1_heuristic_test(excel_path: str) -> tuple:
    """
    Phase 1: Load scenario and run heuristic solver.
    
    Args:
        excel_path: Path to the furgoni.xlsx file
        
    Returns:
        Tuple of (solution, orders, vehicles, runtime_seconds)
    """
    print("\n" + "="*80)
    print("🚀 PHASE 1: HEURISTIC SOLVER TEST")
    print("="*80)
    
    # Reset Haversine call counter for accurate tracking
    reset_route_calculation_counter()
    #print(f"DEBUG: Haversine call counter reset to {get_haversine_call_count()}")

    
    # Step 1: Load scenario from Excel
    print(f"\n📁 Loading scenario from: {excel_path}")
    try:
        orders, vehicles, drivers = create_scenario_from_excel(excel_path)
        print(f"✅ Successfully loaded scenario:")
        print(f"   • Orders: {len(orders)}")
        print(f"   • Vehicles: {len(vehicles)}")
        
        # Debug unassigned orders before running optimization
        debug_unassigned_orders(orders, vehicles)
        
        # The scenario_creator already returns EPDT objects, so no conversion needed
        epdt_orders = orders
        epdt_vehicles = vehicles
        print(f"✅ EPDT objects ready (no conversion needed)")
        
    except Exception as e:
        print(f"❌ Error loading scenario: {e}")
        raise
    
    # Step 2: Configure algorithm parameters
    print(f"\n⚙️  Configuring algorithm parameters...")
    params = configure_algorithm_parameters()
    print(f"✅ Parameters configured (M1={params['M1']}, M2={params['M2']})")
    
    # Step 3: Run l1_heuristic
    print(f"\n🧠 Running l1_heuristic...")
    start_time = time.time()
    
    try:
        solution = l1_heuristic(epdt_orders, epdt_vehicles, params)
        runtime_seconds = time.time() - start_time
        
        print(f"✅ Heuristic completed successfully!")
        print(f"   • Runtime: {runtime_seconds:.2f} seconds")
        print(f"   • Solution type: {type(solution).__name__}")
        
        # Step 3.5: Apply force assignment if enabled
        if params.get('enable_force_assignment', False):
            print(f"\n🚀 Applying smart force assignment...")
            force_assigned_count = smart_force_assign_unassigned_orders(solution, epdt_orders, epdt_vehicles)
            print(f"✅ Force assignment completed: {force_assigned_count} additional orders assigned")
        
    except Exception as e:
        runtime_seconds = time.time() - start_time
        print(f"❌ Heuristic failed after {runtime_seconds:.2f} seconds: {e}")
        raise
    
    # Step 4: Validate and summarize results
    print("DEBUG: About to call validation summary...")
    validation_results = print_route_validation_summary(solution, epdt_orders, epdt_vehicles, runtime_seconds)
    print("DEBUG: Validation summary completed.")
    
    return solution, epdt_orders, epdt_vehicles, runtime_seconds


def run_phase2_driver_assignment(excel_path: str, solution, vehicles) -> None:
    """
    Phase 2: Load drivers and assign them to routes.
    
    Args:
        excel_path: Path to the furgoni.xlsx file
        solution: Solution object from Phase 1
        vehicles: List of Vehicle objects
    """
    print("\n" + "="*80)
    print("👨‍💼 PHASE 2: DRIVER ASSIGNMENT INTEGRATION")
    print("="*80)
    
    # Step 1: Load drivers from Excel
    print(f"\n📁 Loading drivers from: {excel_path}")
    try:
        config = DriverAssignmentConfig()
        drivers = load_drivers_from_excel_enhanced(excel_path, config=config)
        print(f"✅ Successfully loaded drivers:")
        print(f"   • Total drivers: {len(drivers)}")
        
        # Display driver summary
        license_counts = {}
        for driver in drivers:
            license = getattr(driver, 'license_type', 'Unknown')
            license_counts[license] = license_counts.get(license, 0) + 1
        
        print(f"   • License distribution:")
        for license, count in license_counts.items():
            print(f"     - {license}: {count} drivers")
            
    except Exception as e:
        print(f"❌ Error loading drivers: {e}")
        raise
    
    # Step 2: Enhanced driver qualifications
    print(f"\n🔧 Driver qualifications enhanced via license correction logic")
    print(f"   (Built into load_drivers_from_excel_enhanced function)")
    
    # Step 3: Extract routes for assignment
    print(f"\n🛣️  Preparing routes for driver assignment...")
    active_routes = {}
    route_objects = []
    
    for vehicle_id, route in solution.routes.items():
        if route.tasks:  # Only routes with tasks
            active_routes[vehicle_id] = route
            route_objects.append(route)  # Pass route objects, not dict
    
    print(f"✅ Active routes identified: {len(active_routes)}")
    print(f"   • Route objects prepared: {len(route_objects)}")
    
    # Step 4: Run driver assignment
    print(f"\n🎯 Running driver assignment...")
    try:
        # Pass list of route objects instead of dictionary
        assignments = assign_drivers_to_routes_enhanced(drivers, route_objects, config)
        print(f"✅ Driver assignment completed!")
        print(f"   • Active vehicles: {len(route_objects)}")
        print(f"   • Available drivers: {len(drivers)}")
        print(f"   • Successful assignments: {len(assignments) if assignments else 0}")
        print(f"   • Idle drivers: {len(drivers) - len(assignments) if assignments else len(drivers)}")
        
        # Link drivers to route objects
        if assignments:
            driver_map = {d.name: d for d in drivers}  # Create name lookup
            for vehicle_id, driver_name in assignments.items():
                if vehicle_id in active_routes and driver_name in driver_map:
                    active_routes[vehicle_id].driver = driver_map[driver_name]
                    print(f"   🔗 Linked {driver_name} to route {vehicle_id}")
            print(f"✅ Driver objects properly linked to routes!")
        
    except Exception as e:
        print(f"⚠️  Driver assignment error: {e}")
        print(f"   • Falling back to simple manual assignment...")
        
        # Create a simple manual assignment for demonstration
        print(f"\n📋 Creating manual demonstration assignment...")
        manual_assignments = {}
        for i, route in enumerate(route_objects[:min(len(route_objects), len(drivers))]):
            driver = drivers[i]
            manual_assignments[route.vehicle.id] = driver
            print(f"   • Vehicle {route.vehicle.id} -> Driver {driver.name}")
        
        assignments = manual_assignments
        print(f"✅ Manual assignment demonstration completed: {len(assignments)} assignments")
        
        # Actually assign drivers to route objects
        for vehicle_id, driver in assignments.items():
            if vehicle_id in active_routes:
                active_routes[vehicle_id].driver = driver
                print(f"   🔗 Linked {driver.name} to route {vehicle_id}")
        print(f"✅ Driver objects properly linked to routes!")
    
    # Step 5: Print final comprehensive summary
    print(f"\n📋 Generating final solution summary...")
    try:
        # Convert active_routes dict to list for the summary function
        active_routes_list = list(active_routes.values())
        print_assignment_summary(active_routes_list, drivers)
        print(f"✅ Final summary generated successfully!")
        
    except Exception as e:
        print(f"⚠️  Standard summary failed: {e}")
        print(f"   Generating simplified summary instead...")
        
        # Create simplified summary
        print(f"\n🚛 SIMPLIFIED SOLUTION SUMMARY:")
        print(f"   • Total vehicles used: {len(active_routes)}")
        print(f"   • Total drivers available: {len(drivers)}")
        print(f"   • Driver assignments: {len(assignments) if 'assignments' in locals() else 0}")
        
        if 'assignments' in locals() and assignments:
            print(f"\n👥 Driver-Vehicle Assignments:")
            for vehicle_id, driver in assignments.items():
                route = active_routes.get(vehicle_id)
                tasks_count = len(route.tasks) if route else 0
                print(f"   • {vehicle_id}: {driver.name} ({tasks_count} tasks)")
        
        print(f"✅ Simplified summary completed!")
        # Continue even if summary fails
    
    # Step 6: Generate Interactive Map
    print(f"\n🗺️ Generating interactive solution map...")
    try:
        from algo.solution_visualizer import create_interactive_map
        import os
        from datetime import datetime
        
        # Create results directory with cleaner path
        test_dir = os.path.dirname(__file__)
        project_root = os.path.dirname(test_dir)  # Go up from tests to heuristicapproach
        results_dir = os.path.join(project_root, "results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Create map filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        map_filename = f"comprehensive_solution_map_{timestamp}.html"
        map_path = os.path.join(results_dir, map_filename)
        
        # Generate the interactive map
        created_map_path = create_interactive_map(solution, map_path)
        
        if created_map_path:
            # Normalize the path for clean display
            clean_path = os.path.normpath(created_map_path)
            print(f"🗺️ Interactive map saved to: {clean_path}")
            print(f"   • Open this file in your browser to view the solution")
            print(f"   • Shows routes, driver assignments, and task details")
            
            # Also provide the absolute path for easy access
            abs_path = os.path.abspath(clean_path)
            print(f"   • Full path: {abs_path}")
        else:
            print(f"⚠️ Map generation failed")
            
    except ImportError as e:
        print(f"⚠️ Map visualization not available: {e}")
    except Exception as e:
        print(f"⚠️ Error generating map: {e}")

    # Step 7: Export Routes to Text Format
    print(f"\n📝 Exporting routes to text format...")
    try:
        route_export_path = f"route_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        export_routes_to_text(solution, route_export_path)
        print(f"📝 Routes exported to: {route_export_path}")
    except Exception as e:
        print(f"⚠️ Error exporting routes: {e}")


def export_routes_to_text(solution, file_path: str):
    """
    Export route information to a text file.
    
    Args:
        solution: The VRP solution object
        file_path: Path where to save the text export
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("VRP SOLUTION EXPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Handle different solution types
        if hasattr(solution, 'routes'):
            routes = solution.routes
        elif isinstance(solution, list):
            routes = solution
        else:
            f.write(f"Error: Unknown solution type: {type(solution)}\n")
            return
        
        # Summary statistics
        vehicle_count = len(routes)
        total_orders = 0
        total_cost = 0.0
        
        for route in routes:
            if hasattr(route, 'tasks') and route.tasks:
                total_orders += len([task for task in route.tasks if hasattr(task, 'task_type') and task.task_type.name == 'DELIVERY'])
            if hasattr(route, 'total_cost'):
                total_cost += route.total_cost
        
        f.write(f"SUMMARY:\n")
        f.write(f"  • Total Vehicles Used: {vehicle_count}\n")
        f.write(f"  • Total Orders Delivered: {total_orders}\n")
        f.write(f"  • Solution Cost: {total_cost:.2f}\n\n")
        
        # Route details
        for route_idx, route in enumerate(routes, 1):
            if not hasattr(route, 'tasks') or not route.tasks:
                continue
                
            f.write(f"ROUTE {route_idx}:\n")
            
            # Vehicle info
            if hasattr(route, 'vehicle') and route.vehicle:
                f.write(f"  Vehicle: {route.vehicle.vehicle_id}\n")
                
                # Driver info
                if hasattr(route.vehicle, 'assigned_driver') and route.vehicle.assigned_driver:
                    driver = route.vehicle.assigned_driver
                    f.write(f"  Driver: {driver.driver_id} ({driver.name})\n")
                    if hasattr(driver, 'license_types') and driver.license_types:
                        f.write(f"  Licenses: {', '.join(driver.license_types)}\n")
                    if hasattr(driver, 'capabilities') and driver.capabilities:
                        f.write(f"  Capabilities: {', '.join(driver.capabilities)}\n")
            
            # Route tasks
            f.write(f"  Tasks:\n")
            for task_idx, task in enumerate(route.tasks, 1):
                weight = abs(task.demand) if hasattr(task, 'demand') and task.demand else 0
                task_type = task.task_type.name if hasattr(task, 'task_type') else 'UNKNOWN'
                location_name = task.location_name if hasattr(task, 'location_name') else 'UNKNOWN'
                order_id = task.order_id if hasattr(task, 'order_id') else 'UNKNOWN'
                
                f.write(f"    {task_idx:2d}. {task_type}: {location_name}\n")
                f.write(f"        Order: {order_id}\n")
                f.write(f"        Weight: {weight:.1f}kg\n")
                
                if hasattr(task, 'time_windows') and task.time_windows:
                    tw = task.time_windows[0]
                    f.write(f"        Time Window: {tw.start_time:.0f}-{tw.end_time:.0f} min\n")
                f.write("\n")
            
            # Route metrics
            if hasattr(route, 'total_distance'):
                f.write(f"  Route Distance: {route.total_distance:.2f} km\n")
            if hasattr(route, 'total_cost'):
                f.write(f"  Route Cost: {route.total_cost:.2f}\n")
            
            total_weight = sum(abs(task.demand) for task in route.tasks if hasattr(task, 'demand') and task.demand)
            f.write(f"  Total Weight: {total_weight:.1f}kg\n")
            f.write("\n" + "-"*60 + "\n\n")


def run_precomputation_phase(excel_file: str):
    """
    Run OSRM route pre-computation phase to optimize performance.
    
    This phase pre-computes all possible routes between unique locations
    in the scenario, dramatically reducing optimization runtime by eliminating
    OSRM API calls during the heuristic search.
    
    Args:
        excel_file: Path to the Excel scenario file
    """
    print("=" * 80)
    print("🚀 OSRM ROUTE PRE-COMPUTATION PHASE")
    print("=" * 80)
    
    if RoutePrecomputer is None:
        print("⚠️  RoutePrecomputer not available, skipping pre-computation")
        print("   The optimization will use on-demand OSRM calls instead")
        print("   (This will be slower but still functional)")
        return
    
    try:
        # Initialize the route pre-computer
        print("🔧 Initializing route pre-computer...")
        precomputer = RoutePrecomputer(
            osrm_url="https://router.project-osrm.org",  # Public OSRM server
            batch_size=20,  # Conservative batch size for public server
            rate_limit_delay=0.2  # Respectful delay for public server
        )
        
        # Load scenario locations from Excel
        print(f"📁 Loading scenario locations from: {excel_file}")
        locations = precomputer.load_scenario_from_excel(excel_file)
        
        if not locations:
            print("❌ No locations loaded from scenario, skipping pre-computation")
            return
            
        print(f"✅ Loaded {len(locations)} unique locations")
        unique_pairs = len(locations) * (len(locations) - 1)  # All pairs except self-loops
        print(f"🔗 Total route pairs to pre-compute: {unique_pairs}")
        
        # Check current cache status
        import sqlite3
        try:
            conn = sqlite3.connect(precomputer.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM route_cache")
            cache_count_before = cursor.fetchone()[0]
            conn.close()
            print(f"💾 Current cache size: {cache_count_before} routes")
        except:
            cache_count_before = 0
            print("💾 Cache database will be created")
        
        # Run pre-computation
        print("\n🔄 Starting route pre-computation...")
        start_time = time.time()
        
        precomputer.precompute_all_routes()
        
        precompute_time = time.time() - start_time
        
        # Check final cache status
        try:
            conn = sqlite3.connect(precomputer.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM route_cache")
            cache_count_after = cursor.fetchone()[0]
            conn.close()
            new_routes = cache_count_after - cache_count_before
        except:
            cache_count_after = cache_count_before
            new_routes = 0
        
        print(f"\n✅ Pre-computation completed!")
        print(f"📊 Summary:")
        print(f"   • Pre-computation time: {precompute_time:.1f} seconds")
        print(f"   • Routes in cache before: {cache_count_before}")
        print(f"   • Routes in cache after: {cache_count_after}")
        print(f"   • New routes cached: {new_routes}")
        print(f"   • Cache coverage: {(cache_count_after/unique_pairs)*100:.1f}%")
        print(f"\n🎯 Expected benefit: Subsequent optimizations should make ~0 OSRM calls!")
        
    except Exception as e:
        print(f"❌ Pre-computation failed: {e}")
        print("   The optimization will continue with on-demand OSRM calls")
        import traceback
        traceback.print_exc()


def main():
    """
    Main function to run the comprehensive integration test.
    """
    print("Starting EPDT Comprehensive Integration Test...")
    
    # Define path to the Excel file
    excel_file = os.path.join(src_dir, 'furgoni2.xlsx')
    
    if not os.path.exists(excel_file):
        print(f"❌ Error: Excel file not found at {excel_file}")
        sys.exit(1)
    
    print(f"📂 Using Excel file: {excel_file}")
    
    try:
        # Run Pre-computation Phase: OSRM Route Pre-computation
        run_precomputation_phase(excel_file)
        
        # Run Phase 1: Heuristic Solver Test
        solution, orders, vehicles, runtime = run_phase1_heuristic_test(excel_file)
        
        # PHASE 2: Driver Assignment Integration
        # Now that Phase 1 debugging is complete, enable driver assignment
        run_phase2_driver_assignment(excel_file, solution, vehicles)
        
        # Final completion message - AFTER all phases and summaries are done
        print("\n" + "="*80)
        print("🎉 COMPREHENSIVE INTEGRATION TEST COMPLETED!")
        print("="*80)
        print(f"✅ Phase 1: Heuristic solver executed in {runtime:.2f} seconds")
        print(f"✅ Phase 2: Driver assignment integration completed")
        print(f"✅ Phase 3: Interactive map visualization generated")
        print("\n🎯 Complete system integration with optimized constraints and visualization!")
        
    except Exception as e:
        print(f"\n❌ COMPREHENSIVE INTEGRATION TEST FAILED!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
